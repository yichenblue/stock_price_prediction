# Cross-Market Transformer

This directory contains a first-version PyTorch implementation for Hong Kong pre-open stock prediction using:

- Hong Kong history up to `t-1`
- U.S. history up to the most recent completed U.S. session before HK opens on day `t`
- `us_open_prev_night`
- `company_id`

Modules:

- `cross_market_transformer/model.py`
- `cross_market_transformer/baselines.py`
- `cross_market_transformer/data.py`
- `cross_market_transformer/trainer.py`
- `example_train.py`
- `minimal_config.py`
- `run_ablation.py`
- `run_shared_head.py`

Minimal usage:

```python
from cross_market_transformer import CrossMarketDataset, CrossMarketTransformerModel
from cross_market_transformer import ModelConfig, TrainConfig, Trainer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Expected sample fields:

- `x_hk`: `[num_samples, hk_seq_len, hk_input_dim]`
- `x_us`: `[num_samples, us_seq_len, us_input_dim]`
- `company_id`: `[num_samples]`
- `us_open_prev_night`: `[num_samples]`, values in `{0, 1}`
- `target`: shape depends on task

Notes:

- Splits are chronological only through `chronological_split`.
- Full-dataset training now scans `dataset/` automatically and uses all company pairs with `_Standardized.xlsx` inputs.
- Multi-company train/val/test splitting is done per company first, then merged across companies.
- Company-specific prediction heads sit on top of a shared backbone.
- Padding masks are supported for future variable-length sequence handling.
- The Excel loader supports your current files such as `09988_Factors_Standardized.xlsx` and `BABA_Factors_Standardized.xlsx`.
- The trainer now prints one log line per epoch and saves a train-vs-val plot if `matplotlib` is installed.
- The current default task is `regression_peak_trough`:
  - inputs exclude `target_peak` to avoid target leakage
  - target is `[r1, target_peak]`
  - model output is `[r1_pred, trough_logit, neutral_logit, peak_logit]`
  - `softmax(output[:, 1:4])` gives `[trough_prob, neutral_prob, peak_prob]`
  - `target_peak=0` means trough, `target_peak=1` means neutral, and `target_peak=2` means peak
- Other supported targets:
  - `regression_peak_trough`: predict HK `r1` and peak/trough class jointly
  - `regression`: predict raw standardized `r1`, and report `IC`, `MSE/RMSE`, and sign-based `accuracy`
  - `binary_classification`: predict whether `r1 > 0`
  - `multiclass_classification`: discretize `r1` with configured thresholds
- For HK day `t`, the sample uses HK rows up to `t-1` and US rows up to the latest US session with `us_date < hk_date`.
- Scheme A timing features are included:
  - `hk_time_delta`: natural-day distance from each HK token to target HK day
  - `us_time_delta`: natural-day distance from each US token to target HK day
  - `us_sessions_since_last_hk`: number of completed US sessions since the last HK observation
  - `latest_us_gap_days`: natural-day gap between target HK day and latest completed US session
- `USE_US_PREV_NIGHT=False` switches to a lagged-US variant:
  - the most recent completed US session is excluded
  - the US sequence ends one session earlier
  - `us_open_prev_night` is forced to 0 to avoid leaking previous-night information
- `run_ablation.py` compares three standard settings with the same data split and trainer:
  - `hk_us_concat`
  - `hk_transformer_only`
  - `cross_market_transformer`
- `run_shared_head.py` runs the shared-head generalization experiment:
  - all legacy-company samples are used for training
  - no validation split is used
  - `zai_lab` plus `noah` are used only as the held-out test set
- Paths are local by default and can be redirected in Colab by setting `PROJECT_ROOT`.
