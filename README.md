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
- `run_p_index_ablation.py`
- `run_shared_head.py`

Minimal usage:

```python
from cross_market_transformer import CrossMarketDataset, CrossMarketTransformerSharedHeadModel
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
- `p_index_gap_features`: `[num_samples, 3 or 4]`, auxiliary P-index features used by `gap_gate` and `feature_plus_gap`
- `target`: shape depends on task

Notes:

- Splits are chronological only through `chronological_split`.
- Full-dataset training now scans `dataset/` automatically and uses all company pairs with `_Cleaned.xlsx` inputs.
- Multi-company train/val/test splitting is done per company first, then merged across companies.
- The default training entrypoint uses a shared prediction head on top of the cross-market backbone.
- The default main setting trains two separate models with the same shared-head cross-market architecture:
  - `r1`: predicts raw HK next-day `r1`; reports `IC`, `MSE/RMSE/MAE`, and sign accuracy.
  - `peak_trough`: predicts two independent event probabilities `[P(peak), P(trough)]` with sigmoid heads.
- The two default tasks use separate training hyperparameters:
  - `r1`: `num_epochs=60`, `learning_rate=1e-4`, `weight_decay=7e-4`, `dropout=0.3`
  - `peak_trough`: `num_epochs=35`, `learning_rate=5e-5`, `weight_decay=1e-3`, `dropout=0.35`, `class_weight=[6.0, 5.0]` as `[peak_pos_weight, trough_pos_weight]`
- The default main model uses `P_index` as a normal HK/US sequence feature.
- Company-specific heads are still available in `CrossMarketTransformerModel`.
- Padding masks are supported for future variable-length sequence handling.
- The default normalization mode is leak-free rolling normalization:
  - `NORMALIZATION_MODE="rolling"`
  - `ROLLING_NORMALIZATION_WINDOW=252`
  - HK input windows are normalized with HK history available up to `t-1`
  - US input windows are normalized with US history available up to the latest usable US session
- The Excel loader supports your current files such as `09988.HK_Cleaned.xlsx` and `BABA_Cleaned.xlsx`.
- The trainer now prints one log line per epoch and saves a train-vs-val plot if `matplotlib` is installed.
- `target_peak=0` means trough, `target_peak=1` means neutral, and `target_peak=2` means peak.
- `regression_peak_trough` is retained as a joint-target data format for compatibility, but the main scripts now retarget it into two independent single-task datasets before training.
- `P_index` is configurable through `P_INDEX_MODE`:
  - `feature`: use `P_index` as a normal HK/US sequence feature
  - `none`: remove `P_index` entirely
  - `gap_gate`: remove `P_index` from HK/US sequence features, compute `HK P_index(t-1) - US P_index(latest usable US session)`, and inject thresholded discrepancy features into the pre-open query
  - `feature_plus_gap`: use `P_index` as a normal HK/US sequence feature, and also inject `[HK latest P_index, US latest P_index, gap, abs(gap)]` into the pre-open query
  - The default is `feature`, so the default HK/US input dimension includes `P_index`.
  - HK-only baselines accept the batch field but do not use auxiliary cross-market P-index features, so they remain HK-only.
- Other supported targets:
  - `regression_peak_trough`: legacy joint data format that stores `[r1, target_peak]`; model output is `[r1, peak_logit, trough_logit]`
  - `peak_trough_classification`: predict `target_peak` as two independent event logits `[peak_logit, trough_logit]`
  - `regression`: predict raw `r1`, and report `IC`, `MSE/RMSE`, and sign-based `accuracy`
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
- `run_ablation.py` compares two structural ablations. Each ablation trains one `r1` model and one `peak_trough` model:
  - `random_walk`: non-trained baseline; predicts zero next-day return and neutral peak/trough state
  - `hk_us_concat`: main-style HK/US Transformer encoders plus shared head, but no cross-attention; the pre-open query attends over concatenated HK/US tokens
  - `hk_transformer_only`
- `run_p_index_ablation.py` compares:
  - A. `P_index` as a normal feature
  - B. no `P_index`
  - C. `P_index` as a normal feature plus soft discrepancy features
  - Each P-index variant trains one `r1` model and one `peak_trough` model with the shared-head cross-market architecture.
- `run_shared_head.py` runs the shared-head generalization experiment:
  - all legacy-company samples are used for training
  - no validation split is used
  - `zai_lab` plus `noah` are used only as the held-out test set
  - one `r1` model and one `peak_trough` model are trained separately
- Paths are local by default and can be redirected in Colab by setting `PROJECT_ROOT`.
