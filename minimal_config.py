import os

from cross_market_transformer import ModelConfig, TrainConfig

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
HK_LOOKBACK = 30
US_LOOKBACK = 10
TARGET_COL = "r1"
USE_US_PREV_NIGHT = True
NORMALIZATION_MODE = "rolling"
ROLLING_NORMALIZATION_WINDOW = 252

MODEL_CONFIG = ModelConfig(
    hk_input_dim=29,
    us_input_dim=29,
    num_companies=1,
    max_hk_len=HK_LOOKBACK,
    max_us_len=US_LOOKBACK,
    task_type="regression_peak_trough",
    num_classes=3,
    d_model=64,
    n_heads=4,
    num_layers_hk=2,
    num_layers_us=2,
    num_fusion_layers=1,
    head_hidden_dim=64,
    dropout=0.25,
)


TRAIN_CONFIG = TrainConfig(
    batch_size=64,
    num_epochs=120,
    learning_rate=2e-4,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    early_stopping_patience=5,
    checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
    checkpoint_name="cross_market_transformer.pt",
    scheduler_type="plateau",
    class_weight=[5.0, 1.0, 5.0],
    plot_history=True,
    history_plot_name="cross_market_training_history.png",
)
