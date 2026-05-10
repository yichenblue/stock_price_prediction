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
P_INDEX_MODE = "feature"
P_INDEX_GAP_THRESHOLD = 0.02
R1_TASK_TYPE = "regression"
PEAK_TROUGH_TASK_TYPE = "peak_trough_classification"
JOINT_TARGET_TASK_TYPE = "regression_peak_trough"

MODEL_CONFIG = ModelConfig(
    hk_input_dim=29,
    us_input_dim=29,
    num_companies=1,
    max_hk_len=HK_LOOKBACK,
    max_us_len=US_LOOKBACK,
    task_type=R1_TASK_TYPE,
    num_classes=3,
    d_model=64,
    n_heads=4,
    num_layers_hk=2,
    num_layers_us=2,
    num_fusion_layers=1,
    head_hidden_dim=64,
    dropout=0.3,
    p_index_mode=P_INDEX_MODE,
    p_index_gap_feature_dim=3,
)


TRAIN_CONFIG = TrainConfig(
    batch_size=64,
    num_epochs=45,
    learning_rate=1e-4,
    weight_decay=5e-4,
    grad_clip_norm=1.0,
    early_stopping_patience=5,
    checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
    checkpoint_name="cross_market_shared_head_pindex_feature_r1.pt",
    scheduler_type="plateau",
    class_weight=[5.0, 1.0, 5.0],
    r1_loss_weight=50.0,
    peak_trough_loss_weight=1.0,
    plot_history=True,
    history_plot_name="cross_market_shared_head_pindex_feature_r1_history.png",
)
