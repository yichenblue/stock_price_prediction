from cross_market_transformer import ModelConfig, TrainConfig

DATASET_ROOT = "dataset"
HK_LOOKBACK = 30
US_LOOKBACK = 10
TARGET_COL = "r1"
USE_US_PREV_NIGHT = False

MODEL_CONFIG = ModelConfig(
    hk_input_dim=28,
    us_input_dim=28,
    num_companies=1,
    max_hk_len=HK_LOOKBACK,
    max_us_len=US_LOOKBACK,
    task_type="regression",
    num_classes=2,
    d_model=64,
    n_heads=4,
    num_layers_hk=2,
    num_layers_us=2,
    num_fusion_layers=1,
    head_hidden_dim=64,
    dropout=0.25,
)


TRAIN_CONFIG = TrainConfig(
    batch_size=32,
    num_epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    early_stopping_patience=5,
    checkpoint_dir="checkpoints",
    checkpoint_name="cross_market_transformer.pt",
    scheduler_type="plateau",
    plot_history=True,
    history_plot_name="cross_market_training_history.png",
)
