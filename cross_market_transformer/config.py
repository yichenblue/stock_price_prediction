from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    hk_input_dim: int
    us_input_dim: int
    num_companies: int
    max_hk_len: int
    max_us_len: int
    task_type: str = "binary_classification"
    num_classes: int = 2
    d_model: int = 64
    n_heads: int = 4
    dim_feedforward: int = 256
    num_layers_hk: int = 2
    num_layers_us: int = 2
    num_fusion_layers: int = 1
    head_hidden_dim: int = 64
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 30
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_model.pt"
    scheduler_type: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_learning_rate: float = 1e-6
    device: Optional[str] = None
    log_every_n_steps: int = 20
    save_best_only: bool = True
    class_weight: Optional[list[float]] = field(default=None)
    plot_history: bool = True
    history_plot_name: str = "training_history.png"

    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.checkpoint_name

    def history_plot_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.history_plot_name
