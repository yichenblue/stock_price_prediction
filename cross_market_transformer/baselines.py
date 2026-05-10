from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .model import CompanySpecificHeads, HKEncoder, PreOpenAggregator, SharedHead, USEncoder


def _resolve_output_dim(task_type: str, num_classes: int) -> int:
    if task_type == "regression":
        return 1
    if task_type == "regression_peak_trough":
        if num_classes != 3:
            raise ValueError("regression_peak_trough requires num_classes=3.")
        return 1 + num_classes
    if task_type == "peak_trough_classification":
        if num_classes != 3:
            raise ValueError("peak_trough_classification requires num_classes=3.")
        return num_classes
    if task_type == "binary_classification":
        return 1
    if task_type == "multiclass_classification":
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for multiclass classification.")
        return num_classes
    raise ValueError(f"Unsupported task_type: {task_type}")


def masked_mean_pool(x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    # x: [batch, seq_len, hidden_dim]
    if padding_mask is None:
        return x.mean(dim=1)
    valid = (~padding_mask).unsqueeze(-1).float()
    summed = (x * valid).sum(dim=1)
    counts = valid.sum(dim=1).clamp_min(1.0)
    return summed / counts


class HKOnlyBaseline(nn.Module):
    """
    Baseline 1:
    Use only Hong Kong history with simple mean pooling and a small MLP.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        output_dim = _resolve_output_dim(config.task_type, config.num_classes)
        self.hk_projection = nn.Sequential(
            nn.Linear(config.hk_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.session_embedding = nn.Embedding(2, config.d_model)
        self.p_index_gap_encoder = None
        self.company_heads = CompanySpecificHeads(
            num_companies=config.num_companies,
            input_dim=config.d_model,
            hidden_dim=config.head_hidden_dim,
            output_dim=output_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x_hk: torch.Tensor,
        x_us: torch.Tensor,
        hk_time_delta: torch.Tensor,
        us_time_delta: torch.Tensor,
        company_id: torch.Tensor,
        us_open_prev_night: torch.Tensor,
        us_sessions_since_last_hk: torch.Tensor,
        latest_us_gap_days: torch.Tensor,
        p_index_gap_features: torch.Tensor | None = None,
        hk_padding_mask: torch.Tensor | None = None,
        us_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_us, us_padding_mask, hk_time_delta, us_time_delta, us_sessions_since_last_hk, latest_us_gap_days, p_index_gap_features
        hk_hidden = self.hk_projection(x_hk)  # [batch, hk_len, d_model]
        z = masked_mean_pool(hk_hidden, hk_padding_mask)
        z = z + self.session_embedding(us_open_prev_night)
        logits = self.company_heads(z, company_id)
        if self.config.task_type == "regression":
            return logits.squeeze(-1)
        return logits


class HKUSConcatBaseline(nn.Module):
    """
    Baseline 2:
    Keep the main HK/US Transformer encoders and shared head, but remove
    cross-market attention. The pre-open query attends over concatenated
    HK and US encoded tokens directly.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        output_dim = _resolve_output_dim(config.task_type, config.num_classes)
        self.hk_encoder = HKEncoder(
            input_dim=config.hk_input_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers_hk,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_hk_len,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.us_encoder = USEncoder(
            input_dim=config.us_input_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers_us,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_us_len,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.pre_open_aggregator = PreOpenAggregator(
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_companies=config.num_companies,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            use_p_index_gap_gate=_uses_p_index_gap_gate(config),
            p_index_gap_feature_dim=config.p_index_gap_feature_dim,
        )
        self.shared_head = SharedHead(
            input_dim=config.d_model,
            hidden_dim=config.head_hidden_dim,
            output_dim=output_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x_hk: torch.Tensor,
        x_us: torch.Tensor,
        hk_time_delta: torch.Tensor,
        us_time_delta: torch.Tensor,
        company_id: torch.Tensor,
        us_open_prev_night: torch.Tensor,
        us_sessions_since_last_hk: torch.Tensor,
        latest_us_gap_days: torch.Tensor,
        p_index_gap_features: torch.Tensor | None = None,
        hk_padding_mask: torch.Tensor | None = None,
        us_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hk_encoded = self.hk_encoder(x_hk, time_delta=hk_time_delta, padding_mask=hk_padding_mask)
        us_encoded = self.us_encoder(x_us, time_delta=us_time_delta, padding_mask=us_padding_mask)
        encoded = torch.cat([hk_encoded, us_encoded], dim=1)  # [batch, hk_len + us_len, d_model]
        padding_mask = _concat_padding_masks(
            hk_padding_mask,
            us_padding_mask,
            hk_len=hk_encoded.size(1),
            us_len=us_encoded.size(1),
            device=encoded.device,
        )
        z = self.pre_open_aggregator(
            fused_hk=encoded,
            company_id=company_id,
            us_open_prev_night=us_open_prev_night,
            us_sessions_since_last_hk=us_sessions_since_last_hk,
            latest_us_gap_days=latest_us_gap_days,
            p_index_gap_features=p_index_gap_features,
            hk_padding_mask=padding_mask,
        )
        logits = self.shared_head(z)
        if self.config.task_type == "regression":
            return logits.squeeze(-1)
        return logits


def _uses_p_index_gap_gate(config: ModelConfig) -> bool:
    if config.p_index_mode not in {"feature", "none", "gap_gate", "feature_plus_gap"}:
        raise ValueError("config.p_index_mode must be one of: 'feature', 'none', 'gap_gate', 'feature_plus_gap'.")
    return config.p_index_mode in {"gap_gate", "feature_plus_gap"}


def _concat_padding_masks(
    hk_padding_mask: torch.Tensor | None,
    us_padding_mask: torch.Tensor | None,
    hk_len: int,
    us_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if hk_padding_mask is None and us_padding_mask is None:
        return None
    if hk_padding_mask is None:
        batch_size = us_padding_mask.size(0)
        hk_padding_mask = torch.zeros(batch_size, hk_len, dtype=torch.bool, device=device)
    if us_padding_mask is None:
        batch_size = hk_padding_mask.size(0)
        us_padding_mask = torch.zeros(batch_size, us_len, dtype=torch.bool, device=device)
    return torch.cat([hk_padding_mask.to(device), us_padding_mask.to(device)], dim=1)
