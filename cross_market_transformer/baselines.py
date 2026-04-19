from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .model import CompanySpecificHeads


def _resolve_output_dim(task_type: str, num_classes: int) -> int:
    if task_type == "regression":
        return 1
    if task_type == "regression_peak_trough":
        return 1 + num_classes
    if task_type == "binary_classification":
        return 1
    if task_type == "multiclass_classification":
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
        hk_padding_mask: torch.Tensor | None = None,
        us_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_us, us_padding_mask, hk_time_delta, us_time_delta, us_sessions_since_last_hk, latest_us_gap_days
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
    Pool HK and US histories independently, concatenate them, then predict.
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
        self.us_projection = nn.Sequential(
            nn.Linear(config.us_input_dim, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.session_embedding = nn.Embedding(2, config.d_model)
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
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
        hk_padding_mask: torch.Tensor | None = None,
        us_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hk_time_delta, us_time_delta, us_sessions_since_last_hk, latest_us_gap_days
        hk_hidden = self.hk_projection(x_hk)  # [batch, hk_len, d_model]
        us_hidden = self.us_projection(x_us)  # [batch, us_len, d_model]
        hk_z = masked_mean_pool(hk_hidden, hk_padding_mask)
        us_z = masked_mean_pool(us_hidden, us_padding_mask)
        z = self.fusion(torch.cat([hk_z, us_z], dim=-1))
        z = z + self.session_embedding(us_open_prev_night)
        logits = self.company_heads(z, company_id)
        if self.config.task_type == "regression":
            return logits.squeeze(-1)
        return logits
