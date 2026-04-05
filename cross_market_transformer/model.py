from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, d_model)
        self.time_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        time_delta: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x = self.feature_projection(x)  # [batch, seq_len, d_model]
        time_delta = torch.log1p(time_delta.float()).unsqueeze(-1)  # [batch, seq_len, 1]
        x = x + self.time_projection(time_delta)
        x = self.input_norm(x)
        x = self.positional_encoding(x)
        return self.transformer(x, src_key_padding_mask=padding_mask)


class HKEncoder(SequenceEncoder):
    pass


class USEncoder(SequenceEncoder):
    pass


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        hk_encoded: torch.Tensor,
        us_encoded: torch.Tensor,
        hk_padding_mask: torch.Tensor | None = None,
        us_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hk_encoded: [batch, hk_len, d_model]
        # us_encoded: [batch, us_len, d_model]
        attn_output, _ = self.cross_attn(
            query=hk_encoded,
            key=us_encoded,
            value=us_encoded,
            key_padding_mask=us_padding_mask,
            need_weights=False,
        )
        x = self.norm1(hk_encoded + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        if hk_padding_mask is not None:
            x = x.masked_fill(hk_padding_mask.unsqueeze(-1), 0.0)
        return x


class CrossMarketFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        layer_norm_eps: float,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hk_encoded: torch.Tensor,
        us_encoded: torch.Tensor,
        hk_padding_mask: torch.Tensor | None = None,
        us_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = hk_encoded
        for layer in self.layers:
            x = layer(
                hk_encoded=x,
                us_encoded=us_encoded,
                hk_padding_mask=hk_padding_mask,
                us_padding_mask=us_padding_mask,
            )
        return x


class PreOpenAggregator(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_companies: int,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.pre_open_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.company_embedding = nn.Embedding(num_companies, d_model)
        self.session_embedding = nn.Embedding(2, d_model)
        self.global_timing_projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        fused_hk: torch.Tensor,
        company_id: torch.Tensor,
        us_open_prev_night: torch.Tensor,
        us_sessions_since_last_hk: torch.Tensor,
        latest_us_gap_days: torch.Tensor,
        hk_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # fused_hk: [batch, hk_len, d_model]
        batch_size = fused_hk.size(0)
        query = self.pre_open_query.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        query = query + self.company_embedding(company_id).unsqueeze(1)
        query = query + self.session_embedding(us_open_prev_night).unsqueeze(1)
        timing_features = torch.stack(
            [
                torch.log1p(us_sessions_since_last_hk.float()),
                torch.log1p(latest_us_gap_days.float()),
            ],
            dim=-1,
        )  # [batch, 2]
        query = query + self.global_timing_projection(timing_features).unsqueeze(1)

        attended, _ = self.attn(
            query=query,
            key=fused_hk,
            value=fused_hk,
            key_padding_mask=hk_padding_mask,
            need_weights=False,
        )
        latent = self.norm(query + self.dropout(attended))
        return latent[:, 0, :]  # [batch, d_model]


class CompanySpecificHeads(nn.Module):
    def __init__(
        self,
        num_companies: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_companies)
            ]
        )

    def forward(self, z: torch.Tensor, company_id: torch.Tensor) -> torch.Tensor:
        # z: [batch, d_model]
        all_outputs = torch.stack([head(z) for head in self.heads], dim=1)  # [batch, num_companies, output_dim]
        gather_index = company_id.view(-1, 1, 1).expand(-1, 1, self.output_dim)
        outputs = torch.gather(all_outputs, dim=1, index=gather_index).squeeze(1)
        return outputs


class SharedHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [batch, d_model]
        return self.head(z)


class CrossMarketTransformerModel(nn.Module):
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
        self.cross_market_fusion = CrossMarketFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            num_layers=config.num_fusion_layers,
        )
        self.pre_open_aggregator = PreOpenAggregator(
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_companies=config.num_companies,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.company_specific_heads = CompanySpecificHeads(
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
        hk_encoded = self.hk_encoder(x_hk, time_delta=hk_time_delta, padding_mask=hk_padding_mask)
        us_encoded = self.us_encoder(x_us, time_delta=us_time_delta, padding_mask=us_padding_mask)

        fused_hk = self.cross_market_fusion(
            hk_encoded=hk_encoded,
            us_encoded=us_encoded,
            hk_padding_mask=hk_padding_mask,
            us_padding_mask=us_padding_mask,
        )
        z = self.pre_open_aggregator(
            fused_hk=fused_hk,
            company_id=company_id,
            us_open_prev_night=us_open_prev_night,
            us_sessions_since_last_hk=us_sessions_since_last_hk,
            latest_us_gap_days=latest_us_gap_days,
            hk_padding_mask=hk_padding_mask,
        )
        logits = self.company_specific_heads(z, company_id)
        if self.config.task_type == "regression":
            return logits.squeeze(-1)
        return logits


class CrossMarketTransformerSharedHeadModel(nn.Module):
    """
    Ablation model:
    keep the full cross-market backbone, but replace company-specific
    heads with one shared prediction head across all companies.
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
        self.cross_market_fusion = CrossMarketFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            num_layers=config.num_fusion_layers,
        )
        self.pre_open_aggregator = PreOpenAggregator(
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_companies=config.num_companies,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
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
        hk_padding_mask: torch.Tensor | None = None,
        us_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hk_encoded = self.hk_encoder(x_hk, time_delta=hk_time_delta, padding_mask=hk_padding_mask)
        us_encoded = self.us_encoder(x_us, time_delta=us_time_delta, padding_mask=us_padding_mask)

        fused_hk = self.cross_market_fusion(
            hk_encoded=hk_encoded,
            us_encoded=us_encoded,
            hk_padding_mask=hk_padding_mask,
            us_padding_mask=us_padding_mask,
        )
        z = self.pre_open_aggregator(
            fused_hk=fused_hk,
            company_id=company_id,
            us_open_prev_night=us_open_prev_night,
            us_sessions_since_last_hk=us_sessions_since_last_hk,
            latest_us_gap_days=latest_us_gap_days,
            hk_padding_mask=hk_padding_mask,
        )
        logits = self.shared_head(z)
        if self.config.task_type == "regression":
            return logits.squeeze(-1)
        return logits


class HKTransformerOnlyModel(nn.Module):
    """
    Strong ablation baseline:
    keep the HK Transformer encoder and pre-open aggregation,
    but remove the US stream and cross-market fusion entirely.
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
        self.pre_open_aggregator = PreOpenAggregator(
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_companies=config.num_companies,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.company_specific_heads = CompanySpecificHeads(
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
        del x_us, us_padding_mask, us_time_delta
        hk_encoded = self.hk_encoder(x_hk, time_delta=hk_time_delta, padding_mask=hk_padding_mask)
        z = self.pre_open_aggregator(
            fused_hk=hk_encoded,
            company_id=company_id,
            us_open_prev_night=us_open_prev_night,
            us_sessions_since_last_hk=us_sessions_since_last_hk,
            latest_us_gap_days=latest_us_gap_days,
            hk_padding_mask=hk_padding_mask,
        )
        logits = self.company_specific_heads(z, company_id)
        if self.config.task_type == "regression":
            return logits.squeeze(-1)
        return logits


def _resolve_output_dim(task_type: str, num_classes: int) -> int:
    if task_type == "regression":
        return 1
    if task_type == "binary_classification":
        return 1
    if task_type == "multiclass_classification":
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for multiclass classification.")
        return num_classes
    raise ValueError(f"Unsupported task_type: {task_type}")
