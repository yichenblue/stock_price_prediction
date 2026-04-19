from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import SampleBatch


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score: float | None = None
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        task_type: str,
        num_classes: int = 2,
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = torch.device(train_config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.criterion = self._build_loss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.scheduler = self._build_scheduler()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        best_score = float("inf")
        best_score_name = "val_loss" if val_loader is not None else "train_loss"
        history: dict[str, list[dict[str, float]]] = {"train": []}
        if val_loader is not None:
            history["val"] = []
        if test_loader is not None:
            history["test"] = []
        checkpoint_path = self.train_config.checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        initial_train_result = self._run_epoch(train_loader, training=False)
        initial_val_result = self._run_epoch(val_loader, training=False) if val_loader is not None else None
        initial_test_result = self._run_epoch(test_loader, training=False) if test_loader is not None else None
        history["train"].append({"loss": initial_train_result.loss, **initial_train_result.metrics})
        if initial_val_result is not None:
            history["val"].append({"loss": initial_val_result.loss, **initial_val_result.metrics})
        if initial_test_result is not None:
            history["test"].append({"loss": initial_test_result.loss, **initial_test_result.metrics})
        self._log_epoch(
            0,
            initial_train_result,
            initial_val_result,
            train_optimization_loss=None,
            test_result=initial_test_result,
        )

        initial_score = initial_val_result.loss if initial_val_result is not None else initial_train_result.loss
        best_score = initial_score
        self._save_checkpoint(checkpoint_path, 0, best_score)

        for epoch in range(1, self.train_config.num_epochs + 1):
            train_optimization_result = self._run_epoch(train_loader, training=True)
            train_result = self._run_epoch(train_loader, training=False)
            val_result = self._run_epoch(val_loader, training=False) if val_loader is not None else None
            test_result = self._run_epoch(test_loader, training=False) if test_loader is not None else None
            history["train"].append({"loss": train_result.loss, **train_result.metrics})
            if val_result is not None:
                history["val"].append({"loss": val_result.loss, **val_result.metrics})
            if test_result is not None:
                history["test"].append({"loss": test_result.loss, **test_result.metrics})
            self._log_epoch(
                epoch,
                train_result,
                val_result,
                train_optimization_result.loss,
                test_result=test_result,
            )

            scheduler_score = val_result.loss if val_result is not None else train_result.loss
            self._step_scheduler(scheduler_score)

            current_score = val_result.loss if val_result is not None else train_result.loss
            if current_score < best_score:
                best_score = current_score
                self._save_checkpoint(checkpoint_path, epoch, best_score)

        if self.train_config.plot_history:
            self._save_history_plot(history, self.train_config.history_plot_path())
        self.load_checkpoint(checkpoint_path)
        return {"best_score": best_score, "best_score_name": best_score_name, "history": history}

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        result = self._run_epoch(data_loader, training=False)
        return {"loss": result.loss, **result.metrics}

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                logits = self.model(
                    x_hk=batch.x_hk,
                    x_us=batch.x_us,
                    hk_time_delta=batch.hk_time_delta,
                    us_time_delta=batch.us_time_delta,
                    company_id=batch.company_id,
                    us_open_prev_night=batch.us_open_prev_night,
                    us_sessions_since_last_hk=batch.us_sessions_since_last_hk,
                    latest_us_gap_days=batch.latest_us_gap_days,
                    hk_padding_mask=batch.hk_padding_mask,
                    us_padding_mask=batch.us_padding_mask,
                )
                outputs.append(logits.detach().cpu())
        return torch.cat(outputs, dim=0)

    def predict_peak_trough_probabilities(self, data_loader: DataLoader) -> dict[str, torch.Tensor]:
        raw = self.predict(data_loader)
        if self.task_type != "regression_peak_trough":
            raise ValueError("predict_peak_trough_probabilities requires task_type='regression_peak_trough'.")
        probs = torch.softmax(raw[:, 1:4], dim=-1)
        return {
            "r1_pred": raw[:, 0],
            "trough_prob": probs[:, 0],
            "neutral_prob": probs[:, 1],
            "peak_prob": probs[:, 2],
            "peak_trough_probs": probs,
        }

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])

    def _run_epoch(self, data_loader: DataLoader, training: bool) -> EpochResult:
        self.model.train(training)
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []

        for step, batch in enumerate(data_loader, start=1):
            batch = batch.to(self.device)
            logits = self.model(
                x_hk=batch.x_hk,
                x_us=batch.x_us,
                hk_time_delta=batch.hk_time_delta,
                us_time_delta=batch.us_time_delta,
                company_id=batch.company_id,
                us_open_prev_night=batch.us_open_prev_night,
                us_sessions_since_last_hk=batch.us_sessions_since_last_hk,
                latest_us_gap_days=batch.latest_us_gap_days,
                hk_padding_mask=batch.hk_padding_mask,
                us_padding_mask=batch.us_padding_mask,
            )
            loss = self._compute_loss(logits, batch.target)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip_norm)
                self.optimizer.step()

            batch_size = batch.x_hk.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions.append(logits.detach().cpu())
            targets.append(batch.target.detach().cpu())

            if training and step % self.train_config.log_every_n_steps == 0:
                pass

        preds = torch.cat(predictions, dim=0)
        gold = torch.cat(targets, dim=0)
        metrics = self._compute_metrics(preds, gold)
        avg_loss = total_loss / max(total_samples, 1)
        return EpochResult(loss=avg_loss, metrics=metrics)

    def _build_loss(self) -> nn.Module:
        if self.task_type == "regression":
            return nn.MSELoss()
        if self.task_type == "regression_peak_trough":
            return nn.MSELoss()
        if self.task_type == "binary_classification":
            pos_weight = None
            if self.train_config.class_weight is not None:
                if len(self.train_config.class_weight) != 2:
                    raise ValueError("Binary class_weight must contain exactly two values.")
                neg_w, pos_w = self.train_config.class_weight
                pos_weight = torch.tensor([pos_w / max(neg_w, 1e-12)], dtype=torch.float32, device=self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if self.task_type == "multiclass_classification":
            weight = None
            if self.train_config.class_weight is not None:
                weight = torch.tensor(self.train_config.class_weight, dtype=torch.float32, device=self.device)
            return nn.CrossEntropyLoss(weight=weight)
        raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _build_scheduler(self):
        if self.train_config.scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_config.scheduler_factor,
                patience=self.train_config.scheduler_patience,
                min_lr=self.train_config.min_learning_rate,
            )
        if self.train_config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.num_epochs,
                eta_min=self.train_config.min_learning_rate,
            )
        raise ValueError(f"Unsupported scheduler_type: {self.train_config.scheduler_type}")

    def _step_scheduler(self, val_loss: float) -> None:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task_type == "regression":
            return self.criterion(logits.float(), target.float().view_as(logits))
        if self.task_type == "regression_peak_trough":
            r1_loss = self.criterion(logits[:, 0].float(), target[:, 0].float())
            class_weight = None
            if self.train_config.class_weight is not None:
                if len(self.train_config.class_weight) != 3:
                    raise ValueError("regression_peak_trough class_weight must contain exactly three values.")
                class_weight = torch.tensor(self.train_config.class_weight, dtype=torch.float32, device=self.device)
            peak_trough_loss = nn.functional.cross_entropy(
                logits[:, 1:4].float(),
                target[:, 1].long(),
                weight=class_weight,
            )
            return 0.5 * r1_loss + peak_trough_loss
        if self.task_type == "binary_classification":
            return self.criterion(logits.float().view(-1), target.float().view(-1))
        return self.criterion(logits.float(), target.long().view(-1))

    def _compute_metrics(self, preds: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        if self.task_type == "regression":
            preds = preds.view(-1).float()
            target = target.view(-1).float()
            mse = torch.mean((preds - target) ** 2).item()
            mae = torch.mean(torch.abs(preds - target)).item()
            rmse = mse ** 0.5
            ic = _information_coefficient(preds, target)
            sign_accuracy = ((preds >= 0) == (target >= 0)).float().mean().item()
            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "ic": ic,
                "sign_accuracy": sign_accuracy,
            }

        if self.task_type == "regression_peak_trough":
            r1_preds = preds[:, 0].float()
            r1_target = target[:, 0].float()
            r1_mse = torch.mean((r1_preds - r1_target) ** 2).item()
            r1_mae = torch.mean(torch.abs(r1_preds - r1_target)).item()
            r1_rmse = r1_mse ** 0.5
            r1_ic = _information_coefficient(r1_preds, r1_target)
            r1_sign_accuracy = ((r1_preds >= 0) == (r1_target >= 0)).float().mean().item()

            peak_trough_logits = preds[:, 1:4].float()
            peak_trough_labels = torch.argmax(peak_trough_logits, dim=-1)
            peak_trough_target = target[:, 1].long()
            class_metrics = _classification_metrics(
                peak_trough_labels,
                peak_trough_target,
                num_classes=3,
                class_names=("trough", "neutral", "peak"),
            )
            peak_trough_probs = torch.softmax(peak_trough_logits, dim=-1)
            threshold_metrics = {}
            threshold_metrics.update(
                _binary_event_metrics(
                    scores=peak_trough_probs[:, 2],
                    target=peak_trough_target == 2,
                    threshold=0.5,
                    prefix="peak_thr50",
                )
            )
            threshold_metrics.update(
                _binary_event_metrics(
                    scores=peak_trough_probs[:, 0],
                    target=peak_trough_target == 0,
                    threshold=0.5,
                    prefix="trough_thr50",
                )
            )
            return {
                "r1_mse": r1_mse,
                "r1_rmse": r1_rmse,
                "r1_mae": r1_mae,
                "r1_ic": r1_ic,
                "r1_sign_accuracy": r1_sign_accuracy,
                **class_metrics,
                **threshold_metrics,
            }

        if self.task_type == "binary_classification":
            probs = torch.sigmoid(preds.view(-1))
            labels = (probs >= 0.5).long()
            target = target.view(-1).long()
            return _classification_metrics(labels, target, num_classes=2)

        labels = torch.argmax(preds, dim=-1)
        target = target.view(-1).long()
        metrics = _classification_metrics(labels, target, num_classes=self.num_classes)
        return metrics

    def _save_checkpoint(self, path: Path, epoch: int, best_val_loss: float) -> None:
        payload = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path)

    def _log_epoch(
        self,
        epoch: int,
        train_result: EpochResult,
        val_result: EpochResult | None,
        train_optimization_loss: float | None = None,
        test_result: EpochResult | None = None,
    ) -> None:
        train_metrics = self._format_metrics(train_result.metrics)
        optimization_loss_str = ""
        if train_optimization_loss is not None:
            optimization_loss_str = f" train_optim_loss={train_optimization_loss:.6f} |"
        val_str = ""
        if val_result is not None:
            val_metrics = self._format_metrics(val_result.metrics)
            val_str = f" | val_loss={val_result.loss:.6f} | {val_metrics}"
        test_str = ""
        if test_result is not None:
            test_metrics = self._format_metrics(test_result.metrics)
            test_str = f" | test_loss={test_result.loss:.6f} | {test_metrics}"
        print(
            f"Epoch {epoch:03d}/{self.train_config.num_epochs:03d} | "
            f"{optimization_loss_str} train_loss={train_result.loss:.6f} | {train_metrics} | "
            f"{val_str.lstrip()}{test_str}"
        )

    @staticmethod
    def _format_metrics(metrics: dict[str, float]) -> str:
        return " ".join(f"{name}={value:.4f}" for name, value in metrics.items())

    def _save_history_plot(self, history: dict[str, list[dict[str, float]]], path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "Skipping history plot because matplotlib is not installed. "
                f"Install it to save {path}."
            )
            return

        train_history = history["train"]
        val_history = history.get("val")
        test_history = history.get("test")
        epochs = list(range(len(train_history)))

        metric_names = self._plot_metric_names(train_history[0])
        n_metrics = len(metric_names)
        ncols = 2
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for axis, metric_name in zip(axes, metric_names):
            train_values = [entry[metric_name] for entry in train_history]
            axis.plot(epochs, train_values, label="train", marker="o")
            if val_history is not None:
                val_values = [entry[metric_name] for entry in val_history]
                axis.plot(epochs, val_values, label="val", marker="o")
            if test_history is not None:
                test_values = [entry[metric_name] for entry in test_history]
                axis.plot(epochs, test_values, label="test", marker="o")
            axis.set_title(metric_name)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(metric_name)
            axis.legend()
            axis.grid(True, alpha=0.3)

        for axis in axes[n_metrics:]:
            axis.axis("off")

        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training history plot to {path}")

    def _primary_metric_name(self, history_entry: dict[str, float]) -> str | None:
        metric_keys = [key for key in history_entry.keys() if key != "loss"]
        if not metric_keys:
            return None
        if self.task_type == "regression":
            for preferred in ("rmse", "mae", "mse"):
                if preferred in metric_keys:
                    return preferred
        for preferred in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            if preferred in metric_keys:
                return preferred
        return metric_keys[0]

    def _plot_metric_names(self, history_entry: dict[str, float]) -> list[str]:
        if self.task_type == "regression":
            names = []
            for metric_name in ("ic", "sign_accuracy", "mse", "mae"):
                if metric_name in history_entry:
                    names.append(metric_name)
            return names
        if self.task_type == "regression_peak_trough":
            names = []
            for metric_name in (
                "r1_ic",
                "r1_sign_accuracy",
                "r1_mse",
                "peak_precision",
                "peak_recall",
                "peak_f1",
                "trough_precision",
                "trough_recall",
                "trough_f1",
                "peak_thr50_precision",
                "peak_thr50_recall",
                "peak_thr50_f1",
                "trough_thr50_precision",
                "trough_thr50_recall",
                "trough_thr50_f1",
            ):
                if metric_name in history_entry:
                    names.append(metric_name)
            return names

        names = ["loss"]
        for metric_name in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            if metric_name in history_entry:
                names.append(metric_name)
        return names


def _classification_metrics(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, float]:
    accuracy = (preds == target).float().mean().item()
    precision_scores = []
    recall_scores = []
    f1_scores = []
    per_class_metrics = {}

    for cls in range(num_classes):
        tp = ((preds == cls) & (target == cls)).sum().item()
        fp = ((preds == cls) & (target != cls)).sum().item()
        fn = ((preds != cls) & (target == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        if class_names is not None:
            class_name = class_names[cls]
            per_class_metrics[f"{class_name}_precision"] = precision
            per_class_metrics[f"{class_name}_recall"] = recall
            per_class_metrics[f"{class_name}_f1"] = f1

    metrics = {
        "accuracy": accuracy,
        "precision_macro": sum(precision_scores) / num_classes,
        "recall_macro": sum(recall_scores) / num_classes,
        "f1_macro": sum(f1_scores) / num_classes,
    }
    metrics.update(per_class_metrics)
    return metrics


def _binary_event_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    prefix: str,
) -> dict[str, float]:
    preds = scores >= threshold
    target = target.bool()
    tp = (preds & target).sum().item()
    fp = (preds & ~target).sum().item()
    fn = (~preds & target).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    signal_rate = preds.float().mean().item()
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_signal_rate": signal_rate,
    }


def _information_coefficient(preds: torch.Tensor, target: torch.Tensor) -> float:
    preds_centered = preds - preds.mean()
    target_centered = target - target.mean()
    denom = torch.sqrt((preds_centered ** 2).sum()) * torch.sqrt((target_centered ** 2).sum())
    if torch.isclose(denom, torch.tensor(0.0, device=denom.device)):
        return 0.0
    return ((preds_centered * target_centered).sum() / denom).item()
