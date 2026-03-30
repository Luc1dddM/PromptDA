"""Trainer utilities for PromptDA fine-tuning and evaluation."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from promptda.utils.logger import Log
from training.loss import ScaleAndShiftInvariantLoss
from training.metrics import aggregate_metrics, compute_depth_metrics


class Trainer:
    """Owns train/eval loops, checkpointing, and metric plotting."""

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        scheduler,
        device: torch.device,
        ckpt_dir: str,
        wandb_run: Any = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.loss_fn = ScaleAndShiftInvariantLoss()
        self.best_abs_rel = float("inf")
        self.wandb_run = wandb_run
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "AbsRel": [],
            "MAE": [],
            "RMSE": [],
            "Log10": [],
            "delta1": [],
            "delta2": [],
            "delta3": [],
            "SILog": [],
        }

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train {epoch}", leave=False)):
            image    = batch["color_img"].to(self.device)           # (B, 3, H, W)
            depth_gt = batch["high_res_depth_img"].to(self.device)  # (B, 1, H, W)
            prompt   = batch["low_res_depth_img"].to(self.device)   # (B, 1, h, w)
            boxes = [b.to(self.device) for b in batch["bounding_box"]]

            # First-batch sanity checks.
            if epoch == 1 and batch_idx == 0:
                total_boxes = sum(len(b) for b in boxes)
                Log.info(f"[DEBUG] Batch boxes: {[len(b) for b in boxes]} — total={total_boxes}")
                if total_boxes == 0:
                    Log.warn(
                        "[DEBUG] All boxes are empty. MLF may be bypassed. "
                        "Verify offline box generation."
                    )

            pred = self.model(image, prompt, boxes)

            if pred.shape[-2:] != depth_gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = self.loss_fn(pred, depth_gt)

            if self.optimizer is None:
                raise RuntimeError("Optimizer is None in training mode.")

            # First-batch gradient sanity check for MLF.
            if epoch == 1 and batch_idx == 0:
                self.optimizer.zero_grad()
                loss.backward()
                mlf_params = list(self.model.mlf.parameters())
                grads = [p.grad for p in mlf_params if p.grad is not None]
                if grads:
                    grad_norm = sum(g.norm().item() for g in grads)
                    Log.info(f"[DEBUG] MLF gradient norm = {grad_norm:.6f}")
                else:
                    Log.warn("[DEBUG] MLF has no gradients. Check graph and inputs.")
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                total_loss += loss.item()
                continue

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader, epoch: int) -> tuple[float, dict]:
        """Run one evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        metrics_list = []

        for batch in tqdm(loader, desc=f"Val   {epoch}", leave=False):
            image    = batch["color_img"].to(self.device)
            depth_gt = batch["high_res_depth_img"].to(self.device)
            prompt   = batch["low_res_depth_img"].to(self.device)
            boxes = [b.to(self.device) for b in batch["bounding_box"]]

            pred = self.model(image, prompt, boxes)

            if pred.shape[-2:] != depth_gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = self.loss_fn(pred, depth_gt)

            total_loss += loss.item()
            metrics_list.append(compute_depth_metrics(pred, depth_gt))

        avg_loss = total_loss / len(loader)
        avg_metrics = aggregate_metrics(metrics_list)
        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        """Save checkpoint state."""
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "metrics": metrics,
            "history": self.history,
        }
        path = self.ckpt_dir / f"{tag}.pth"
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str):
        """Load checkpoint state and restore history when available."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        if self.optimizer is not None and state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if "history" in state:
            self.history = state["history"]

        Log.info(f"Loaded checkpoint from {path} (epoch {state.get('epoch', 'unknown')})")
        return state.get("epoch", 0)

    def plot_history(self):
        """Save JSON metrics and a curve figure in the run directory."""
        if len(self.history["val_loss"]) == 0:
            return

        with open(self.ckpt_dir / "metrics_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        epochs = list(range(1, len(self.history["val_loss"]) + 1))
        has_train = len(self.history["train_loss"]) == len(epochs)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        if has_train:
            plt.plot(epochs, self.history["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss", marker="o")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history["AbsRel"], label="AbsRel", color="red", marker="o")
        plt.title("AbsRel (lower is better)")
        plt.xlabel("Epoch")
        plt.ylabel("AbsRel")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history["delta1"], label=r"$\delta < 1.25$", marker="o")
        plt.plot(epochs, self.history["delta2"], label=r"$\delta < 1.25^2$", marker="o")
        plt.plot(epochs, self.history["delta3"], label=r"$\delta < 1.25^3$", marker="o")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Ratio")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.ckpt_dir / "training_curves.png", dpi=150)
        plt.close()

    def log_wandb_metrics(
        self,
        epoch: int,
        val_loss: float,
        metrics: dict,
        train_loss: float | None = None,
        stage: str = "train",
    ):
        """Log metrics to Weights & Biases when a run is available."""
        if self.wandb_run is None:
            return

        log_data = {
            "epoch": epoch,
            "val/loss": val_loss,
            "val/AbsRel": metrics["AbsRel"],
            "val/MAE": metrics["MAE"],
            "val/RMSE": metrics["RMSE"],
            "val/Log10": metrics["Log10"],
            "val/delta1": metrics["delta1"],
            "val/delta2": metrics["delta2"],
            "val/delta3": metrics["delta3"],
            "val/SILog": metrics["SILog"],
            "stage": stage,
        }

        if train_loss is not None:
            log_data["train/loss"] = train_loss

        if self.optimizer is not None:
            for idx, group in enumerate(self.optimizer.param_groups):
                log_data[f"lr/group_{idx}"] = float(group.get("lr", 0.0))

            if len(self.optimizer.param_groups) >= 2:
                log_data["lr/dpt_head"] = float(self.optimizer.param_groups[0].get("lr", 0.0))
                log_data["lr/mlf"] = float(self.optimizer.param_groups[1].get("lr", 0.0))

        self.wandb_run.log(log_data)

    def fit(self, train_loader, val_loader, epochs: int):
        """Run full training and update metrics/checkpoints each epoch."""
        start_epoch = len(self.history["train_loss"]) + 1

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, metrics = self.eval_epoch(val_loader, epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["AbsRel"].append(metrics["AbsRel"])
            self.history["MAE"].append(metrics["MAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["Log10"].append(metrics["Log10"])
            self.history["delta1"].append(metrics["delta1"])
            self.history["delta2"].append(metrics["delta2"])
            self.history["delta3"].append(metrics["delta3"])
            self.history["SILog"].append(metrics["SILog"])

            Log.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"AbsRel={metrics['AbsRel']:.4f} | "
                f"δ<1.25={metrics['delta1']:.4f}"
            )

            if metrics["AbsRel"] < self.best_abs_rel:
                self.best_abs_rel = metrics["AbsRel"]
                self.save_checkpoint(epoch, metrics, tag="best")
                Log.info(f"  ✓ best.pth saved → AbsRel={self.best_abs_rel:.4f}")

            self.save_checkpoint(epoch, metrics, tag="latest")
            self.plot_history()
            self.log_wandb_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
                stage="train",
            )

        Log.info(f"Training done. Best AbsRel = {self.best_abs_rel:.4f}")