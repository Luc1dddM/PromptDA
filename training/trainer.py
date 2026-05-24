"""Trainer utilities for PromptDA fine-tuning and evaluation."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from promptda.utils.logger import Log
from training.loss import CombinedLoss
from training.loss_laplace import RobustLaplaceNLLLoss
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
        uncertainty: bool = False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.uncertainty = uncertainty
        self.loss_fn = RobustLaplaceNLLLoss() if uncertainty else CombinedLoss()
        self.best_l1 = float("inf")
        self.global_step = 0
        self.start_epoch = 1
        self.wandb_run = wandb_run
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "L1": [],
            "MAE": [],
            "RMSE": [],
        }

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train {epoch}", leave=False)):
            image = batch["color_img"].to(self.device)
            depth_gt = batch["high_res_depth_img"].to(self.device)
            prompt = batch["low_res_depth_img"].to(self.device)
            pred = self.model(image, prompt)

            # GT is loaded at PromptDA output/RGB size; mismatches indicate data pipeline drift.
            pred_depth = pred["mu"] if isinstance(pred, dict) else pred
            if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                raise RuntimeError(
                    f"Prediction shape {tuple(pred_depth.shape[-2:])} does not match "
                    f"GT shape {tuple(depth_gt.shape[-2:])}."
                )

            if epoch == 1 and batch_idx == 0:
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                Log.info(f"[DEBUG] Trainable params: {trainable}")

            loss, loss_info = self.loss_fn(pred, depth_gt)

            if self.optimizer is None:
                raise RuntimeError("Optimizer is None in training mode.")

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader, epoch: int) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        metrics_list = []

        for batch in tqdm(loader, desc=f"Val   {epoch}", leave=False):
            image = batch["color_img"].to(self.device)
            depth_gt = batch["high_res_depth_img"].to(self.device)
            prompt = batch["low_res_depth_img"].to(self.device)
            pred = self.model(image, prompt)

            # GT is loaded at PromptDA output/RGB size; mismatches indicate data pipeline drift.
            pred_depth = pred["mu"] if isinstance(pred, dict) else pred
            if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                raise RuntimeError(
                    f"Prediction shape {tuple(pred_depth.shape[-2:])} does not match "
                    f"GT shape {tuple(depth_gt.shape[-2:])}."
                )

            loss, _ = self.loss_fn(pred, depth_gt)

            total_loss += loss.item()
            # compute_depth_metrics expects a tensor — pass mu for uncertainty, pred for legacy
            metrics_input = pred["mu"] if isinstance(pred, dict) else pred
            metrics_list.append(compute_depth_metrics(metrics_input, depth_gt))

        avg_loss = total_loss / len(loader)
        avg_metrics = aggregate_metrics(metrics_list)
        return avg_loss, avg_metrics

    def save_checkpoint(self, epoch: int, metrics: dict, tag: str = "latest"):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": metrics,
            "history": self.history,
            "best_l1": self.best_l1,
        }
        path = self.ckpt_dir / f"{tag}.pth"
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)

        if self.optimizer is not None and state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])

        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])

        if "history" in state:
            self.history = state["history"]

        self.global_step = int(state.get("global_step", 0))
        last_epoch = int(state.get("epoch", 0))
        self.start_epoch = last_epoch + 1

        if "best_l1" in state:
            self.best_l1 = state["best_l1"]
        elif "best_abs_rel" in state:
            self.best_l1 = state["best_abs_rel"]
        elif self.history.get("L1"):
            self.best_l1 = min(self.history["L1"])
        elif self.history.get("AbsRel"):
            self.best_l1 = min(self.history["AbsRel"])
        elif state.get("metrics") and "L1" in state["metrics"]:
            self.best_l1 = state["metrics"]["L1"]
        elif state.get("metrics") and "AbsRel" in state["metrics"]:
            self.best_l1 = state["metrics"]["AbsRel"]

        Log.info(
            f"Loaded checkpoint from {path} | epoch={last_epoch} | "
            f"global_step={self.global_step} | best L1={self.best_l1:.4f}"
        )
        return last_epoch

    def plot_history(self):
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
        plt.plot(epochs, self.history["L1"], label="L1", color="red", marker="o")
        plt.title("L1 (lower is better)")
        plt.xlabel("Epoch")
        plt.ylabel("L1")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history["RMSE"], label="RMSE", marker="o")
        plt.title("RMSE (lower is better)")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
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
        if self.wandb_run is None:
            return

        log_data = {
            "epoch": epoch,
            "val/loss": val_loss,
            "val/L1": metrics["L1"],
            "val/RMSE": metrics["RMSE"],
            "stage": stage,
        }

        if train_loss is not None:
            log_data["train/loss"] = train_loss

        if self.optimizer is not None:
            for idx, group in enumerate(self.optimizer.param_groups):
                log_data[f"lr/group_{idx}"] = float(group.get("lr", 0.0))

        self.wandb_run.log(log_data)

    def fit(self, train_loader, val_loader, epochs: int):
        start_epoch = self.start_epoch

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, metrics = self.eval_epoch(val_loader, epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["L1"].append(metrics["L1"])
            self.history["MAE"].append(metrics["L1"])
            self.history["RMSE"].append(metrics["RMSE"])

            Log.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"L1={metrics['L1']:.4f} | RMSE={metrics['RMSE']:.4f}"
            )

            if metrics["L1"] < self.best_l1:
                self.best_l1 = metrics["L1"]
                self.save_checkpoint(epoch, metrics, tag="best")
                Log.info(f"  ✓ best.pth saved → L1={self.best_l1:.4f}")

            self.save_checkpoint(epoch, metrics, tag="latest")
            self.plot_history()
            self.log_wandb_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
                stage="train",
            )

        Log.info(f"Training done. Best L1 = {self.best_l1:.4f}")
