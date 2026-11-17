"""Train TransPolyE on multi-domain initial-boundary data."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    ComplexNormalizer,
    MultiDomainPatternDataset,
    collate_samples,
    load_normalization_stats,
    prepare_batch_coordinates,
)
from model import TransPolyE


@dataclass
class TrainConfig:
    # Edit these paths before running.
    train_data_dir: Path = Path("data/train/matfile")
    val_data_dir: Path = Path("data/val/matfile")
    train_list: Path = Path("data/train/poly_data.txt")
    val_list: Path = Path("data/val/poly_data.txt")

    checkpoint_path: Path = Path("checkpoints/best_model_multidomain.pth")
    log_path: Path = Path("training_multidomain.csv")

    learning_rate: float = 6e-4
    batch_size: int = 50
    num_epochs: int = 100
    weight_decay: float = 1e-5
    cls_weight: float = 1.0
    reg_weight: float = 10.0
    grad_clip: float = 1.0

    nx: int = 200
    nt: int = 100
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_classes: int = 6
    dropout: float = 0.1
    seed: int = 3407
    num_workers: int = 0


def set_seed(seed: int) -> None:
    """Set random seeds used by the training script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    logits: torch.Tensor,
    params_pred: torch.Tensor,
    class_target: torch.Tensor,
    param_target: torch.Tensor,
    cls_weight: float,
    reg_weight: float,
):
    """Return the weighted joint loss and detached components."""
    cls_loss = F.cross_entropy(logits, class_target.long())
    reg_loss = F.mse_loss(params_pred, param_target)
    total = cls_weight * cls_loss + reg_weight * reg_loss
    return total, cls_loss.detach(), reg_loss.detach()


def run_epoch(
    model: TransPolyE,
    loader: DataLoader,
    normalizer: ComplexNormalizer,
    device: torch.device,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
):
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    correct = 0
    sample_count = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for (u0, u1, u2), target, domain_infos in loader:
            u0 = u0.to(device=device, dtype=torch.float32)
            u1 = u1.to(device=device, dtype=torch.float32)
            u2 = u2.to(device=device, dtype=torch.float32)
            class_target = target[:, 0].to(device)
            param_target = target[:, 1:].to(device)

            u0, u1, u2 = normalizer.normalize(u0, u1, u2)
            t_coords, x_coords = prepare_batch_coordinates(
                nx=config.nx,
                nt=config.nt,
                domain_infos=domain_infos,
                device=device,
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            params_pred, logits = model(u0, u1, u2, t_coords, x_coords)
            loss, cls_loss, reg_loss = compute_loss(
                logits=logits,
                params_pred=params_pred,
                class_target=class_target,
                param_target=param_target,
                cls_weight=config.cls_weight,
                reg_weight=config.reg_weight,
            )

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            batch_size = class_target.size(0)
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_reg += reg_loss.item()
            correct += (logits.argmax(dim=1) == class_target.long()).sum().item()
            sample_count += batch_size

    num_batches = max(len(loader), 1)
    return {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls / num_batches,
        "reg_loss": total_reg / num_batches,
        "accuracy": 100.0 * correct / max(sample_count, 1),
    }


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MultiDomainPatternDataset(config.train_data_dir, config.train_list)
    val_dataset = MultiDomainPatternDataset(config.val_data_dir, config.val_list)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )

    stats = load_normalization_stats(config.train_list)
    normalizer = ComplexNormalizer(*stats, device=device)

    model = TransPolyE(
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        nx=config.nx,
        nt=config.nt,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    with config.log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ]
        )

        for epoch in range(1, config.num_epochs + 1):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                normalizer=normalizer,
                device=device,
                config=config,
                optimizer=optimizer,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                normalizer=normalizer,
                device=device,
                config=config,
                optimizer=None,
            )

            scheduler.step(val_metrics["loss"])
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                [
                    epoch,
                    lr,
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                ]
            )
            handle.flush()

            print(
                f"Epoch {epoch:03d}/{config.num_epochs} | "
                f"train loss {train_metrics['loss']:.6f}, "
                f"train acc {train_metrics['accuracy']:.2f}% | "
                f"val loss {val_metrics['loss']:.6f}, "
                f"val acc {val_metrics['accuracy']:.2f}%"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": val_metrics["accuracy"],
                    "Nx": config.nx,
                    "Nt": config.nt,
                    "stats": stats,
                    "model_config": {
                        "d_model": config.d_model,
                        "nhead": config.nhead,
                        "num_encoder_layers": config.num_encoder_layers,
                        "num_classes": config.num_classes,
                        "dropout": config.dropout,
                    },
                }
                torch.save(checkpoint, config.checkpoint_path)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
