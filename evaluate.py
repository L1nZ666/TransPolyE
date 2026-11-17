"""Evaluate TransPolyE on a metadata-listed test set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    MultiDomainPatternDataset,
    collate_samples,
    prepare_batch_coordinates,
)
from predict import load_model


@dataclass
class EvalConfig:
    checkpoint_path: Path = Path("checkpoints/best_model_multidomain.pth")
    test_data_dir: Path = Path("data/test/matfile")
    test_list: Path = Path("data/test/poly_data.txt")
    batch_size: int = 50


def main() -> None:
    config = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalizer, nx, nt = load_model(config.checkpoint_path, device)
    dataset = MultiDomainPatternDataset(config.test_data_dir, config.test_list)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )

    correct = 0
    total = 0
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    target_squared_sum = 0.0
    scalar_count = 0

    per_domain = {}

    model.eval()
    with torch.no_grad():
        for (u0, u1, u2), target, domain_infos in loader:
            u0 = u0.to(device=device, dtype=torch.float32)
            u1 = u1.to(device=device, dtype=torch.float32)
            u2 = u2.to(device=device, dtype=torch.float32)

            class_target = target[:, 0].long().to(device)
            param_target = target[:, 1:].to(device)

            u0, u1, u2 = normalizer.normalize(u0, u1, u2)
            t_coords, x_coords = prepare_batch_coordinates(
                nx=nx,
                nt=nt,
                domain_infos=domain_infos,
                device=device,
            )
            params, logits = model(u0, u1, u2, t_coords, x_coords)

            predicted = logits.argmax(dim=1)
            error = params - param_target

            correct += (predicted == class_target).sum().item()
            total += class_target.numel()
            squared_error_sum += torch.sum(error**2).item()
            absolute_error_sum += torch.sum(torch.abs(error)).item()
            target_squared_sum += torch.sum(param_target**2).item()
            scalar_count += param_target.numel()

            for index, info in enumerate(domain_infos):
                domain_id = int(info["domain_id"])
                stats = per_domain.setdefault(
                    domain_id,
                    {"correct": 0, "total": 0, "sq": 0.0, "abs": 0.0, "target_sq": 0.0, "n": 0},
                )
                stats["correct"] += int(predicted[index] == class_target[index])
                stats["total"] += 1
                stats["sq"] += torch.sum(error[index] ** 2).item()
                stats["abs"] += torch.sum(torch.abs(error[index])).item()
                stats["target_sq"] += torch.sum(param_target[index] ** 2).item()
                stats["n"] += param_target[index].numel()

    mse = squared_error_sum / scalar_count
    mae = absolute_error_sum / scalar_count
    rel_l2 = np.sqrt(squared_error_sum / max(target_squared_sum, 1e-16))

    print(f"Accuracy: {100.0 * correct / total:.4f}%")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"Relative L2: {rel_l2:.8f}")

    print("\nPer-domain metrics")
    for domain_id in sorted(per_domain):
        stats = per_domain[domain_id]
        domain_mse = stats["sq"] / stats["n"]
        domain_mae = stats["abs"] / stats["n"]
        domain_rel = np.sqrt(stats["sq"] / max(stats["target_sq"], 1e-16))
        domain_acc = 100.0 * stats["correct"] / stats["total"]
        print(
            f"Domain {domain_id}: accuracy={domain_acc:.4f}%, "
            f"MSE={domain_mse:.8f}, MAE={domain_mae:.8f}, "
            f"relative L2={domain_rel:.8f}"
        )


if __name__ == "__main__":
    main()
