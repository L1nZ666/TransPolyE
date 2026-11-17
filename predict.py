"""Run TransPolyE inference for one MATLAB sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data_utils import (
    ComplexNormalizer,
    complex_to_channels,
    load_mat_traces,
    prepare_coordinates,
)
from model import TransPolyE
from polynomials import CLASS_NAMES, polynomial_roots


@dataclass
class PredictConfig:
    # Edit these paths and ranges before running.
    checkpoint_path: Path = Path("checkpoints/best_model_multidomain.pth")
    sample_path: Path = Path("data/sample/matfile/7233.mat")
    t_range: tuple[float, float] = (-5.0, 5.0)
    x_range: tuple[float, float] = (-5.0, 5.0)
    root_scale: float = 5.0


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | str,
):
    """Load a checkpoint produced by the public training script or the original notebook."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    nx = int(checkpoint["Nx"])
    nt = int(checkpoint["Nt"])
    stats = checkpoint["stats"]

    saved = checkpoint.get("model_config", {})
    model = TransPolyE(
        d_model=int(saved.get("d_model", 256)),
        nhead=int(saved.get("nhead", 8)),
        num_encoder_layers=int(saved.get("num_encoder_layers", 6)),
        nx=nx,
        nt=nt,
        num_classes=int(saved.get("num_classes", 6)),
        dropout=float(saved.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = ComplexNormalizer(*stats, device=device)
    return model, normalizer, nx, nt


def predict_traces(
    model: TransPolyE,
    normalizer: ComplexNormalizer,
    u0: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    t_range: tuple[float, float],
    x_range: tuple[float, float],
    nx: int,
    nt: int,
    device: torch.device | str,
):
    """Predict one polynomial class and its two continuous parameters."""
    tensors = [
        complex_to_channels(trace).unsqueeze(0).to(device)
        for trace in (u0, u1, u2)
    ]

    if tensors[0].shape[-1] != nx:
        raise ValueError(f"Expected u0 length {nx}, got {tensors[0].shape[-1]}.")
    if tensors[1].shape[-1] != nt or tensors[2].shape[-1] != nt:
        raise ValueError(
            f"Expected boundary length {nt}, got "
            f"{tensors[1].shape[-1]} and {tensors[2].shape[-1]}."
        )

    u0_t, u1_t, u2_t = normalizer.normalize(*tensors)
    t_coords, x_coords = prepare_coordinates(
        nx=nx,
        nt=nt,
        t_range=t_range,
        x_range=x_range,
        batch_size=1,
        device=device,
    )

    with torch.no_grad():
        params, logits = model(u0_t, u1_t, u2_t, t_coords, x_coords)

    class_id = int(logits.argmax(dim=1).item())
    gamma1, gamma2 = (float(value) for value in params[0].cpu().tolist())
    return class_id, gamma1, gamma2


def predict_mat_file(
    model: TransPolyE,
    normalizer: ComplexNormalizer,
    sample_path: str | Path,
    t_range: tuple[float, float],
    x_range: tuple[float, float],
    nx: int,
    nt: int,
    device: torch.device | str,
):
    """Load one MATLAB sample and run inference."""
    u0, u1, u2 = load_mat_traces(sample_path)
    return predict_traces(
        model=model,
        normalizer=normalizer,
        u0=u0,
        u1=u1,
        u2=u2,
        t_range=t_range,
        x_range=x_range,
        nx=nx,
        nt=nt,
        device=device,
    )


def main() -> None:
    config = PredictConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalizer, nx, nt = load_model(config.checkpoint_path, device)
    class_id, gamma1, gamma2 = predict_mat_file(
        model=model,
        normalizer=normalizer,
        sample_path=config.sample_path,
        t_range=config.t_range,
        x_range=config.x_range,
        nx=nx,
        nt=nt,
        device=device,
    )

    roots = polynomial_roots(
        class_id=class_id,
        gamma1=gamma1,
        gamma2=gamma2,
        scale=config.root_scale,
    )

    print(f"Predicted class: {class_id} ({CLASS_NAMES[class_id]})")
    print(f"Predicted gamma1: {gamma1:.6f}")
    print(f"Predicted gamma2: {gamma2:.6f}")
    print("Predicted root coordinates [x, t]:")
    print(roots)


if __name__ == "__main__":
    main()
