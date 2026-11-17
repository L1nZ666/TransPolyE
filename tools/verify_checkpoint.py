"""Verify that a TransPolyE checkpoint is compatible with the public model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data_utils import prepare_coordinates
from model import TransPolyE
from predict import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, nx, nt = load_model(args.checkpoint, device)

    u0 = torch.zeros(1, 2, nx, device=device)
    u1 = torch.zeros(1, 2, nt, device=device)
    u2 = torch.zeros(1, 2, nt, device=device)
    t_coords, x_coords = prepare_coordinates(
        nx=nx,
        nt=nt,
        t_range=(-5.0, 5.0),
        x_range=(-5.0, 5.0),
        batch_size=1,
        device=device,
    )

    with torch.no_grad():
        params, logits = model(u0, u1, u2, t_coords, x_coords)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print("Checkpoint loaded successfully.")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Nx, Nt: {nx}, {nt}")
    print(f"Parameter output shape: {tuple(params.shape)}")
    print(f"Classification output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
