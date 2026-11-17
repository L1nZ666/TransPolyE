"""Dataset, normalization, and coordinate utilities for TransPolyE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleMeta:
    filename: str
    class_id: int
    gamma1: float
    gamma2: float
    domain_id: int
    t_range: tuple[float, float]
    x_range: tuple[float, float]


def _parse_sample_line(line: str) -> SampleMeta | None:
    parts = line.strip().split()
    if len(parts) < 9 or not parts[0].lower().endswith(".mat"):
        return None
    return SampleMeta(
        filename=parts[0],
        class_id=int(float(parts[1])),
        gamma1=float(parts[2]),
        gamma2=float(parts[3]),
        domain_id=int(float(parts[4])),
        t_range=(float(parts[5]), float(parts[6])),
        x_range=(float(parts[7]), float(parts[8])),
    )


def read_sample_metadata(list_file: str | Path) -> list[SampleMeta]:
    """Read all sample entries from a TransPolyE metadata file."""
    lines = Path(list_file).read_text(encoding="utf-8").splitlines()
    samples = [sample for line in lines if (sample := _parse_sample_line(line))]
    if not samples:
        raise ValueError(f"No sample entries were found in {list_file}.")
    return samples


def load_normalization_stats(
    list_file: str | Path,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Read the 12 normalization values stored in the statistics line.

    Expected order:
    u0(Re mean, Re std, Im mean, Im std),
    u1(Re mean, Re std, Im mean, Im std),
    u2(Re mean, Re std, Im mean, Im std).
    """
    lines = [
        line.strip()
        for line in Path(list_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    stats = None
    for line in reversed(lines):
        parts = line.split()
        if len(parts) != 12:
            continue
        try:
            stats = [float(value) for value in parts]
            break
        except ValueError:
            continue

    if stats is None:
        raise ValueError(
            f"Could not find a 12-value normalization-statistics line in {list_file}."
        )

    def pair(offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor([[stats[offset]], [stats[offset + 2]]], dtype=torch.float32)
        std = torch.tensor([[stats[offset + 1]], [stats[offset + 3]]], dtype=torch.float32)
        return mean, std

    return pair(0), pair(4), pair(8)


def load_mat_traces(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load u0, u1, and u2 complex traces from one MATLAB sample."""
    mat = sio.loadmat(path)
    if "u_data" not in mat:
        raise KeyError(f"'u_data' was not found in {path}.")

    u_data = mat["u_data"][0, 0]
    return (
        np.asarray(u_data["u0"]).squeeze(),
        np.asarray(u_data["u1"]).squeeze(),
        np.asarray(u_data["u2"]).squeeze(),
    )


def complex_to_channels(values: np.ndarray) -> torch.Tensor:
    """Convert a complex vector to a [2, L] float tensor."""
    values = np.asarray(values).squeeze()
    return torch.stack(
        [
            torch.from_numpy(values.real).float(),
            torch.from_numpy(values.imag).float(),
        ],
        dim=0,
    )


class MultiDomainPatternDataset(Dataset):
    """MAT-file dataset with class, parameter, and physical-domain metadata."""

    def __init__(self, root: str | Path, list_file: str | Path) -> None:
        self.root = Path(root)
        self.samples = read_sample_metadata(list_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        meta = self.samples[index]
        u0, u1, u2 = load_mat_traces(self.root / meta.filename)

        traces = (
            complex_to_channels(u0),
            complex_to_channels(u1),
            complex_to_channels(u2),
        )
        target = torch.tensor(
            [meta.class_id, meta.gamma1, meta.gamma2],
            dtype=torch.float32,
        )
        domain_info = {
            "domain_id": meta.domain_id,
            "t_range": meta.t_range,
            "x_range": meta.x_range,
        }
        return traces, target, domain_info


def collate_samples(batch):
    """Collate tensors while retaining per-sample physical-domain metadata."""
    traces = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    domains = [item[2] for item in batch]

    u0 = torch.stack([item[0] for item in traces], dim=0)
    u1 = torch.stack([item[1] for item in traces], dim=0)
    u2 = torch.stack([item[2] for item in traces], dim=0)
    target = torch.stack(targets, dim=0)
    return (u0, u1, u2), target, domains


class ComplexNormalizer:
    """Apply fixed channel-wise normalization to the three complex traces."""

    def __init__(
        self,
        stats_u0: tuple[torch.Tensor, torch.Tensor],
        stats_u1: tuple[torch.Tensor, torch.Tensor],
        stats_u2: tuple[torch.Tensor, torch.Tensor],
        device: torch.device | str,
    ) -> None:
        self.mean_u0, self.std_u0 = self._prepare(stats_u0, device)
        self.mean_u1, self.std_u1 = self._prepare(stats_u1, device)
        self.mean_u2, self.std_u2 = self._prepare(stats_u2, device)

    @staticmethod
    def _prepare(
        stats: tuple[torch.Tensor, torch.Tensor],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = stats
        return (
            mean.to(device).view(1, 2, 1),
            std.to(device).view(1, 2, 1),
        )

    def normalize(
        self,
        u0: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-8
        return (
            (u0 - self.mean_u0) / (self.std_u0 + eps),
            (u1 - self.mean_u1) / (self.std_u1 + eps),
            (u2 - self.mean_u2) / (self.std_u2 + eps),
        )


def prepare_coordinates(
    nx: int,
    nt: int,
    t_range: tuple[float, float],
    x_range: tuple[float, float],
    batch_size: int,
    device: torch.device | str,
):
    """Create physical coordinates for one shared observation domain."""
    t_min, t_max = t_range
    x_min, x_max = x_range

    t_u0 = torch.full((batch_size, nx, 1), t_min, device=device, dtype=torch.float32)
    x_u0 = (
        torch.linspace(x_min, x_max, nx, device=device)
        .view(1, nx, 1)
        .expand(batch_size, -1, -1)
    )

    t_line = (
        torch.linspace(t_min, t_max, nt, device=device)
        .view(1, nt, 1)
        .expand(batch_size, -1, -1)
    )
    x_u1 = torch.full((batch_size, nt, 1), x_min, device=device, dtype=torch.float32)
    x_u2 = torch.full((batch_size, nt, 1), x_max, device=device, dtype=torch.float32)

    t_coords = {"u0": t_u0, "u1": t_line, "u2": t_line.clone()}
    x_coords = {"u0": x_u0, "u1": x_u1, "u2": x_u2}
    return t_coords, x_coords


def prepare_batch_coordinates(
    nx: int,
    nt: int,
    domain_infos: Iterable[dict],
    device: torch.device | str,
):
    """Create coordinate tensors when samples in one mini-batch use different domains."""
    t_acc = {"u0": [], "u1": [], "u2": []}
    x_acc = {"u0": [], "u1": [], "u2": []}

    for info in domain_infos:
        t_coords, x_coords = prepare_coordinates(
            nx=nx,
            nt=nt,
            t_range=info["t_range"],
            x_range=info["x_range"],
            batch_size=1,
            device=device,
        )
        for key in t_acc:
            t_acc[key].append(t_coords[key])
            x_acc[key].append(x_coords[key])

    return (
        {key: torch.cat(value, dim=0) for key, value in t_acc.items()},
        {key: torch.cat(value, dim=0) for key, value in x_acc.items()},
    )
