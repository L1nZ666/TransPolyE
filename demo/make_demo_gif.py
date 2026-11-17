"""Create an animated TransPolyE demo from a trained checkpoint.

The animation shows how changing initial observations map to predicted
characteristic-polynomial root distributions.

Edit the configuration block below, then run this file directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from data_utils import load_mat_traces, read_sample_metadata
from polynomials import CLASS_NAMES, polynomial_roots
from predict import load_model, predict_traces

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DemoConfig:
    checkpoint_path: Path = PROJECT_ROOT / "checkpoints" / "best_model_multidomain.pth"

    data_dir: Path = PROJECT_ROOT / "data" / "sample" / "matfile"

    list_file: Path = PROJECT_ROOT / "data" / "sample" / "poly_data.txt"

    output_path: Path = PROJECT_ROOT / "assets" / "transpolye_demo.gif"

    class_id: int = 0
    domain_id: int | None = None
    num_frames: int = 24
    root_scale: float = 5.0
    fps: int = 5


def choose_samples(config: DemoConfig):
    """Select evenly spaced samples after sorting by gamma1."""
    samples = [
        sample
        for sample in read_sample_metadata(config.list_file)
        if sample.class_id == config.class_id
        and (config.domain_id is None or sample.domain_id == config.domain_id)
    ]
    if not samples:
        raise ValueError("No samples match the selected class/domain.")

    samples.sort(key=lambda sample: (sample.gamma1, sample.gamma2))
    count = min(config.num_frames, len(samples))
    indices = np.linspace(0, len(samples) - 1, count, dtype=int)
    return [samples[index] for index in indices]


def build_demo(config: DemoConfig = DemoConfig()) -> Path:
    """Build and save the animated GIF."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, nx, nt = load_model(config.checkpoint_path, device)

    selected = choose_samples(config)
    frames = []

    for meta in selected:
        u0, u1, u2 = load_mat_traces(config.data_dir / meta.filename)
        class_id, gamma1, gamma2 = predict_traces(
            model=model,
            normalizer=normalizer,
            u0=u0,
            u1=u1,
            u2=u2,
            t_range=meta.t_range,
            x_range=meta.x_range,
            nx=nx,
            nt=nt,
            device=device,
        )
        roots = polynomial_roots(class_id, gamma1, gamma2, config.root_scale)
        x_grid = np.linspace(meta.x_range[0], meta.x_range[1], len(u0))
        frames.append(
            {
                "meta": meta,
                "x_grid": x_grid,
                "u0": u0,
                "class_id": class_id,
                "gamma1": gamma1,
                "gamma2": gamma2,
                "roots": roots,
            }
        )

    all_roots = np.vstack([frame["roots"] for frame in frames if len(frame["roots"])])
    if len(all_roots):
        root_limit = max(5.0, 1.15 * float(np.max(np.abs(all_roots))))
    else:
        root_limit = 5.0

    max_u0 = max(float(np.max(np.abs(frame["u0"]))) for frame in frames)

    fig, (ax_input, ax_roots) = plt.subplots(1, 2, figsize=(10, 4.5))
    line, = ax_input.plot([], [], linewidth=2)
    scatter = ax_roots.scatter([], [], marker="*", s=120)

    ax_input.set_xlabel("x")
    ax_input.set_ylabel(r"$|u_0(x)|$")
    ax_input.set_ylim(0, 1.1 * max_u0)
    ax_input.grid(alpha=0.25)

    ax_roots.set_xlabel("Re(z)")
    ax_roots.set_ylabel("Im(z)")
    ax_roots.set_xlim(-root_limit, root_limit)
    ax_roots.set_ylim(-root_limit, root_limit)
    ax_roots.set_aspect("equal", adjustable="box")
    ax_roots.axhline(0, linewidth=0.8)
    ax_roots.axvline(0, linewidth=0.8)
    ax_roots.grid(alpha=0.25)

    title = fig.suptitle("")

    def update(frame_index: int):
        frame = frames[frame_index]
        x_grid = frame["x_grid"]
        amplitude = np.abs(frame["u0"])
        roots = frame["roots"]

        line.set_data(x_grid, amplitude)
        ax_input.set_xlim(x_grid[0], x_grid[-1])
        ax_input.set_title(
            rf"Initial trace: true $\gamma_1={frame['meta'].gamma1:.3f}$, "
            rf"$\gamma_2={frame['meta'].gamma2:.3f}$"
        )

        scatter.set_offsets(roots if len(roots) else np.empty((0, 2)))
        ax_roots.set_title(
            rf"Predicted roots: $\hat{{\gamma}}_1={frame['gamma1']:.3f}$, "
            rf"$\hat{{\gamma}}_2={frame['gamma2']:.3f}$"
        )

        title.set_text(
            f"TransPolyE | predicted {CLASS_NAMES[frame['class_id']]} "
            f"(frame {frame_index + 1}/{len(frames)})"
        )
        return line, scatter, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / config.fps,
        blit=False,
        repeat=True,
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(config.output_path, writer=PillowWriter(fps=config.fps))
    plt.close(fig)
    return config.output_path


if __name__ == "__main__":
    path = build_demo()
    print(f"Saved animation to: {path}")
