"""Create a compact representative sample dataset from the full TransPolyE data.

The script copies existing MATLAB samples; it does not regenerate data or retrain
TransPolyE. By default, it selects one sample for each class in each requested
domain and writes a matching metadata file.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import read_sample_metadata


def find_stats_line(list_file: Path) -> str | None:
    """Return the trailing 12-value normalization-statistics line, if present."""
    lines = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in reversed(lines):
        parts = line.split()
        if len(parts) != 12:
            continue
        try:
            [float(value) for value in parts]
            return line
        except ValueError:
            continue
    return None


def format_meta(sample) -> str:
    """Serialize one SampleMeta entry in the original metadata format."""
    return (
        f"{sample.filename} {sample.class_id} {sample.gamma1:.16g} "
        f"{sample.gamma2:.16g} {sample.domain_id} "
        f"{sample.t_range[0]:.16g} {sample.t_range[1]:.16g} "
        f"{sample.x_range[0]:.16g} {sample.x_range[1]:.16g}"
    )


def choose_representatives(samples, domain_ids: list[int], samples_per_class: int):
    """Choose samples close to the median parameter values for each class/domain."""
    selected = []
    classes = sorted({sample.class_id for sample in samples})

    for domain_id in domain_ids:
        for class_id in classes:
            group = [
                sample
                for sample in samples
                if sample.domain_id == domain_id and sample.class_id == class_id
            ]
            if not group:
                print(f"Warning: no sample found for domain={domain_id}, class={class_id}.")
                continue

            # Deterministic, representative selection rather than random sampling.
            group.sort(key=lambda s: (s.gamma1, s.gamma2, s.filename))
            if samples_per_class == 1:
                indices = [len(group) // 2]
            else:
                indices = [
                    round(i * (len(group) - 1) / (samples_per_class - 1))
                    for i in range(samples_per_class)
                ]
            selected.extend(group[index] for index in indices)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True, help="Directory containing the original .mat files.")
    parser.add_argument("--source-list", type=Path, required=True, help="Original metadata file, e.g. poly_data.txt.")
    parser.add_argument("--output", type=Path, default=Path("data/sample"), help="Output directory.")
    parser.add_argument(
        "--domains",
        type=int,
        nargs="+",
        default=None,
        help="Domain IDs to include. If omitted, all domain IDs in the metadata are used.",
    )
    parser.add_argument("--samples-per-class", type=int, default=1, help="Samples selected per class and domain.")
    args = parser.parse_args()

    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1.")

    samples = read_sample_metadata(args.source_list)
    domain_ids = args.domains if args.domains is not None else sorted({sample.domain_id for sample in samples})
    selected = choose_representatives(samples, domain_ids, args.samples_per_class)
    if not selected:
        raise RuntimeError("No samples were selected. Check the requested domain IDs.")

    mat_dir = args.output / "matfile"
    mat_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines = []
    total_bytes = 0
    for sample in selected:
        source = args.source_data / sample.filename
        if not source.exists():
            raise FileNotFoundError(source)
        destination = mat_dir / sample.filename
        shutil.copy2(source, destination)
        total_bytes += destination.stat().st_size
        metadata_lines.append(format_meta(sample))

    stats_line = find_stats_line(args.source_list)
    if stats_line is not None:
        metadata_lines.append(stats_line)

    list_path = args.output / "poly_data.txt"
    list_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    print(f"Created sample dataset with {len(selected)} MATLAB files.")
    print(f"Metadata: {list_path}")
    print(f"Data size: {total_bytes / (1024 ** 2):.2f} MiB")
    print("Selected samples:")
    for sample in selected:
        print(
            f"  domain={sample.domain_id}, class={sample.class_id}, "
            f"gamma=({sample.gamma1:.4g}, {sample.gamma2:.4g}), {sample.filename}"
        )


if __name__ == "__main__":
    main()
