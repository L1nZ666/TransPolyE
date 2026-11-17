"""Characteristic-polynomial utilities used by TransPolyE."""

from __future__ import annotations

import numpy as np


CLASS_NAMES = (
    "2RW triangle",
    "3RW triangle",
    "3RW pentagon",
    "4RW triangle",
    "4RW pentagon",
    "4RW heptagon",
)


def polynomial_coefficients(
    class_id: int,
    gamma1: float,
    gamma2: float,
) -> np.ndarray:
    """Return polynomial coefficients in descending powers."""
    g1 = float(gamma1)
    g2 = float(gamma2)

    if class_id == 0:
        coeffs = [-1, 0, 0, -g2 + 1j * g1]
    elif class_id == 1:
        coeffs = [
            1, 0, 0, -5j * g1 + 5 * g2, 0, 0,
            10j * g2 * g1 - 5 * g2**2 + 5 * g1**2,
        ]
    elif class_id == 2:
        coeffs = [1, 0, 0, 0, 0, -1j * g2 - g1, 0]
    elif class_id == 3:
        coeffs = [
            1, 0, 0, -15j * g1 + 15 * g2, 0, 0, 0, 0, 0,
            -525j * g2**2 * g1
            + 175j * g1**3
            + 175 * g2**3
            - 525 * g1**2 * g2,
            0,
        ]
    elif class_id == 4:
        coeffs = [
            1, 0, 0, 0, 0, -7j * g2 - 7 * g1, 0, 0, 0, 0,
            -14j * g1 * g2 / 3 + 7 * g2**2 / 3 - 7 * g1**2 / 3,
        ]
    elif class_id == 5:
        coeffs = [1, 0, 0, 0, 0, 0, 0, 1j * g1 - g2, 0, 0, 0]
    else:
        raise ValueError(f"class_id must be in [0, 5], got {class_id}.")

    return np.asarray(coeffs, dtype=np.complex128)


def polynomial_roots(
    class_id: int,
    gamma1: float,
    gamma2: float,
    scale: float = 5.0,
) -> np.ndarray:
    """Return scaled complex-root coordinates as an [N, 2] array."""
    roots = np.roots(polynomial_coefficients(class_id, gamma1, gamma2))
    return np.column_stack((scale * roots.real, scale * roots.imag))
