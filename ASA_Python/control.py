"""Discrete-time linearization helpers used by the virtual thrust workflow."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

FloatArray = NDArray[np.float64]


def linear_discrete_time_matrices(
    tk: float,
    tk_1: float,
    yk: Iterable[float] | FloatArray,
    A: Iterable[float] | FloatArray,
    B: Iterable[float] | FloatArray,
    c: Iterable[float] | FloatArray,
    uk: Iterable[float] | FloatArray | None,
    szx: int,
    szu: int,
    ode_options: object | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return exact ZOH ``(Ak, Bk, ck, xk_1)`` for constant ``A, B, c``."""

    del ode_options

    xk = np.asarray(yk, dtype=float).reshape(-1)[:szx]
    A_mat = np.asarray(A, dtype=float).reshape((szx, szx))
    B_mat = np.asarray(B, dtype=float).reshape((szx, szu))
    c_vec = np.asarray(c, dtype=float).reshape(szx)
    u_vec = np.zeros(szu, dtype=float) if uk is None else np.asarray(uk, dtype=float).reshape(szu)

    dt = float(tk_1 - tk)
    if np.isclose(dt, 0.0):
        Ak = np.eye(szx, dtype=float)
        Bk = np.zeros((szx, szu), dtype=float)
        ck = np.zeros(szx, dtype=float)
        return Ak, Bk, ck, xk.copy()

    aug = np.zeros((szx + szu + 1, szx + szu + 1), dtype=float)
    aug[:szx, :szx] = A_mat
    aug[:szx, szx : szx + szu] = B_mat
    aug[:szx, -1] = c_vec
    transition = expm(aug * dt)

    Ak = transition[:szx, :szx]
    Bk = transition[:szx, szx : szx + szu]
    ck = transition[:szx, -1]
    xk_1 = Ak @ xk + Bk @ u_vec + ck
    return Ak, Bk, ck, xk_1
