"""Direction cosine matrices: construction, conversion, and validity checks.

Ported from ``AttitudeDeterminationLibrary.m`` (ASA_MATLAB).

Convention flag used throughout: ``"col"`` means the matrix acts on column
vectors (``v_B = [C] v_R``), ``"row"`` means it acts on row vectors
(``v_B = v_R [C]``).  The two differ by a transpose.
"""

import numpy as np

__all__ = [
    "angle_between_vectors",
    "check_dcm_constraints",
    "dcm_from_euler_angle_seq",
    "dcm_from_euler_angle_seq_sym",
    "dcm_from_single_euler_angle",
    "dcm_from_space_rotations",
    "dcm_from_space_rotations_sym",
    "dcm_single_euler_angle_2",
    "dcm_to_euler_313",
    "dcm_to_euler_axis_angle",
    "dual_trig_inverse",
    "euler_axis_angle_to_dcm",
    "rotation_dyadic_from_axis_angle",
    "skew_symmetric",
]


def _apply_convention(dcm, convention):
    """Transpose when the requested convention differs from the computed one."""
    if convention.lower() == "row":
        return dcm.T
    if convention.lower() == "col":
        return dcm
    raise ValueError('Invalid convention. Use "row" or "col".')


# --- Vector operations ------------------------------------------------------
def angle_between_vectors(vec1, vec2):
    """Angle between two vectors in radians, theta = acos(a.b / |a||b|)."""
    vec1 = np.asarray(vec1, dtype=float).ravel()
    vec2 = np.asarray(vec2, dtype=float).ravel()
    cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def skew_symmetric(v):
    """Skew-symmetric (cross-product) matrix of a 3-vector."""
    v = np.asarray(v, dtype=float).ravel()
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


# --- Validity ---------------------------------------------------------------
def check_dcm_constraints(dcm, tol=1.0e-6):
    """Check the unit-norm, orthogonality, and right-handedness constraints.

    Returns a dict with the row/column norms, the two orthogonality residuals,
    the determinant, and a single ``valid`` flag.
    """
    dcm = np.asarray(dcm, dtype=float)
    if dcm.shape != (3, 3):
        raise ValueError("DCM must be a 3x3 matrix.")

    row_norms = np.linalg.norm(dcm, axis=1)
    col_norms = np.linalg.norm(dcm, axis=0)
    row_orthogonality_error = np.abs(dcm @ dcm.T - np.eye(3)).max()
    col_orthogonality_error = np.abs(dcm.T @ dcm - np.eye(3)).max()
    determinant = float(np.linalg.det(dcm))

    return {
        "row_norms": row_norms,
        "col_norms": col_norms,
        "row_orthogonality_error": float(row_orthogonality_error),
        "col_orthogonality_error": float(col_orthogonality_error),
        "determinant": determinant,
        "rows_unit": bool(np.all(np.abs(row_norms - 1.0) < tol)),
        "cols_unit": bool(np.all(np.abs(col_norms - 1.0) < tol)),
        "orthogonal": bool(row_orthogonality_error < tol and col_orthogonality_error < tol),
        "right_handed": bool(abs(determinant - 1.0) < tol),
        "valid": bool(row_orthogonality_error < tol and abs(determinant - 1.0) < tol),
    }


# --- Euler axis and angle ---------------------------------------------------
def euler_axis_angle_to_dcm(axis, theta, convention="col"):
    """DCM for a principal rotation of ``theta`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=float).ravel()
    e1, e2, e3 = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)

    dcm = np.array([
        [c + (1 - c) * e1**2, (1 - c) * e1 * e2 + s * e3, (1 - c) * e1 * e3 - s * e2],
        [(1 - c) * e1 * e2 - s * e3, c + (1 - c) * e2**2, (1 - c) * e2 * e3 + s * e1],
        [(1 - c) * e1 * e3 + s * e2, (1 - c) * e2 * e3 - s * e1, c + (1 - c) * e3**2],
    ])
    return _apply_convention(dcm, convention)


def dcm_to_euler_axis_angle(dcm):
    """Euler axis and angle of a DCM, returned as ``(axis, theta)``.

    The axis is ``None`` for the identity, where it is undefined.  The
    ``theta = pi`` case is handled through ``A + I`` since sin(theta) vanishes.
    """
    dcm = np.asarray(dcm, dtype=float)
    cos_theta = np.clip((np.trace(dcm) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if np.isclose(cos_theta, 1.0):
        return None, theta  # identity rotation, axis undefined
    if np.isclose(cos_theta, -1.0):
        # theta = pi, so A = -I + 2 e e^T; normalize the strongest column of A + I.
        a_plus_i = dcm + np.eye(3)
        column = np.unravel_index(np.abs(a_plus_i).argmax(), a_plus_i.shape)[1]
        axis = a_plus_i[:, column]
        return axis / np.linalg.norm(axis), theta

    axis = (1.0 / (2.0 * np.sin(theta))) * np.array([
        dcm[1, 2] - dcm[2, 1],
        dcm[2, 0] - dcm[0, 2],
        dcm[0, 1] - dcm[1, 0],
    ])
    return axis, theta


def rotation_dyadic_from_axis_angle(axis, theta):
    """Rotation dyadic R = I cos(t) - [axis x] sin(t) + (axis axis^T)(1 - cos t)."""
    axis = np.asarray(axis, dtype=float).ravel()
    axis = axis / np.linalg.norm(axis)
    return (np.eye(3) * np.cos(theta)
            - skew_symmetric(axis) * np.sin(theta)
            + np.outer(axis, axis) * (1.0 - np.cos(theta)))


# --- Single-axis rotations --------------------------------------------------
def dcm_from_single_euler_angle(axis, angle, convention="col"):
    """Elementary rotation about axis 1, 2, or 3, column convention by default."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 1:
        dcm = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    elif axis == 2:
        dcm = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    elif axis == 3:
        dcm = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Invalid rotation axis. Axis must be 1, 2, or 3.")
    return _apply_convention(dcm, convention)


def dcm_single_euler_angle_2(axis, angle):
    """Elementary rotation in the row-vector convention (the C matrix)."""
    return dcm_from_single_euler_angle(axis, angle, convention="row")


# --- Body-fixed (intrinsic) Euler sequences ---------------------------------
def _elementary_row(axis, cos_a, sin_a, wrap):
    """Row-convention elementary rotation, built with the caller's array type."""
    if axis == 1:
        entries = [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]
    elif axis == 2:
        entries = [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]
    elif axis == 3:
        entries = [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
    else:
        raise ValueError("Invalid axis. Each element of seq must be 1, 2, or 3.")
    return wrap(entries)


def dcm_from_euler_angle_seq(seq, angles, convention="row"):
    """Composite DCM for a body-fixed sequence, built as R1 @ R2 @ R3."""
    if len(seq) != 3 or len(angles) != 3:
        raise ValueError("Both seq and angles must be vectors of length 3.")

    dcm = np.eye(3)
    for axis, angle in zip(seq, angles):
        dcm = dcm @ _elementary_row(axis, np.cos(angle), np.sin(angle),
                                    lambda e: np.array(e, dtype=float))
    return dcm if convention.lower() == "row" else _apply_convention(dcm, "row")


def dcm_from_euler_angle_seq_sym(seq, angles, convention="row"):
    """Symbolic version of :func:`dcm_from_euler_angle_seq` (needs sympy)."""
    import sympy as sp  # lazy so ASA keeps numpy/scipy as its only hard deps

    if len(seq) != 3 or len(angles) != 3:
        raise ValueError("Both seq and angles must be vectors of length 3.")

    dcm = sp.eye(3)
    for axis, angle in zip(seq, angles):
        dcm = sp.simplify(dcm * _elementary_row(axis, sp.cos(angle), sp.sin(angle), sp.Matrix))
    if convention.lower() == "col":
        return dcm.T
    if convention.lower() == "row":
        return dcm
    raise ValueError('Invalid convention. Use "row" or "col".')


# --- Space-fixed (extrinsic) sequences --------------------------------------
# One table serves both the numeric and the symbolic entry points.  The MATLAB
# source carried two copies of it and the numeric copy had 231/312/132 shifted
# by one entry; this table is the self-consistent symbolic one.
_SPACE_ROTATIONS = {
    "123": lambda c1, s1, c2, s2, c3, s3: [
        [c2 * c3, s1 * s2 * c3 - s3 * c1, c1 * s2 * c3 + s3 * s1],
        [c2 * s3, s1 * s2 * s3 + c3 * c1, c1 * s2 * s3 - c3 * s1],
        [-s2, s1 * c2, c1 * c2]],
    "231": lambda c1, s1, c2, s2, c3, s3: [
        [c1 * c2, -s2, s1 * c2],
        [c1 * s2 * c3 + s3 * s1, c2 * c3, s1 * s2 * c3 - s3 * c1],
        [c1 * s2 * s3 - c3 * s1, c2 * s3, s1 * s2 * s3 + c3 * c1]],
    "312": lambda c1, s1, c2, s2, c3, s3: [
        [s1 * s2 * s3 + c3 * c1, c1 * s2 * s3 - c3 * s1, c2 * s3],
        [s1 * c2, c1 * c2, -s2],
        [s1 * s2 * c3 - s3 * c1, c1 * s2 * c3 + s3 * s1, c2 * c3]],
    "132": lambda c1, s1, c2, s2, c3, s3: [
        [c2 * c3, -c1 * s2 * c3 + s3 * s1, s1 * s2 * c3 + s3 * c1],
        [s2, c1 * c2, -s1 * c2],
        [-c2 * s3, c1 * s2 * s3 + c3 * s1, -s1 * s2 * s3 + c3 * c1]],
    "213": lambda c1, s1, c2, s2, c3, s3: [
        [-s1 * s2 * s3 + c3 * c1, -c2 * s3, c1 * s2 * s3 + c3 * s1],
        [s1 * s2 * c3 + s3 * c1, c2 * c3, -c1 * s2 * c3 + s3 * s1],
        [-s1 * c2, s2, c1 * c2]],
    "321": lambda c1, s1, c2, s2, c3, s3: [
        [c1 * c2, -s1 * c2, s2],
        [c1 * s2 * s3 + c3 * s1, -s1 * s2 * s3 + c3 * c1, -c2 * s3],
        [-c1 * s2 * c3 + s3 * s1, s1 * s2 * c3 + s3 * c1, c2 * c3]],
    "121": lambda c1, s1, c2, s2, c3, s3: [
        [c2, s1 * s2, c1 * s2],
        [s2 * s3, -s1 * c2 * s3 + c3 * c1, -c1 * c2 * s3 - c3 * s1],
        [-s2 * c3, s1 * c2 * c3 + s3 * c1, c1 * c2 * c3 - s3 * s1]],
    "131": lambda c1, s1, c2, s2, c3, s3: [
        [c2, -c1 * s2, s1 * s2],
        [s2 * c3, c1 * c2 * c3 - s3 * s1, -s1 * c2 * c3 - s3 * c1],
        [s2 * s3, c1 * c2 * s3 + c3 * s1, -s1 * c2 * s3 + c3 * c1]],
    "212": lambda c1, s1, c2, s2, c3, s3: [
        [-s1 * c2 * s3 + c3 * c1, s2 * s3, c1 * c2 * s3 + c3 * s1],
        [s1 * s2, c2, -c1 * s2],
        [-s1 * c2 * c3 - s3 * c1, s2 * c3, c1 * c2 * c3 - s3 * s1]],
    "232": lambda c1, s1, c2, s2, c3, s3: [
        [c1 * c2 * c3 - s3 * s1, -s2 * c3, s1 * c2 * c3 + s3 * c1],
        [c1 * s2, c2, s1 * s2],
        [-c1 * c2 * s3 - c3 * s1, s2 * s3, -s1 * c2 * s3 + c3 * c1]],
    "313": lambda c1, s1, c2, s2, c3, s3: [
        [-s1 * c2 * s3 + c3 * c1, -c1 * c2 * s3 - c3 * s1, s2 * s3],
        [s1 * c2 * c3 + s3 * c1, c1 * c2 * c3 - s3 * s1, -s2 * c3],
        [s1 * s2, c1 * s2, c2]],
    "323": lambda c1, s1, c2, s2, c3, s3: [
        [c1 * c2 * c3 - s3 * s1, -s1 * c2 * c3 - s3 * c1, s2 * c3],
        [c1 * c2 * s3 + c3 * s1, -s1 * c2 * s3 + c3 * c1, s2 * s3],
        [-c1 * s2, s1 * s2, c2]],
}


def _space_rotation_entries(seq, cos_fn, sin_fn, angles):
    key = "".join(str(int(axis)) for axis in seq)
    if key not in _SPACE_ROTATIONS:
        raise ValueError(f"Rotation sequence {key} not recognized.")
    c1, c2, c3 = (cos_fn(a) for a in angles)
    s1, s2, s3 = (sin_fn(a) for a in angles)
    return _SPACE_ROTATIONS[key](c1, s1, c2, s2, c3, s3)


def dcm_from_space_rotations(seq, angles, convention="row"):
    """Space-fixed (extrinsic) DCM for a three-axis rotation sequence."""
    if len(seq) != 3 or len(angles) != 3:
        raise ValueError("Both the rotation sequence and angles must be length 3.")
    dcm = np.array(_space_rotation_entries(seq, np.cos, np.sin, angles), dtype=float)
    return dcm if convention.lower() == "row" else _apply_convention(dcm, "row")


def dcm_from_space_rotations_sym(seq, thetas):
    """Symbolic version of :func:`dcm_from_space_rotations` (needs sympy)."""
    import sympy as sp

    return sp.Matrix(_space_rotation_entries(seq, sp.cos, sp.sin, thetas))


# --- Extraction -------------------------------------------------------------
def dcm_to_euler_313(dcm):
    """3-1-3 Euler angles ``(phi, theta, psi)`` from a DCM, in radians."""
    dcm = np.asarray(dcm, dtype=float)
    if abs(dcm[2, 0]) != 1.0:
        theta = float(np.arccos(np.clip(dcm[2, 0], -1.0, 1.0)))
        phi = float(np.arctan2(dcm[2, 1], dcm[2, 2]))
        psi = float(np.arctan2(dcm[1, 0], dcm[0, 0]))
        return phi, theta, psi

    psi = 0.0  # gimbal lock, psi is free
    if dcm[2, 0] == -1.0:
        theta = np.pi / 2.0
        phi = psi + float(np.arctan2(dcm[0, 1], dcm[0, 2]))
    else:
        theta = -np.pi / 2.0
        phi = -psi + float(np.arctan2(-dcm[0, 1], -dcm[0, 2]))
    return phi, theta, psi


# --- Trig helpers -----------------------------------------------------------
def dual_trig_inverse(trig_func, value):
    """All solutions in degrees within [-360, 360] of sin/cos/tan(x) = value."""
    name = trig_func.lower()
    angles = []

    if name in ("sin", "cos"):
        if abs(value) > 1.0:
            raise ValueError(f"For {name}, value must be in [-1, 1].")
        principal = np.degrees(np.arcsin(value) if name == "sin" else np.arccos(value))
        partner = (180.0 - principal) if name == "sin" else -principal
        for k in range(-2, 3):
            angles.extend([principal + 360.0 * k, partner + 360.0 * k])
    elif name == "tan":
        principal = np.degrees(np.arctan(value))
        angles.extend(principal + 180.0 * k for k in range(-3, 4))
    else:
        raise ValueError("Unsupported trig function. Use 'sin', 'cos', or 'tan'.")

    angles = [a for a in angles if -360.0 <= a <= 360.0]
    return np.unique(np.round(angles, 12))
