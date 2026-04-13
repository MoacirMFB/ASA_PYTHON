import numpy as np

from ..control import linear_discrete_time_matrices


def test_linear_discrete_time_matrices_double_integrator():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    c = np.zeros(2)
    x0 = np.array([3.0, -2.0])
    y0 = np.concatenate((x0, np.eye(2).reshape(-1, order="F"), np.zeros(2), np.zeros(2)))
    Ak, Bk, ck, x1 = linear_discrete_time_matrices(0.0, 0.1, y0, A, B, c, np.array([4.0]), 2, 1)

    np.testing.assert_allclose(Ak, np.array([[1.0, 0.1], [0.0, 1.0]]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Bk, np.array([[0.005], [0.1]]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ck, np.zeros(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x1, np.array([2.82, -1.6]), rtol=1e-12, atol=1e-12)


def test_linear_discrete_time_matrices_handles_affine_term():
    A = np.zeros((2, 2))
    B = np.eye(2)
    c = np.array([1.0, -2.0])
    x0 = np.array([4.0, 5.0])
    y0 = np.concatenate((x0, np.eye(2).reshape(-1, order="F"), np.zeros(4), np.zeros(2)))
    Ak, Bk, ck, x1 = linear_discrete_time_matrices(3.0, 5.0, y0, A, B, c, np.array([0.5, 1.5]), 2, 2)

    np.testing.assert_allclose(Ak, np.eye(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Bk, 2.0 * np.eye(2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ck, np.array([2.0, -4.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x1, np.array([7.0, 4.0]), rtol=1e-12, atol=1e-12)
