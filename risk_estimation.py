import numpy as np

def get_factor_based_covariance(R : np.ndarray, F : np.ndarray):
    """
    # Parameters
    R: T x N return matrix
    F : T x K factor matrix

    where T = training window size, N = asset universe size, K = factor universe size

    # Returns
    N x N factor-based asset return covariance matrix
    """

    # B: K x N factor loadings matrix s.t. R = FB
    # B = inverse(F.T @ F) @ F.T @ R
    B = np.linalg.pinv(F.T @ F) @ F.T @ R

    # T x N residuals
    E = R - (F @ B)

    # Sigma
    Sigma_F = np.cov(F.T)   # K x K factor covariance
    Sigma_E = np.cov(E.T)   # N x N residual covariance

    return (B @ Sigma_F @ B) + Sigma_E