"""
Observed log-likelihood for cure-mixture AFT model.
"""
import numpy as np
from scipy.special import expit, logsumexp

from .distributions import logf0_trunc, logS0_trunc

_PI_MIN, _PI_MAX = 1e-12, 1.0 - 1e-12
_ETA_MIN, _ETA_MAX = -50.0, 50.0


def compute_loglik(dataset, gamma: np.ndarray, log_sigma: float, alpha: np.ndarray | None = None, pi: np.ndarray | None = None):
    """
    Compute per-observation and total observed log-likelihood.
    Provide either alpha (for logistic π = sigmoid(Z@α)) or precomputed pi (e.g. from booster).

    Returns:
        per_obs_loglik: (n,)
        total_loglik: scalar
    """
    t = np.asarray(dataset.t, dtype=np.float64).ravel()
    group = np.asarray(dataset.group, dtype=np.float64).ravel()
    X = np.asarray(dataset.X, dtype=np.float64)
    Z = np.asarray(dataset.Z, dtype=np.float64)
    H = float(dataset.H)

    n = len(t)
    sigma = np.exp(float(log_sigma))
    eta = X @ np.asarray(gamma, dtype=np.float64).ravel()
    eta = np.clip(eta, _ETA_MIN, _ETA_MAX)

    if pi is not None:
        pi = np.asarray(pi, dtype=np.float64).ravel()
        pi = np.clip(pi, _PI_MIN, _PI_MAX)
    else:
        if alpha is None:
            raise ValueError("compute_loglik: provide either alpha or pi")
        pi = expit(Z @ np.asarray(alpha, dtype=np.float64).ravel())
        pi = np.clip(pi, _PI_MIN, _PI_MAX)

    logf0 = logf0_trunc(t, eta, sigma, H)
    logS0 = logS0_trunc(t, eta, sigma, H)

    per_obs_loglik = np.empty(n, dtype=np.float64)
    mask1 = group == 1
    mask2 = group == 2
    mask3 = group == 3

    per_obs_loglik[mask1] = np.log(pi[mask1]) + logf0[mask1]
    per_obs_loglik[mask2] = np.log1p(-pi[mask2])
    per_obs_loglik[mask3] = logsumexp(
        np.stack([np.log1p(-pi[mask3]), np.log(pi[mask3]) + logS0[mask3]], axis=1),
        axis=1,
    )

    total_loglik = float(np.sum(per_obs_loglik))
    return per_obs_loglik, total_loglik
