"""
Log-normal AFT with horizon truncation. All probabilities in log for numerical stability.
Model: log T = eta + sigma * eps, eps ~ N(0,1), eta = x^T gamma, sigma > 0.
"""
import numpy as np
from scipy.special import log_ndtr

_EPS = 1e-12
_LOG2 = np.log(2.0)


def _clip_t(t: np.ndarray, H: float) -> np.ndarray:
    """Clip t to [1e-12, H] for safe log and CDF."""
    t = np.asarray(t, dtype=np.float64)
    return np.clip(t, _EPS, float(H))


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """
    log(1 - exp(-x)) for x > 0, numerically stable.
    x > log(2): use log1p(-exp(-x))
    x <= log(2): use log(-expm1(-x))
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    large = x > _LOG2
    out[large] = np.log1p(-np.exp(-x[large]))
    out[~large] = np.log(-np.expm1(-x[~large]))
    return out


def logF_lognormal(t: np.ndarray, eta: np.ndarray, sigma: float) -> np.ndarray:
    """Log CDF of log-normal: P(T <= t). Returns log F(t)."""
    t = np.asarray(t, dtype=np.float64)
    z = (np.log(np.maximum(t, _EPS)) - eta) / sigma
    return log_ndtr(z)


def logf_lognormal(t: np.ndarray, eta: np.ndarray, sigma: float) -> np.ndarray:
    """Log PDF of log-normal. Returns log f(t)."""
    t = np.asarray(t, dtype=np.float64)
    log_t = np.log(np.maximum(t, _EPS))
    z = (log_t - eta) / sigma
    return -log_t - np.log(sigma) - 0.5 * (np.log(2 * np.pi) + z * z)


def logF0_trunc(t: np.ndarray, eta: np.ndarray, sigma: float, H: float) -> np.ndarray:
    """Truncated CDF: log(F(t)/F(H)) = log F(t) - log F(H)."""
    t = _clip_t(t, H)
    log_F_t = logF_lognormal(t, eta, sigma)
    log_F_H = logF_lognormal(np.full_like(t, H), eta, sigma)
    return log_F_t - log_F_H


def logf0_trunc(t: np.ndarray, eta: np.ndarray, sigma: float, H: float) -> np.ndarray:
    """Truncated PDF: log(f(t)/F(H)) = log f(t) - log F(H)."""
    t = _clip_t(t, H)
    log_f_t = logf_lognormal(t, eta, sigma)
    log_F_H = logF_lognormal(np.full_like(t, H), eta, sigma)
    return log_f_t - log_F_H


def logS0_trunc(t: np.ndarray, eta: np.ndarray, sigma: float, H: float) -> np.ndarray:
    """
    Truncated survival: log(1 - F0(t)) = log(1 - exp(logF0)), numerically stable.
    Uses log1mexp: log(1 - exp(logF0)) = log1mexp(-logF0) when logF0 <= 0.
    """
    t = _clip_t(t, H)
    logF0 = logF0_trunc(t, eta, sigma, H)
    # log(1 - exp(logF0)). When logF0 <= 0: let x = -logF0 >= 0, then log(1 - exp(-x)) = log1mexp(x)
    x = -logF0
    x = np.maximum(x, 1e-300)  # avoid x=0 (would give log1mexp(0)=log0=-inf)
    return _log1mexp(x)


def self_test() -> None:
    """Check outputs are finite, logF0_trunc(H,...) ≈ 0."""
    np.random.seed(42)
    H = 48.0
    eta = np.array(0.5, dtype=np.float64)
    sigma = 1.2

    # t near 0 and near H
    t_lo = np.array([_EPS, 1e-8, 1e-6, 1e-4, 0.1], dtype=np.float64)
    t_hi = np.array([H - 1e-6, H - 1e-4, H - 0.1, H - 1e-12, H], dtype=np.float64)
    t_mid = np.linspace(1.0, H - 1.0, 10)
    t_all = np.concatenate([t_lo, t_mid, t_hi])

    eta_arr = np.full(len(t_all), eta)
    assert np.all(np.isfinite(logF_lognormal(t_all, eta_arr, sigma))), "logF nan/inf"
    assert np.all(np.isfinite(logf_lognormal(t_all, eta_arr, sigma))), "logf nan/inf"
    assert np.all(np.isfinite(logF0_trunc(t_all, eta_arr, sigma, H))), "logF0 nan/inf"
    assert np.all(np.isfinite(logf0_trunc(t_all, eta_arr, sigma, H))), "logf0 nan/inf"
    assert np.all(np.isfinite(logS0_trunc(t_all, eta_arr, sigma, H))), "logS0 nan/inf"

    # logF0_trunc(H, ...) = log(1) = 0
    t_H = np.array([H], dtype=np.float64)
    logF0_at_H = logF0_trunc(t_H, np.array([eta]), sigma, H)
    assert np.allclose(logF0_at_H, 0.0, atol=1e-10), f"logF0_trunc(H,...) = {logF0_at_H[0]} != 0"

    print("OK")


if __name__ == "__main__":
    self_test()
