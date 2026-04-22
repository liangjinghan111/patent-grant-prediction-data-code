"""
EM algorithm for cure-mixture + truncated log-normal AFT model.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtr

from .distributions import logf0_trunc, logS0_trunc
from .likelihood import compute_loglik

_PI_MIN, _PI_MAX = 1e-12, 1.0 - 1e-12
_ETA_MIN, _ETA_MAX = -50.0, 50.0
_LOG_SIGMA_MIN, _LOG_SIGMA_MAX = -5.0, 5.0
_T_MIN = 1e-12
_L2_PENALTY = 1e-6
_L2_PENALTY_ITERS = 10


def _init_alpha(dataset):
    """
    用 group1 vs group2 做 logistic 回归得到 alpha_init，不包含 group3。
    y=1 (group1), y=0 (group2).
    """
    Z = np.asarray(dataset.Z, dtype=np.float64)
    group = np.asarray(dataset.group, dtype=np.float64).ravel()
    mask12 = (group == 1) | (group == 2)
    if not mask12.any():
        return np.zeros(Z.shape[1], dtype=np.float64)

    Z12 = Z[mask12]
    y12 = (group[mask12] == 1).astype(np.float64)
    q_dim = Z.shape[1]

    def neg_ll(a):
        pi_a = expit(Z12 @ a)
        pi_a = np.clip(pi_a, _PI_MIN, _PI_MAX)
        return -np.sum(y12 * np.log(pi_a) + (1.0 - y12) * np.log1p(-pi_a))

    res = minimize(neg_ll, np.zeros(q_dim), method="L-BFGS-B", options={"maxiter": 500})
    return np.asarray(res.x, dtype=np.float64)


def _init_gamma_log_sigma(dataset):
    """
    只用 group1 做 untruncated log-normal 回归：
    log t = X @ gamma + sigma * eps → gamma 用 OLS，sigma = std(residuals).
    """
    t = np.asarray(dataset.t, dtype=np.float64).ravel()
    X = np.asarray(dataset.X, dtype=np.float64)
    group = np.asarray(dataset.group, dtype=np.float64).ravel()
    mask1 = group == 1

    if not mask1.any():
        return np.zeros(X.shape[1], dtype=np.float64), 0.0

    t1 = np.maximum(t[mask1], _T_MIN)
    X1 = X[mask1]
    log_t1 = np.log(t1)
    n1, p = X1.shape

    if n1 < p + 1:
        return np.zeros(p, dtype=np.float64), 0.0

    gamma, _, _, _ = np.linalg.lstsq(X1, log_t1, rcond=None)
    gamma = np.asarray(gamma, dtype=np.float64)
    eta1 = X1 @ gamma
    eta1 = np.clip(eta1, _ETA_MIN, _ETA_MAX)
    resid = log_t1 - eta1
    sigma = np.std(resid)
    sigma = max(sigma, 1e-6)
    log_sigma = np.clip(np.log(sigma), _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
    return gamma, float(log_sigma)


def _get_pi(Z: np.ndarray, incidence_state) -> np.ndarray:
    """π from incidence state: alpha (logistic) or fitted booster (predict_proba)."""
    if isinstance(incidence_state, np.ndarray):
        pi = expit(Z @ np.asarray(incidence_state, dtype=np.float64).ravel())
    else:
        pi = incidence_state.predict_proba(Z)[:, 1]
    return np.clip(pi, _PI_MIN, _PI_MAX)


def e_step(dataset, incidence_state, gamma: np.ndarray, log_sigma: float):
    """
    E-step: 计算软标签 q_i。
    incidence_state: alpha (ndarray) for logistic, or fitted booster for π.
    """
    t = np.asarray(dataset.t, dtype=np.float64).ravel()
    group = np.asarray(dataset.group, dtype=np.float64).ravel()
    X = np.asarray(dataset.X, dtype=np.float64)
    Z = np.asarray(dataset.Z, dtype=np.float64)
    H = float(dataset.H)

    sigma = np.exp(float(log_sigma))
    eta = X @ np.asarray(gamma, dtype=np.float64).ravel()
    eta = np.clip(eta, _ETA_MIN, _ETA_MAX)

    pi = _get_pi(Z, incidence_state)

    logS0 = logS0_trunc(t, eta, sigma, H)

    q = np.zeros(len(t), dtype=np.float64)
    q[group == 1] = 1.0
    q[group == 2] = 0.0
    mask3 = group == 3
    if mask3.any():
        logA = np.log(pi[mask3]) + logS0[mask3]
        logB = np.log1p(-pi[mask3])
        q[mask3] = np.exp(logA - logsumexp(np.stack([logA, logB], axis=1), axis=1))

    return q


def m_step_extensive(dataset, q: np.ndarray, alpha_init: np.ndarray, weight_g3: float | None = None):
    """
    最大化 Σ_i w_i [ q_i log π_i + (1-q_i) log(1-π_i) ]，π_i = σ(Z_i'α)。
    与正文一致：加权 binomial GLM，用 IRLS（Fisher scoring）求解 α。
    weight_g3 ∈ (0,1] 时，对 group==3 的样本将 w_i 乘以该系数（调参用；基线仅 G1+G2 时通常为 None）。
    """
    Z = np.asarray(dataset.Z, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).ravel()
    group = np.asarray(dataset.group, dtype=np.float64).ravel()
    n, d = Z.shape
    w_sw = np.ones(n, dtype=np.float64)
    if weight_g3 is not None and 0.0 < float(weight_g3) <= 1.0:
        w_sw[group == 3] *= float(weight_g3)

    alpha = np.asarray(alpha_init, dtype=np.float64).ravel().copy()

    for _ in range(200):
        eta = Z @ alpha
        eta = np.clip(eta, _ETA_MIN, _ETA_MAX)
        pi = expit(eta)
        pi = np.clip(pi, _PI_MIN, _PI_MAX)
        var = np.maximum(pi * (1.0 - pi), 1e-12)
        z_work = eta + (q - pi) / var
        W_eff = w_sw * var
        W_eff = np.maximum(W_eff, 1e-12)
        sw = np.sqrt(W_eff)
        ZW = Z * sw[:, None]
        tgt = z_work * sw
        alpha_new, *_ = np.linalg.lstsq(ZW, tgt, rcond=None)
        alpha_new = np.asarray(alpha_new, dtype=np.float64).ravel()
        if np.max(np.abs(alpha_new - alpha)) < 1e-10:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha


def m_step_extensive_adaboost(dataset, q: np.ndarray, n_estimators: int = 25, max_depth: int = 2, learning_rate: float = 1.0, random_state: int = 42, weight_g3: float | None = None, use_soft_labels_g3: bool = False):
    """
    M-step for incidence using AdaBoost within EM: fit AdaBoost to soft labels q.
    use_soft_labels_g3: if True, for group-3 use duplicated (z,1) with weight q and (z,0) with weight 1-q
        instead of single hard label (recommended to better exploit g3; see e.g. soft-label / probability weighting in EM).
    weight_g3: if in (0,1], scale group-3 sample weights (for both hard and soft formulations).
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    Z = np.asarray(dataset.Z, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).ravel()
    group = np.asarray(dataset.group, dtype=np.float64).ravel()
    mask1 = group == 1
    mask2 = group == 2
    mask3 = group == 3

    if use_soft_labels_g3 and mask3.any():
        # Soft-label formulation for g3: each g3 sample becomes two weighted samples (z,1,q) and (z,0,1-q)
        Z_g1, Z_g2 = Z[mask1], Z[mask2]
        Z_g3 = Z[mask3]
        q_g3 = np.clip(q[mask3], _PI_MIN, _PI_MAX)
        w_g3 = weight_g3 if (weight_g3 is not None and 0 < weight_g3 <= 1.0) else 1.0
        n1, n2, n3 = Z_g1.shape[0], Z_g2.shape[0], Z_g3.shape[0]
        Z_exp = np.vstack([Z_g1, Z_g2, Z_g3, Z_g3])
        y_exp = np.array([1] * n1 + [0] * n2 + [1] * n3 + [0] * n3, dtype=np.int64)
        w_exp = np.concatenate([
            2.0 * np.ones(n1),
            2.0 * np.ones(n2),
            w_g3 * q_g3,
            w_g3 * (1.0 - q_g3),
        ])
        sample_weight = w_exp
        Z_fit, y_fit = Z_exp, y_exp
    else:
        y_hard = (q >= 0.5).astype(np.int64)
        sample_weight = 2.0 * np.maximum(q, 1.0 - q)
        if weight_g3 is not None and 0 < weight_g3 <= 1.0 and mask3.any():
            sample_weight = sample_weight.copy()
            sample_weight[mask3] *= weight_g3
        Z_fit, y_fit = Z, y_hard

    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    clf.fit(Z_fit, y_fit, sample_weight=sample_weight)
    return clf


def _logf0_logS0_grad_eta_sigma(t, eta, sigma, H):
    """
    Analytic gradient of logf0_trunc and logS0_trunc w.r.t. eta and sigma.

    logf0 = log f(t) - log F(H), z = (log t - eta)/sigma, z_H = (log H - eta)/sigma
    λ(x) = φ(x)/Φ(x) = exp(-x²/2) / (sqrt(2π) * Φ(x))

    ∂logf0/∂eta = (z + λ(z_H)) / sigma
    ∂logf0/∂sigma = (z² - 1 + λ(z_H)*z_H) / sigma

    logS0 = log(1 - F0), F0 = exp(logF0)
    ∂logF0/∂eta = (λ(z_H) - λ(z)) / sigma
    ∂logF0/∂sigma = (λ(z_H)*z_H - λ(z)*z) / sigma
    ∂logS0/∂eta = (-F0/(1-F0)) * ∂logF0/∂eta
    ∂logS0/∂sigma = (-F0/(1-F0)) * ∂logF0/∂sigma

    Returns: (dlogf0_deta, dlogf0_dsigma, dlogS0_deta, dlogS0_dsigma)
    """
    _EPS = 1e-12
    t = np.asarray(t, dtype=np.float64)
    t = np.clip(t, _EPS, float(H))
    log_t = np.log(t)
    eta = np.asarray(eta, dtype=np.float64)
    sigma = float(sigma)

    z = (log_t - eta) / sigma
    z_H = (np.log(H) - eta) / sigma

    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    phi_zH = np.exp(-0.5 * z_H * z_H) / np.sqrt(2 * np.pi)
    Phi_z = np.maximum(ndtr(z), 1e-300)
    Phi_zH = np.maximum(ndtr(z_H), 1e-300)
    lam_z = phi_z / Phi_z
    lam_zH = phi_zH / Phi_zH

    dlogf0_deta = (z + lam_zH) / sigma
    dlogf0_dsigma = (z * z - 1.0 + lam_zH * z_H) / sigma

    logF0 = np.log(np.maximum(Phi_z / Phi_zH, 1e-300))
    F0 = np.exp(np.clip(logF0, -500.0, 0.0))
    F0 = np.minimum(F0, 1.0 - 1e-15)
    coef = -F0 / (1.0 - F0)

    dlogF0_deta = (lam_zH - lam_z) / sigma
    dlogF0_dsigma = (lam_zH * z_H - lam_z * z) / sigma
    dlogS0_deta = coef * dlogF0_deta
    dlogS0_dsigma = coef * dlogF0_dsigma

    return dlogf0_deta, dlogf0_dsigma, dlogS0_deta, dlogS0_dsigma


def m_step_timing(
    dataset,
    q: np.ndarray,
    gamma_init: np.ndarray,
    log_sigma_init: float,
    iter_idx=None,
):
    """
    最大化 Σ_i q_i [ I(group==1) log f0(t_i) + I(group==3) log S0(t_i) ]。
    Group2 绝不进入。使用 analytic gradient，L-BFGS-B，log_sigma ∈ [-5,5]。
    前 _L2_PENALTY_ITERS 次迭代启用 L2 penalty: 1e-6 * ||gamma||^2。
    返回 gamma_new, log_sigma_new。
    """
    t = np.asarray(dataset.t, dtype=np.float64).ravel()
    group = np.asarray(dataset.group, dtype=np.float64).ravel()
    X = np.asarray(dataset.X, dtype=np.float64)
    H = float(dataset.H)
    q = np.asarray(q, dtype=np.float64).ravel()
    p = X.shape[1]
    n = len(t)
    use_l2 = iter_idx is not None and iter_idx < _L2_PENALTY_ITERS

    mask1 = group == 1
    mask3 = group == 3

    def neg_obj_and_grad(params):
        g = params[:p]
        ls = np.clip(params[p], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
        sig = np.exp(ls)
        eta = np.clip(X @ g, _ETA_MIN, _ETA_MAX)

        dlogf0_deta, dlogf0_dsigma, dlogS0_deta, dlogS0_dsigma = _logf0_logS0_grad_eta_sigma(
            t, eta, sig, H
        )

        logf0 = logf0_trunc(t, eta, sig, H)
        logS0 = logS0_trunc(t, eta, sig, H)
        ll = np.sum(q[mask1] * logf0[mask1]) + np.sum(q[mask3] * logS0[mask3])

        w = np.zeros(n, dtype=np.float64)
        w[mask1] = q[mask1] * dlogf0_deta[mask1]
        w[mask3] = q[mask3] * dlogS0_deta[mask3]
        dll_deta = w

        dll_dsigma = np.sum(q[mask1] * dlogf0_dsigma[mask1]) + np.sum(
            q[mask3] * dlogS0_dsigma[mask3]
        )

        dll_dgamma = X.T @ dll_deta
        dll_dlog_sigma = dll_dsigma * sig

        if use_l2:
            penalty = _L2_PENALTY * np.sum(g * g)
            dpenalty_dgamma = 2.0 * _L2_PENALTY * g
            ll = ll - penalty
            dll_dgamma = dll_dgamma - dpenalty_dgamma

        grad = np.concatenate([dll_dgamma, [dll_dlog_sigma]])
        return -ll, -grad

    def obj_jac(params):
        f, g = neg_obj_and_grad(params)
        return f, g

    x0 = np.concatenate([gamma_init, [log_sigma_init]])
    res = minimize(
        obj_jac,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None)] * p + [(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)],
        options={"maxiter": 500},
    )
    gamma_new = np.asarray(res.x[:p], dtype=np.float64)
    log_sigma_new = float(np.clip(res.x[p], _LOG_SIGMA_MIN, _LOG_SIGMA_MAX))
    return gamma_new, log_sigma_new


def run_em(
    dataset,
    max_iter: int = 200,
    tol: float = 1e-6,
    incidence: str = "logistic",
    adaboost_kwargs: dict | None = None,
    init_gamma: np.ndarray | None = None,
    init_log_sigma: float | None = None,
    init_incidence_state=None,
    weight_g3: float | None = None,
    use_soft_labels_g3: bool = False,
):
    """
    EM 主循环：E-step → M-step extensive → M-step timing → observed loglik。

    incidence: "logistic"（正文：加权 logistic / IRLS）| "adaboost"（实现备选）。
    weight_g3：logistic 时在 m_step_extensive 中对 G3 行的似然项乘以该系数；adaboost 时并入 AdaBoost 的权重逻辑。
    use_soft_labels_g3：仅 incidence="adaboost" 时生效。
    可选 warm start：传入 init_gamma, init_log_sigma, init_incidence_state（logistic 时为 alpha 向量，adaboost 时为已拟合的 booster），
    则跳过默认初始化，从给定状态继续迭代（用于先在小数据集 g1+g2 上预热再在含 g3 的全集上跑）。
    """
    em_limit = max_iter
    q_dim = dataset.Z.shape[1]
    if init_gamma is not None and init_log_sigma is not None:
        gamma = np.asarray(init_gamma, dtype=np.float64).ravel()
        log_sigma = float(init_log_sigma)
        if gamma.shape[0] != dataset.X.shape[1]:
            gamma = _init_gamma_log_sigma(dataset)[0]
            log_sigma = _init_gamma_log_sigma(dataset)[1]
    else:
        gamma, log_sigma = _init_gamma_log_sigma(dataset)
    if init_incidence_state is not None:
        incidence_state = init_incidence_state
    else:
        if incidence == "logistic":
            incidence_state = _init_alpha(dataset)
        else:
            incidence_state = _init_alpha(dataset)
    adaboost_kwargs = dict(adaboost_kwargs or {})
    if weight_g3 is not None and "weight_g3" not in adaboost_kwargs:
        adaboost_kwargs["weight_g3"] = weight_g3
    if use_soft_labels_g3 and "use_soft_labels_g3" not in adaboost_kwargs:
        adaboost_kwargs["use_soft_labels_g3"] = True

    loglik_history = []
    loglik_old = -np.inf

    for it in range(em_limit):
        q = e_step(dataset, incidence_state, gamma, log_sigma)
        if incidence == "logistic":
            a0 = incidence_state if isinstance(incidence_state, np.ndarray) else _init_alpha(dataset)
            incidence_state = m_step_extensive(dataset, q, a0, weight_g3=weight_g3)
        else:
            incidence_state = m_step_extensive_adaboost(dataset, q, **adaboost_kwargs)

        pi = _get_pi(dataset.Z, incidence_state)
        _, total_loglik = compute_loglik(dataset, gamma, log_sigma, pi=pi)
        loglik_history.append(total_loglik)
        diff = abs(total_loglik - loglik_old)

        eta = np.clip(dataset.X @ gamma, _ETA_MIN, _ETA_MAX)
        sigma = np.exp(log_sigma)
        logf0 = logf0_trunc(dataset.t, eta, sigma, dataset.H)
        logS0 = logS0_trunc(dataset.t, eta, sigma, dataset.H)
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        group = np.asarray(dataset.group, dtype=np.float64).ravel()
        mask1 = group == 1
        mask3 = group == 3
        Q_extensive = float(np.sum(q_arr * np.log(pi) + (1.0 - q_arr) * np.log1p(-pi)))
        Q_timing = float(np.sum(q_arr[mask1] * logf0[mask1]) + np.sum(q_arr[mask3] * logS0[mask3]))

        print(
            f"iter {it + 1}, observed_loglik {total_loglik:.6f}, "
            f"Q_extensive {Q_extensive:.6f}, Q_timing {Q_timing:.6f}, diff {diff:.6e}"
        )
        loglik_old = total_loglik

        if diff < tol:
            print(f"Converged at iter {it + 1}")
            break

    return incidence_state, gamma, log_sigma, loglik_history


def main():
    print("本独立包请运行: python scripts/run_exp1_v8.py（见 README）")


if __name__ == "__main__":
    main()
