"""Portfolio VaR / Expected Shortfall with regime-shift correlation stress.

Risk factors (USDC numeraire): ETH, wstETH, WBTC, sDAI daily log-returns.
Positions are revalued in full under each scenario:

  * wstETH / sDAI reserves : dV = N (e^r - 1)
  * Uniswap v3 LP          : dV = V_LP(p0 e^r_ETH) - V_LP(p0) + daily fee carry
  * Morpho USDC lender     : short-put payoff on collateral (gap risk)
        b_k(r) = U * max(0, 1 - e^{r_k} (1 - s) / ltv)
        dV     = N_k (APY/365 - b_k(r_k))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.config import CRYPTO_FACTORS, DEFAULT_LLTV, RISK_FACTORS
from src.models import amm_backtester as amm

POSITIONS = ["Morpho wstETH/USDC", "Morpho WBTC/USDC", "Uniswap v3 ETH/USDC", "wstETH reserve", "sDAI reserve"]
ALPHAS = (0.95, 0.99)


@dataclass(frozen=True)
class Exposures:
    morpho_wsteth_usd: float
    morpho_wbtc_usd: float
    lp_capital_usd: float
    wsteth_reserve_usd: float
    sdai_reserve_usd: float
    eth_price: float
    lp_width: float = 0.10
    lp_daily_fee_usd: float = 0.0
    morpho_apy: float = 0.06
    utilization: float = 0.90
    borrower_ltv: float = DEFAULT_LLTV
    liq_slippage: float = 0.02

    @property
    def total(self) -> float:
        return (self.morpho_wsteth_usd + self.morpho_wbtc_usd + self.lp_capital_usd
                + self.wsteth_reserve_usd + self.sdai_reserve_usd)


# ---------------------------------------------------------------------------
# Moments & regime stress
# ---------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices[RISK_FACTORS]).diff().dropna()


def nearest_psd_correlation(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Eigenvalue-clip to PSD, then rescale to unit diagonal."""
    c = 0.5 * (corr + corr.T)
    w, v = np.linalg.eigh(c)
    c = (v * np.maximum(w, eps)) @ v.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return c


def stress_correlation(corr: pd.DataFrame, target: float = 0.90, lam: float = 1.0,
                       block: list[str] = CRYPTO_FACTORS) -> pd.DataFrame:
    """rho'_ij = (1 - lam) rho_ij + lam * target for i != j inside the crypto block."""
    c = corr.to_numpy(dtype=float).copy()
    idx = [corr.columns.get_loc(a) for a in block]
    for i in idx:
        for j in idx:
            if i != j:
                c[i, j] = (1.0 - lam) * c[i, j] + lam * target
    c = nearest_psd_correlation(c)
    return pd.DataFrame(c, index=corr.index, columns=corr.columns)


def regime_moments(returns: pd.DataFrame, regime: str = "normal", vol_mult: float = 2.5,
                   corr_target: float = 0.90, lam: float = 1.0) -> dict:
    mu = returns.mean()
    vol = returns.std(ddof=1).clip(lower=1e-8)   # guard zero-variance factors
    c = returns.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(c, 1.0)
    corr = pd.DataFrame(c, index=returns.columns, columns=returns.columns)
    if regime == "contagion":
        corr_used = stress_correlation(corr, corr_target, lam)
        vol_used = vol * vol_mult
    else:
        corr_used = pd.DataFrame(nearest_psd_correlation(corr.to_numpy()), index=corr.index, columns=corr.columns)
        vol_used = vol
    cov = np.outer(vol, vol) * corr.to_numpy()
    cov_used = np.outer(vol_used, vol_used) * corr_used.to_numpy()
    return {"mu": mu, "vol": vol, "corr": corr, "cov": cov,
            "vol_used": vol_used, "corr_used": corr_used, "cov_used": cov_used}


def _chol(cov: np.ndarray) -> np.ndarray:
    jitter = 1e-14 * np.trace(cov) / cov.shape[0]
    return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))


def recolor_returns(returns: pd.DataFrame, mu: np.ndarray, cov: np.ndarray, cov_stress: np.ndarray) -> np.ndarray:
    """r' = mu + L' L^{-1} (r - mu): imposes stressed dependence, keeps fat-tailed marginal shapes."""
    L, Ls = _chol(cov), _chol(cov_stress)
    centred = returns.to_numpy() - mu
    white = np.linalg.solve(L, centred.T)
    return (mu[:, None] + Ls @ white).T


# ---------------------------------------------------------------------------
# Full-revaluation pricer
# ---------------------------------------------------------------------------
def position_pnl(scen: np.ndarray, ex: Exposures) -> np.ndarray:
    """Scenario factor log-returns (n x 4, order RISK_FACTORS) -> P&L per position (n x 5)."""
    scen = np.atleast_2d(scen)
    r_eth, r_wst, r_btc, r_sdai = scen[:, 0], scen[:, 1], scen[:, 2], scen[:, 3]

    def morpho(notional: float, r: np.ndarray) -> np.ndarray:
        bad = ex.utilization * np.maximum(0.0, 1.0 - np.exp(r) * (1.0 - ex.liq_slippage) / ex.borrower_ltv)
        return notional * (ex.morpho_apy / 365.0 - bad)

    p0 = ex.eth_price
    if ex.lp_capital_usd > 0:
        pa, pb = amm.range_bounds(p0, ex.lp_width)
        L = amm.liquidity_for_value(ex.lp_capital_usd, p0, pa, pb)
        lp = amm.position_value(L, p0 * np.exp(r_eth), pa, pb) - ex.lp_capital_usd + ex.lp_daily_fee_usd
    else:
        lp = np.zeros_like(r_eth)

    return np.column_stack([
        morpho(ex.morpho_wsteth_usd, r_wst),
        morpho(ex.morpho_wbtc_usd, r_btc),
        lp,
        ex.wsteth_reserve_usd * np.expm1(r_wst),
        ex.sdai_reserve_usd * np.expm1(r_sdai),
    ])


def linear_deltas(ex: Exposures) -> tuple[np.ndarray, float]:
    """$ delta per factor (d PnL / d log-return) and deterministic daily carry."""
    lp_delta = 0.0
    if ex.lp_capital_usd > 0:
        pa, pb = amm.range_bounds(ex.eth_price, ex.lp_width)
        L = amm.liquidity_for_value(ex.lp_capital_usd, ex.eth_price, pa, pb)
        x, _ = amm.token_amounts(L, ex.eth_price, pa, pb)
        lp_delta = float(x) * ex.eth_price
    delta = np.array([lp_delta, ex.wsteth_reserve_usd, 0.0, ex.sdai_reserve_usd])
    carry = (ex.morpho_wsteth_usd + ex.morpho_wbtc_usd) * ex.morpho_apy / 365.0 + ex.lp_daily_fee_usd
    return delta, carry


# ---------------------------------------------------------------------------
# Risk measures (inputs are LOSSES X = -PnL)
# ---------------------------------------------------------------------------
def historical_var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return var, max(es, var)


def parametric_var_es(mean_loss: float, sigma: float, alpha: float) -> tuple[float, float]:
    z = stats.norm.ppf(alpha)
    return mean_loss + z * sigma, mean_loss + sigma * stats.norm.pdf(z) / (1.0 - alpha)


def cornish_fisher_z(z, skew: float, exkurt: float):
    return (z + (z**2 - 1) * skew / 6.0 + (z**3 - 3 * z) * exkurt / 24.0
            - (2 * z**3 - 5 * z) * skew**2 / 36.0)


def cornish_fisher_var_es(losses: np.ndarray, alpha: float, n_grid: int = 400) -> tuple[float, float]:
    """Modified VaR; ES = mean of the CF quantile over (alpha, 1). Quantile forced monotone."""
    losses = np.asarray(losses, dtype=float)
    m, sd = losses.mean(), losses.std(ddof=1)
    S, K = stats.skew(losses), stats.kurtosis(losses)  # excess kurtosis
    u = np.linspace(0.5, 0.99999, 20_000)
    q = np.maximum.accumulate(m + sd * cornish_fisher_z(stats.norm.ppf(u), S, K))
    var = float(np.interp(alpha, u, q))
    tail_u = np.linspace(alpha, 0.99999, n_grid)
    es = float(np.mean(np.interp(tail_u, u, q)))
    return var, max(es, var)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def compute_portfolio_risk(daily_prices: pd.DataFrame, ex: Exposures, regime: str = "normal",
                           vol_mult: float = 2.5, corr_target: float = 0.90, lam: float = 1.0,
                           horizon_days: int = 1, n_mc: int = 20_000, seed: int = 42) -> dict:
    rets = log_returns(daily_prices)
    mom = regime_moments(rets, regime, vol_mult, corr_target, lam)
    mu = mom["mu"].to_numpy()
    scale = np.sqrt(horizon_days)

    # Historical scenarios (stressed regime -> recoloured)
    hist_scen = rets.to_numpy() if regime != "contagion" else recolor_returns(rets, mu, mom["cov"], mom["cov_used"])
    pnl_hist_pos = position_pnl(hist_scen, ex)
    loss_hist = -pnl_hist_pos.sum(axis=1)

    # Monte Carlo full revaluation (Gaussian)
    rng = np.random.default_rng(seed)
    Ls = _chol(mom["cov_used"])
    mc_scen = mu + rng.standard_normal((n_mc, len(RISK_FACTORS))) @ Ls.T
    pnl_mc_pos = position_pnl(mc_scen, ex)
    loss_mc = -pnl_mc_pos.sum(axis=1)

    # Delta-normal
    delta, carry = linear_deltas(ex)
    sigma_p = float(np.sqrt(delta @ mom["cov_used"] @ delta))
    mean_loss = -(float(delta @ mu) + carry)

    rows = []
    for method in ("Historical", "Parametric (delta-normal)", "Cornish-Fisher", "Monte Carlo (full reval)"):
        row = {"method": method}
        for a in ALPHAS:
            if method == "Historical":
                v, e = historical_var_es(loss_hist, a)
            elif method.startswith("Parametric"):
                v, e = parametric_var_es(mean_loss, sigma_p, a)
            elif method == "Cornish-Fisher":
                v, e = cornish_fisher_var_es(loss_hist, a)
            else:
                v, e = historical_var_es(loss_mc, a)
            tag = int(round(a * 100))
            row[f"VaR {tag}%"] = v * scale
            row[f"ES {tag}%"] = e * scale
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")

    # Component ES (MC, 99%): E[position loss | portfolio loss >= VaR99]
    var99 = np.quantile(loss_mc, 0.99)
    tail = loss_mc >= var99
    comp = pd.DataFrame({
        "position": POSITIONS,
        "component_ES99": (-pnl_mc_pos[tail]).mean(axis=0) * scale,
        "standalone_ES99": [historical_var_es(-pnl_mc_pos[:, i], 0.99)[1] * scale for i in range(len(POSITIONS))],
    }).set_index("position")

    # Euler component VaR (delta-normal, 99%)
    z99 = stats.norm.ppf(0.99)
    euler = (delta * (mom["cov_used"] @ delta) / sigma_p * z99) if sigma_p > 0 else np.zeros_like(delta)

    return {
        "table": table,
        "table_pct": table / ex.total,
        "corr_normal": mom["corr"],
        "corr_used": mom["corr_used"],
        "vol_normal_ann": mom["vol"] * np.sqrt(365),
        "vol_used_ann": mom["vol_used"] * np.sqrt(365),
        "loss_hist": loss_hist * scale,
        "loss_mc": loss_mc * scale,
        "component_es": comp,
        "euler_var99_by_factor": pd.Series(euler * scale, index=RISK_FACTORS),
        "loss_skew": float(stats.skew(loss_hist)),
        "loss_exkurt": float(stats.kurtosis(loss_hist)),
        "n_obs": len(rets),
    }


# ---------------------------------------------------------------------------
# Hidden counterparty / protocol dependency map
# ---------------------------------------------------------------------------
DEPENDENCIES = ["Morpho Blue", "Chainlink", "Lido", "Uniswap v3", "Circle (USDC)", "BitGo (WBTC)", "Sky/Maker (sDAI)"]
DEFAULT_LGD = {"Morpho Blue": 0.50, "Chainlink": 0.15, "Lido": 0.30, "Uniswap v3": 0.40,
               "Circle (USDC)": 0.10, "BitGo (WBTC)": 0.50, "Sky/Maker (sDAI)": 0.25}


def dependency_matrix(lp_usdc_share: float) -> pd.DataFrame:
    """Fraction of each position's value that depends on each protocol/oracle/issuer."""
    m = pd.DataFrame(0.0, index=POSITIONS, columns=DEPENDENCIES)
    m.loc["Morpho wstETH/USDC", ["Morpho Blue", "Chainlink", "Lido", "Circle (USDC)"]] = 1.0
    m.loc["Morpho WBTC/USDC", ["Morpho Blue", "Chainlink", "BitGo (WBTC)", "Circle (USDC)"]] = 1.0
    m.loc["Uniswap v3 ETH/USDC", "Uniswap v3"] = 1.0
    m.loc["Uniswap v3 ETH/USDC", "Circle (USDC)"] = lp_usdc_share
    m.loc["wstETH reserve", "Lido"] = 1.0
    m.loc["sDAI reserve", "Sky/Maker (sDAI)"] = 1.0
    return m


def dependency_exposure(allocation: dict[str, float], lp_usdc_share: float,
                        lgd: dict[str, float] | None = None) -> dict:
    lgd = {**DEFAULT_LGD, **(lgd or {})}
    m = dependency_matrix(lp_usdc_share)
    alloc = pd.Series(allocation).reindex(POSITIONS).fillna(0.0)
    usd = m.mul(alloc, axis=0)
    gross = usd.sum(axis=0)
    total = float(alloc.sum())
    shares = gross / gross.sum() if gross.sum() > 0 else gross
    summary = pd.DataFrame({
        "gross_exposure_usd": gross,
        "pct_of_aum": gross / total if total else 0.0,
        "lgd": pd.Series(lgd),
        "loss_if_failure_usd": gross * pd.Series(lgd),
    }).reindex(DEPENDENCIES)
    return {
        "matrix_usd": usd,
        "summary": summary.sort_values("gross_exposure_usd", ascending=False),
        "hhi": float((shares**2).sum()),
        "overlap_ratio": float(gross.sum() / total) if total else 0.0,
    }
