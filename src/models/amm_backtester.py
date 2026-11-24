"""Uniswap v3 concentrated-liquidity math and strategy backtester.

Conventions: price ``p`` is token1 per token0 (USDC per ETH). ``x`` = ETH
amount, ``y`` = USDC amount. With sqrt prices a = sqrt(p_a), b = sqrt(p_b),
s = sqrt(p), the closed-form v3 reserve equations are

    p_a < p < p_b :  x = L (1/s - 1/b),  y = L (s - a)
    p <= p_a      :  x = L (1/a - 1/b),  y = 0
    p >= p_b      :  x = 0,              y = L (b - a)

which is the in-range invariant (x + L/b)(y + L a) = L^2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import UNIV3_FEE_TIER


# ---------------------------------------------------------------------------
# Closed-form position math
# ---------------------------------------------------------------------------
def range_bounds(p0: float, width: float) -> tuple[float, float]:
    """Symmetric percentage range [p0(1-w), p0(1+w)]."""
    if not 0 < width < 1:
        raise ValueError("width must be in (0, 1)")
    return p0 * (1.0 - width), p0 * (1.0 + width)


def token_amounts(L: float, p, pa: float, pb: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    a, b = np.sqrt(pa), np.sqrt(pb)
    s = np.clip(np.sqrt(p), a, b)
    x = L * (1.0 / s - 1.0 / b)
    y = L * (s - a)
    return x, y


def position_value(L: float, p, pa: float, pb: float) -> np.ndarray:
    """USD value of a v3 position with liquidity L at price p."""
    x, y = token_amounts(L, p, pa, pb)
    return x * np.asarray(p, dtype=float) + y


def liquidity_for_value(value: float, p: float, pa: float, pb: float) -> float:
    """Liquidity L that a USD amount buys when deposited at price p."""
    unit = float(position_value(1.0, p, pa, pb))
    if unit <= 0:
        raise ValueError("degenerate range")
    return value / unit


def eth_value_share(p: float, pa: float, pb: float) -> float:
    """Fraction of position value held in ETH at price p (scale-free)."""
    x, _ = token_amounts(1.0, p, pa, pb)
    return float(x * p / position_value(1.0, p, pa, pb))


def impermanent_loss(p, p0: float, pa: float, pb: float) -> np.ndarray:
    """IL(p) = V_LP(p) / V_hold(p) - 1, V_hold = initial LP tokens held passively."""
    L = 1.0
    x0, y0 = token_amounts(L, p0, pa, pb)
    p = np.asarray(p, dtype=float)
    v_hold = x0 * p + y0
    return position_value(L, p, pa, pb) / v_hold - 1.0


def il_closed_form_in_range(r, p0: float, pa: float, pb: float) -> np.ndarray:
    """Closed-form v3 IL for p = r * p0 inside [pa, pb]:

        IL(r) = (2 sqrt(r) - 1 - r) / (1 + r - sqrt(pa/p0) - r sqrt(p0/pb))
    """
    r = np.asarray(r, dtype=float)
    num = 2.0 * np.sqrt(r) - 1.0 - r
    den = 1.0 + r - np.sqrt(pa / p0) - r * np.sqrt(p0 / pb)
    return num / den


def lvr_per_period(L: float, p, sigma_period: float, pa: float, pb: float) -> np.ndarray:
    """Loss-Versus-Rebalancing (Milionis et al. 2022) for a v3 position.

    LVR rate = sigma^2 p^2 |dx/dp| / 2 with dx/dp = -L / (2 p^{3/2}) in range,
    i.e. LVR = sigma^2 L sqrt(p) / 4 per unit time; zero when out of range.
    """
    p = np.asarray(p, dtype=float)
    in_rng = (p >= pa) & (p <= pb)
    return np.where(in_rng, sigma_period**2 * L * np.sqrt(p) / 4.0, 0.0)


def pool_liquidity_from_tvl(tvl_usd, p, ref_width: float = 0.20) -> np.ndarray:
    """Competing pool liquidity implied by TVL concentrated at +/- ref_width around p.

    L_pool = TVL / (sqrt(p) * (2 - 1/sqrt(1+w) - sqrt(1-w)))
    """
    p = np.asarray(p, dtype=float)
    k = 2.0 - 1.0 / np.sqrt(1.0 + ref_width) - np.sqrt(1.0 - ref_width)
    return np.asarray(tvl_usd, dtype=float) / (np.sqrt(p) * k)


# ---------------------------------------------------------------------------
# Strategy backtests
# ---------------------------------------------------------------------------
def _realized_sigma_per_period(prices: np.ndarray) -> float:
    lr = np.diff(np.log(prices))
    return float(np.std(lr, ddof=1)) if lr.size > 1 else 0.0


def backtest_hodl(prices: np.ndarray, capital: float) -> pd.DataFrame:
    p0 = prices[0]
    x, y = capital / (2.0 * p0), capital / 2.0
    value = x * prices + y
    return pd.DataFrame({
        "value": value, "cum_fees": 0.0, "cum_costs": 0.0, "il_usd": 0.0,
        "cum_lvr": 0.0, "in_range": True, "rebalance": False,
        "net_pnl": value - capital,
    })


def backtest_passive_lp(prices: np.ndarray, volumes: np.ndarray, capital: float, width: float,
                        pool_tvl: float, fee_tier: float = UNIV3_FEE_TIER,
                        ref_width: float = 0.20) -> pd.DataFrame:
    p0 = prices[0]
    pa, pb = range_bounds(p0, width)
    L = liquidity_for_value(capital, p0, pa, pb)
    x0, y0 = token_amounts(L, p0, pa, pb)
    value = position_value(L, prices, pa, pb)
    in_range = (prices >= pa) & (prices <= pb)
    L_pool = pool_liquidity_from_tvl(pool_tvl, prices, ref_width)
    fees = fee_tier * volumes * (L / (L + L_pool)) * in_range
    sigma = _realized_sigma_per_period(prices)
    lvr = lvr_per_period(L, prices, sigma, pa, pb)
    cum_fees = np.cumsum(fees)
    return pd.DataFrame({
        "value": value, "cum_fees": cum_fees, "cum_costs": 0.0,
        "il_usd": value - (x0 * prices + y0), "cum_lvr": np.cumsum(lvr),
        "in_range": in_range, "rebalance": False,
        "range_low": pa, "range_high": pb,
        "net_pnl": value + cum_fees - capital,
    })


def backtest_rebalanced_lp(prices: np.ndarray, volumes: np.ndarray, capital: float, width: float,
                           pool_tvl: float, rebalance_after: int = 6, gas_usd: float = 40.0,
                           slippage_bps: float = 5.0, fee_tier: float = UNIV3_FEE_TIER,
                           ref_width: float = 0.20) -> pd.DataFrame:
    """Re-centre the range after ``rebalance_after`` consecutive out-of-range periods.

    Rebalance cost = gas + |ETH value to swap| * (fee_tier + slippage). The swap
    notional is the gap between the current ETH value and the ETH value the new
    centred range requires at the current price.
    """
    n = prices.size
    p0 = prices[0]
    pa, pb = range_bounds(p0, width)
    L = liquidity_for_value(capital, p0, pa, pb)
    x0, y0 = token_amounts(L, p0, pa, pb)
    sigma = _realized_sigma_per_period(prices)
    L_pool = pool_liquidity_from_tvl(pool_tvl, prices, ref_width)

    value = np.empty(n); fees = np.zeros(n); costs = np.zeros(n); lvr = np.zeros(n)
    in_range = np.zeros(n, dtype=bool); rebal = np.zeros(n, dtype=bool)
    lo = np.empty(n); hi = np.empty(n)
    out_count = 0
    for t in range(n):
        p = prices[t]
        inr = pa <= p <= pb
        in_range[t] = inr
        if inr:
            fees[t] = fee_tier * volumes[t] * L / (L + L_pool[t])
            lvr[t] = sigma**2 * L * np.sqrt(p) / 4.0
            out_count = 0
        else:
            out_count += 1
        v = float(position_value(L, p, pa, pb))
        if out_count >= rebalance_after and t < n - 1:
            x_cur, _ = token_amounts(L, p, pa, pb)
            npa, npb = range_bounds(p, width)
            target_eth = eth_value_share(p, npa, npb) * v
            swap_notional = abs(target_eth - float(x_cur) * p)
            cost = gas_usd + swap_notional * (fee_tier + slippage_bps / 1e4)
            v -= cost
            costs[t] = cost
            pa, pb = npa, npb
            L = liquidity_for_value(v, p, pa, pb)
            rebal[t] = True
            out_count = 0
        value[t] = v
        lo[t], hi[t] = pa, pb

    cum_fees = np.cumsum(fees)
    return pd.DataFrame({
        "value": value, "cum_fees": cum_fees, "cum_costs": np.cumsum(costs),
        "il_usd": value - (x0 * prices + y0), "cum_lvr": np.cumsum(lvr),
        "in_range": in_range, "rebalance": rebal,
        "range_low": lo, "range_high": hi,
        "net_pnl": value + cum_fees - capital,
    })


def run_backtest(price: pd.Series, volume_usd: pd.Series, capital: float, width: float = 0.10,
                 pool_tvl: float = 150e6, rebalance_after: int = 6, gas_usd: float = 40.0,
                 slippage_bps: float = 5.0, fee_tier: float = UNIV3_FEE_TIER,
                 ref_width: float = 0.20) -> dict:
    """Run HODL / passive LP / auto-rebalanced LP on the same price & volume series."""
    if len(price) < 2:
        raise ValueError("need at least two observations")
    prices = price.to_numpy(dtype=float)
    vols = volume_usd.to_numpy(dtype=float)
    idx = price.index
    days = (idx[-1] - idx[0]).total_seconds() / 86400.0 if isinstance(idx, pd.DatetimeIndex) else len(prices) / 24.0
    days = max(days, 1e-9)

    out = {
        "HODL 50/50": backtest_hodl(prices, capital),
        "Passive LP": backtest_passive_lp(prices, vols, capital, width, pool_tvl, fee_tier, ref_width),
        "Auto-Rebalanced LP": backtest_rebalanced_lp(prices, vols, capital, width, pool_tvl,
                                                     rebalance_after, gas_usd, slippage_bps,
                                                     fee_tier, ref_width),
    }
    for df in out.values():
        df.index = idx
        df["price"] = prices

    summary = []
    for name, df in out.items():
        fees = float(df["cum_fees"].iloc[-1])
        summary.append({
            "strategy": name,
            "final_value": float(df["value"].iloc[-1]),
            "fees_usd": fees,
            "costs_usd": float(df["cum_costs"].iloc[-1]),
            "il_usd": float(df["il_usd"].iloc[-1]),
            "lvr_usd": float(df["cum_lvr"].iloc[-1]),
            "net_pnl_usd": float(df["net_pnl"].iloc[-1]),
            "net_return": float(df["net_pnl"].iloc[-1]) / capital,
            "fee_apr": fees / capital * 365.0 / days,
            "il_pct": float(df["il_usd"].iloc[-1]) / capital,
            "time_in_range": float(df["in_range"].mean()),
            "rebalances": int(df["rebalance"].sum()),
        })
    return {"series": out, "summary": pd.DataFrame(summary).set_index("strategy"), "days": days}
