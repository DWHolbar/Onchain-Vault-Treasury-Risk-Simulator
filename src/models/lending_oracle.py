"""Morpho Blue lending vault: oracle lag, liquidation cascade and bad debt.

Key closed forms
----------------
Liquidation Incentive Factor (Morpho Blue):
    LIF = min(M, 1 / (beta * LLTV + (1 - beta))),  M = 1.15, beta = 0.3

Constant-product DEX selling q USD of collateral into a pool with USD depth R:
    average execution slippage  s(q) = q / (R + q)
    post-trade marginal price   P' = P * (R / (R + q))^2

A liquidator repays debt valued at the ORACLE price O and receives LIF times
that in collateral, which it sells at DEX spot P with slippage s. It is
profitable iff
    LIF * (P / O) * (1 - s) >= 1   <=>   s <= s* = 1 - O / (LIF * P)
so the maximum collateral (USD) the market can absorb in one step is
    q_max = R * max(0, LIF * P / O - 1).
Liquidatable demand beyond q_max is a backlog. Positions whose collateral is
exhausted (oracle LTV >= 1/LIF) realise bad debt, socialised to suppliers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEFAULT_LLTV, MORPHO_LIF_BETA, MORPHO_LIF_MAX


def morpho_lif(lltv: float, max_lif: float = MORPHO_LIF_MAX, beta: float = MORPHO_LIF_BETA) -> float:
    return min(max_lif, 1.0 / (beta * lltv + (1.0 - beta)))


def critical_slippage(lif: float, oracle: float = 1.0, spot: float = 1.0) -> float:
    """s* = 1 - O / (LIF * P): slippage above which liquidation is unprofitable."""
    return 1.0 - oracle / (lif * spot)


def cpmm_slippage(q_usd, depth_usd):
    q = np.asarray(q_usd, dtype=float)
    return q / (depth_usd + q)


def crash_shape(T: int = 120, k: float = 4.0) -> np.ndarray:
    """Front-loaded crash profile g(t) = (1 - e^{-k t/T}) / (1 - e^{-k}), t = 0..T, g(0)=0, g(T)=1."""
    t = np.arange(T + 1, dtype=float)
    return (1.0 - np.exp(-k * t / T)) / (1.0 - np.exp(-k))


# ---------------------------------------------------------------------------
# Chainlink-style push oracle
# ---------------------------------------------------------------------------
class PushOracle:
    """Posts a new round when |P - last_round| / last_round >= deviation or the heartbeat
    elapses; the round lands ``latency`` steps after the trigger."""

    def __init__(self, p0: float, deviation: float = 0.01, heartbeat: int = 60, latency: int = 1):
        self.deviation, self.heartbeat, self.latency = deviation, heartbeat, max(0, int(latency))
        self.value = p0
        self.last_round = p0
        self.last_t = 0
        self.pending: list[tuple[int, float]] = []

    def observe(self, t: int, price: float) -> tuple[float, bool]:
        triggered = False
        if abs(price - self.last_round) / self.last_round >= self.deviation or t - self.last_t >= self.heartbeat:
            self.last_round, self.last_t = price, t
            self.pending.append((t + self.latency, price))
            triggered = True
        while self.pending and self.pending[0][0] <= t:
            self.value = self.pending.pop(0)[1]
        return self.value, triggered


def simulate_oracle(spot: np.ndarray, deviation: float = 0.01, heartbeat: int = 60,
                    latency: int = 1) -> pd.DataFrame:
    spot = np.asarray(spot, dtype=float)
    orc = PushOracle(spot[0], deviation, heartbeat, latency)
    vals, trig = zip(*(orc.observe(t, p) for t, p in enumerate(spot)))
    o = np.array(vals)
    return pd.DataFrame({"spot": spot, "oracle": o, "round_triggered": trig,
                         "divergence": (o - spot) / spot})


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketParams:
    name: str
    kpk_supply_usd: float
    p0: float
    ownership: float = 0.40           # KPK share of market supply
    utilization0: float = 0.85
    lltv: float = DEFAULT_LLTV
    dex_depth_usd: float = 25e6       # R0: USD depth of collateral side of DEX pool
    liquidity_withdrawal: float = 2.0  # phi: R_t = R0 exp(-phi * drawdown)
    arb_recovery: float = 0.50        # per-step decay of liquidation price impact (CEX arbitrage)
    oracle_dex_weight: float = 0.30   # omega: share of DEX impact visible to the aggregated oracle
    ltv_min: float = 0.40
    beta_a: float = 2.0
    beta_b: float = 2.0
    n_buckets: int = 200


@dataclass(frozen=True)
class OracleParams:
    deviation: float = 0.01
    heartbeat: int = 60
    latency: int = 1


@dataclass(frozen=True)
class AgentParams:
    enabled: bool = False
    util_trigger: float = 0.90
    divergence_trigger: float = 0.015


def borrower_book(p: MarketParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic book: equal-debt buckets at Beta-quantile LTVs in [ltv_min, LLTV]."""
    u = (np.arange(p.n_buckets) + 0.5) / p.n_buckets
    ltv = p.ltv_min + (p.lltv - p.ltv_min) * stats.beta.ppf(u, p.beta_a, p.beta_b)
    supply = p.kpk_supply_usd / p.ownership
    debt_total = p.utilization0 * supply
    debt = np.full(p.n_buckets, debt_total / p.n_buckets)
    coll = debt / (ltv * p.p0)
    return ltv, debt, coll


# ---------------------------------------------------------------------------
# Single-market simulation
# ---------------------------------------------------------------------------
def simulate_market(p: MarketParams, spot_exo: np.ndarray, oracle_ref_exo: np.ndarray | None = None,
                    oracle: OracleParams = OracleParams(), agent: AgentParams = AgentParams(),
                    others_withdraw_rate: float = 0.05, settle_minutes: int = 240) -> dict:
    """Minute-by-minute cascade.

    After the shock window the exogenous price is held at its final level for
    ``settle_minutes`` (oracle keeps updating, impact decays, liquidators keep
    working the backlog). Bad debt is what remains once the market settles, so
    it is not an artefact of where the oracle round happens to sit at minute T.

    spot_exo       exogenous collateral USD price path (before liquidation impact)
    oracle_ref_exo if given, oracle tracks this exogenous series (e.g. an exchange-rate
                   oracle = ETH/USD x wstETH rate that ignores a market depeg);
                   otherwise it tracks the impacted DEX/market price.
    """
    spot_exo = np.asarray(spot_exo, dtype=float)
    T = spot_exo.size
    n_steps = T + max(0, int(settle_minutes))
    spot_run = np.concatenate([spot_exo, np.full(n_steps - T, spot_exo[-1])])
    ref_run = None if oracle_ref_exo is None else np.concatenate(
        [np.asarray(oracle_ref_exo, dtype=float), np.full(n_steps - T, oracle_ref_exo[-1])])
    lif = morpho_lif(p.lltv)
    _, D, C = borrower_book(p)
    open_ = np.ones_like(D, dtype=bool)
    S = p.kpk_supply_usd / p.ownership
    K = p.kpk_supply_usd
    orc = PushOracle(p.p0 if oracle_ref_exo is None else oracle_ref_exo[0],
                     oracle.deviation, oracle.heartbeat, oracle.latency)
    log_impact = 0.0
    agent_on = False
    agent_t = None
    kpk_withdrawn = 0.0
    realized_bad = 0.0
    kpk_realized_loss = 0.0

    rec = {k: np.zeros(n_steps) for k in (
        "spot_exo", "spot", "oracle", "divergence", "utilization", "dex_depth", "liq_demand_usd",
        "liq_executed_usd", "slippage_required", "slippage_star", "excess_borrow_usd",
        "realized_bad_debt_cum", "kpk_supply", "kpk_withdrawn_cum", "debt", "supply")}
    rec["oracle_update"] = np.zeros(n_steps, dtype=bool)
    rec["breach"] = np.zeros(n_steps, dtype=bool)
    rec["agent_active"] = np.zeros(n_steps, dtype=bool)

    for t in range(n_steps):
        P = spot_run[t] * np.exp(log_impact)
        dd = max(0.0, 1.0 - P / p.p0)
        R = p.dex_depth_usd * np.exp(-p.liquidity_withdrawal * dd)
        # Aggregated (CEX-weighted) oracle sees only a fraction omega of DEX liquidation impact.
        P_obs = spot_run[t] * np.exp(p.oracle_dex_weight * log_impact)
        O, upd = orc.observe(t, P_obs if ref_run is None else ref_run[t])
        d = (O - P) / P
        Dtot = float(D[open_].sum())
        U = Dtot / S if S > 0 else 1.0

        # --- KPK exit agent moves first ------------------------------------
        if agent.enabled and not agent_on and (U > agent.util_trigger or abs(d) > agent.divergence_trigger):
            agent_on, agent_t = True, t
        if agent_on and K > 0:
            w = min(K, max(0.0, S - Dtot))
            K -= w; S -= w; kpk_withdrawn += w
        # --- other lenders panic-withdraw ------------------------------------
        w_o = min(others_withdraw_rate * dd * max(S - K, 0.0), max(0.0, S - Dtot))
        S -= w_o

        # --- liquidations ----------------------------------------------------
        ltv_o = np.where(open_, D / np.maximum(C * O, 1e-300), 0.0)
        liq = open_ & (ltv_o > p.lltv)
        demand = executed = 0.0
        s_req = 0.0
        s_star = critical_slippage(lif, O, P)
        if liq.any():
            idx = np.where(liq)[0]
            idx = idx[np.argsort(-ltv_o[idx])]                 # most underwater first
            caps_collateral = D[idx] * lif / O >= C[idx]       # oracle LTV >= 1/LIF
            seize_full = np.minimum(C[idx], D[idx] * lif / O)  # collateral units
            usd = seize_full * P
            demand = float(usd.sum())
            s_req = float(cpmm_slippage(demand, R))
            q_max = R * max(0.0, lif * P / O - 1.0)
            executed = min(demand, q_max)
            if executed > 0:
                cum_prev = np.cumsum(usd) - usd
                frac = np.clip((executed - cum_prev) / usd, 0.0, 1.0)
                seized = frac * seize_full
                repaid = seized * O / lif
                C[idx] -= seized
                D[idx] -= repaid
                exhausted = (frac >= 1.0) & caps_collateral
                if exhausted.any():
                    ex_idx = idx[exhausted]
                    bad = float(D[ex_idx].sum())
                    D[ex_idx] = 0.0
                    C[ex_idx] = 0.0
                    open_[ex_idx] = False
                    if bad > 0 and S > 0:
                        share = K / S
                        kpk_realized_loss += bad * share
                        K -= bad * share
                        realized_bad += bad
                        S -= bad
                log_impact += 2.0 * np.log(R / (R + executed))
        # price impact decays as arbitrageurs restore the DEX to the external price
        log_impact *= (1.0 - p.arb_recovery)

        rec["spot_exo"][t] = spot_run[t]; rec["spot"][t] = P; rec["oracle"][t] = O
        rec["oracle_update"][t] = upd; rec["divergence"][t] = d; rec["utilization"][t] = U
        rec["dex_depth"][t] = R; rec["liq_demand_usd"][t] = demand; rec["liq_executed_usd"][t] = executed
        rec["slippage_required"][t] = s_req; rec["slippage_star"][t] = s_star
        rec["breach"][t] = demand > 0 and s_req > s_star
        rec["excess_borrow_usd"][t] = p.lltv * float(C[open_].sum()) * max(0.0, O - P)
        rec["realized_bad_debt_cum"][t] = realized_bad
        rec["kpk_supply"][t] = K; rec["kpk_withdrawn_cum"][t] = kpk_withdrawn
        rec["debt"][t] = float(D[open_].sum()); rec["supply"][t] = S
        rec["agent_active"][t] = agent_on

    # --- mark residual underwater positions to market at T ------------------
    P_T, R_T = rec["spot"][-1], rec["dex_depth"][-1]
    need = open_ & (D > p.lltv * C * P_T)
    q_open = float((C[need] * P_T).sum())
    s_T = float(cpmm_slippage(q_open, R_T)) if q_open > 0 else 0.0
    mtm_bad = float(np.maximum(0.0, D[open_] - C[open_] * P_T * (1.0 - s_T)).sum())
    kpk_mtm = mtm_bad * (K / S) if S > 0 else 0.0

    full = pd.DataFrame(rec)
    full.index.name = "minute"
    full["phase"] = np.where(full.index < T, "shock", "settlement")
    df = full.iloc[:T]
    total_bad = realized_bad + mtm_bad
    window = df["divergence"] > 0
    summary = {
        "market": p.name,
        "lif": lif,
        "s_star_at_par": critical_slippage(lif),
        "realized_bad_debt": realized_bad,
        "mtm_bad_debt": mtm_bad,
        "total_bad_debt": total_bad,
        "kpk_loss": kpk_realized_loss + kpk_mtm,
        "kpk_withdrawn": kpk_withdrawn,
        "agent_trigger_minute": agent_t,
        "max_slippage_required": float(full["slippage_required"].max()),
        "breach_minutes": int(full["breach"].sum()),
        "liquidated_usd": float(full["liq_executed_usd"].sum()),
        "residual_liquidatable_debt": float(D[need].sum()),
        "max_divergence": float(df["divergence"].max()),
        "arb_window_minutes": int(window.sum()),
        "arb_window_start": int(window.idxmax()) if window.any() else None,
        "max_excess_borrow_usd": float(df["excess_borrow_usd"].max()),
        "insolvent_borrow_threshold": 1.0 / p.lltv - 1.0,
        "instant_insolvent_flag": bool((df["divergence"] > 1.0 / p.lltv - 1.0).any()),
        "peak_utilization": float(df["utilization"].max()),
    }
    return {"timeseries": df, "timeseries_full": full, "summary": summary}


# ---------------------------------------------------------------------------
# Vault (wstETH/USDC + WBTC/USDC markets)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VaultScenario:
    shock: float = -0.25              # in [-0.40, -0.05]
    scenario: str = "eth_crash"       # "eth_crash" | "wsteth_depeg"
    oracle_type: str = "market"       # "market" | "exchange_rate" (matters for depeg)
    btc_beta: float = 0.70
    horizon_min: int = 120
    crash_k: float = 4.0


def simulate_vault(wsteth_mkt: MarketParams, wbtc_mkt: MarketParams, scen: VaultScenario,
                   oracle: OracleParams = OracleParams(), agent: AgentParams = AgentParams(),
                   others_withdraw_rate: float = 0.05, settle_minutes: int = 240) -> dict:
    g = crash_shape(scen.horizon_min, scen.crash_k)
    if scen.scenario == "wsteth_depeg":
        wst_path = wsteth_mkt.p0 * (1.0 + scen.shock * g)
        btc_path = np.full_like(g, wbtc_mkt.p0)
        wst_oracle_ref = np.full_like(g, wsteth_mkt.p0) if scen.oracle_type == "exchange_rate" else None
    else:
        wst_path = wsteth_mkt.p0 * (1.0 + scen.shock * g)
        btc_path = wbtc_mkt.p0 * (1.0 + scen.btc_beta * scen.shock * g)
        wst_oracle_ref = wst_path if scen.oracle_type == "exchange_rate" else None

    res = {}
    for mkt, path, ref in ((wsteth_mkt, wst_path, wst_oracle_ref), (wbtc_mkt, btc_path, None)):
        if mkt.kpk_supply_usd > 0:
            res[mkt.name] = simulate_market(mkt, path, ref, oracle, agent, others_withdraw_rate, settle_minutes)

    keys = ["realized_bad_debt", "mtm_bad_debt", "total_bad_debt", "kpk_loss", "kpk_withdrawn", "liquidated_usd"]
    agg = {k: float(sum(r["summary"][k] for r in res.values())) for k in keys}
    agg["max_slippage_required"] = max((r["summary"]["max_slippage_required"] for r in res.values()), default=0.0)
    agg["breach"] = any(r["summary"]["breach_minutes"] > 0 for r in res.values())
    return {"markets": res, "summary": agg}


def shock_sweep(wsteth_mkt: MarketParams, wbtc_mkt: MarketParams, base: VaultScenario,
                shocks: np.ndarray | None = None, oracle: OracleParams = OracleParams(),
                agent: AgentParams = AgentParams(), others_withdraw_rate: float = 0.05) -> dict:
    """Run the vault over a shock grid and locate the breaking thresholds."""
    if shocks is None:
        shocks = -np.round(np.arange(0.05, 0.4001, 0.005), 4)
    rows = []
    for sh in shocks:
        r = simulate_vault(wsteth_mkt, wbtc_mkt, replace(base, shock=float(sh)), oracle, agent,
                           others_withdraw_rate)["summary"]
        rows.append({"shock": float(sh), **r})
    df = pd.DataFrame(rows).sort_values("shock", ascending=False).reset_index(drop=True)
    breach = df[df["breach"]]
    bad = df[df["total_bad_debt"] > 1.0]
    return {
        "table": df,
        "slippage_breach_shock": float(breach["shock"].iloc[0]) if len(breach) else None,
        "bad_debt_shock": float(bad["shock"].iloc[0]) if len(bad) else None,
    }
