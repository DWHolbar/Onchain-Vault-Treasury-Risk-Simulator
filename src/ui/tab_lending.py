"""Tab 2: Morpho lending vault, oracle lag & liquidation cascade."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PortfolioState
from src.data_fetcher import MarketData
from src.models import lending_oracle as lo
from src.ui import charts


@st.cache_data(show_spinner="Simulating liquidation cascade…")
def _run(m1: lo.MarketParams, m2: lo.MarketParams, scen: lo.VaultScenario, orc: lo.OracleParams,
         agent: lo.AgentParams, kappa: float) -> dict:
    return {
        "single": lo.simulate_vault(m1, m2, scen, orc, agent, kappa),
        "single_no_agent": lo.simulate_vault(m1, m2, scen, orc, lo.AgentParams(False), kappa),
        "sweep": lo.shock_sweep(m1, m2, scen, None, orc, lo.AgentParams(False), kappa),
        "sweep_agent": lo.shock_sweep(m1, m2, scen, None, orc, lo.AgentParams(True, agent.util_trigger,
                                                                              agent.divergence_trigger), kappa),
    }


def render(state: PortfolioState, md: MarketData) -> None:
    st.subheader("Morpho Lending Vault: Oracle Lag & Liquidation Cascade")
    st.caption("USDC lent through Morpho Blue against wstETH and WBTC (LLTV 86%). A 120-minute front-loaded crash, "
               "followed by a settlement phase that holds the price, keeps the oracle updating and lets liquidators "
               "work through the backlog. Bad debt is what is still unrecoverable once the market has settled.")
    if state.morpho_usd <= 0:
        st.info("Allocate capital to the Morpho vault in the sidebar to run this tab.")
        return

    c = st.columns(4)
    scenario = c[0].radio("Scenario", ["ETH crash", "wstETH depeg"], horizontal=True)
    shock = -c[1].slider("Shock size (%)", 5, 40, 25, 1) / 100
    oracle_type = c[2].radio("wstETH oracle type", ["Market price", "Exchange-rate"], horizontal=True,
                             help="An exchange-rate oracle (ETH/USD × wstETH redemption rate) does not see a market depeg")
    agent_on = c[3].toggle("KPK Automated Exit Agent", value=False,
                           help="Withdraws idle liquidity when utilization > trigger or |oracle divergence| > trigger")

    with st.expander("Oracle, liquidity & market parameters", expanded=False):
        o = st.columns(4)
        dev = o[0].number_input("Oracle deviation threshold (%)", 0.1, 5.0, 1.0, 0.1) / 100
        hb = int(o[1].number_input("Heartbeat (min)", 5, 1440, 60, 5))
        lat = int(o[2].number_input("Update latency (min)", 0, 10, 1, 1))
        omega = o[3].slider("Oracle weight on DEX impact ω", 0.0, 1.0, 0.30, 0.05)
        q = st.columns(4)
        depth_w = q[0].number_input("wstETH DEX depth R₀ ($M)", 5.0, 500.0, 25.0, 5.0) * 1e6
        depth_b = q[1].number_input("WBTC DEX depth R₀ ($M)", 5.0, 500.0, 40.0, 5.0) * 1e6
        phi = q[2].slider("Panic liquidity withdrawal φ", 0.0, 5.0, 2.0, 0.25, help="R_t = R₀·exp(−φ·drawdown)")
        arb = q[3].slider("Arbitrage recovery / min", 0.05, 1.0, 0.5, 0.05)
        r = st.columns(4)
        own = r[0].slider("KPK share of market supply", 0.05, 1.0, 0.40, 0.05)
        u0 = r[1].slider("Initial utilization", 0.50, 0.95, 0.85, 0.01)
        kappa = r[2].slider("Other lenders' panic withdrawal κ", 0.0, 0.3, 0.05, 0.01)
        btc_beta = r[3].slider("WBTC beta to ETH shock", 0.0, 1.5, 0.7, 0.05)
        a = st.columns(2)
        u_trig = a[0].slider("Agent utilization trigger", 0.80, 0.99, 0.90, 0.01)
        d_trig = a[1].slider("Agent divergence trigger (%)", 0.5, 5.0, 1.5, 0.1) / 100

    px = md.daily_prices.iloc[-1]
    alloc = state.allocation()
    m1 = lo.MarketParams("wstETH/USDC", alloc["Morpho wstETH/USDC"], float(px["wstETH"]), own, u0,
                         dex_depth_usd=depth_w, liquidity_withdrawal=phi, arb_recovery=arb, oracle_dex_weight=omega)
    m2 = lo.MarketParams("WBTC/USDC", alloc["Morpho WBTC/USDC"], float(px["WBTC"]), own, u0,
                         dex_depth_usd=depth_b, liquidity_withdrawal=phi, arb_recovery=arb, oracle_dex_weight=omega)
    scen = lo.VaultScenario(shock, "wsteth_depeg" if scenario == "wstETH depeg" else "eth_crash",
                            "exchange_rate" if oracle_type == "Exchange-rate" else "market", btc_beta)
    orc = lo.OracleParams(dev, hb, lat)
    agent = lo.AgentParams(agent_on, u_trig, d_trig)
    out = _run(m1, m2, scen, orc, agent, kappa)
    single, base, sw, swa = out["single"], out["single_no_agent"], out["sweep"], out["sweep_agent"]

    lif = lo.morpho_lif(0.86)
    k = st.columns(5)
    k[0].metric(f"LIF (+{lif - 1:.2%} bonus)", f"{lif:.4f}")
    k[1].metric("Critical slippage s* = 1 − 1/LIF", f"{lo.critical_slippage(lif):.2%}")
    k[2].metric("Slippage > s* from", f"{sw['slippage_breach_shock']:.1%}" if sw["slippage_breach_shock"] is not None else "not in −5…−40%")
    k[3].metric("Unrecoverable bad debt from", f"{sw['bad_debt_shock']:.1%}" if sw["bad_debt_shock"] is not None else "not in −5…−40%")
    loss_on, loss_off = single["summary"]["kpk_loss"], base["summary"]["kpk_loss"]
    k[4].metric("KPK loss at this shock" + (f" · agent saves ${(loss_off - loss_on) / 1e6:,.2f}M" if agent_on else ""),
                f"${loss_on / 1e6:,.2f}M")

    st.plotly_chart(charts.bad_debt_curve(sw["table"], swa["table"], sw["slippage_breach_shock"], sw["bad_debt_shock"],
                                          "Bad debt vs. crash severity (shock sweep −5% … −40%)"),
                    use_container_width=True)

    mkt_name = st.radio("Market detail", list(single["markets"].keys()), horizontal=True)
    mres = single["markets"][mkt_name]
    ts, s = mres["timeseries"], mres["summary"]

    st.plotly_chart(charts.spot_vs_oracle(ts, f"{mkt_name}: DEX spot vs. lagged oracle (shaded = oracle above "
                                              f"spot → over-borrow / arbitrage window)"), use_container_width=True)
    w = st.columns(4)
    w[0].metric("Arbitrage window" + (f" · from minute {s['arb_window_start']}" if s["arb_window_start"] is not None else ""),
                f"{s['arb_window_minutes']} min")
    w[1].metric("Max oracle divergence", f"{s['max_divergence']:.2%}")
    w[2].metric("Peak excess borrow capacity", f"${s['max_excess_borrow_usd'] / 1e6:,.2f}M",
                help="LLTV·Σ collateral·(oracle − spot)⁺: debt drawable against overstated collateral")
    w[3].metric("Instant-insolvent borrow", "YES" if s["instant_insolvent_flag"] else "no",
                help=f"Triggered when oracle divergence > 1/LLTV − 1 = {s['insolvent_borrow_threshold']:.1%}")
    if s["instant_insolvent_flag"]:
        st.error("Oracle divergence exceeds 1/LLTV − 1. A borrower can post collateral, borrow at the stale "
                 "oracle price and walk away with more USDC than the collateral is worth at spot.")

    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.liquidation_flow(ts, "Liquidation demand vs. executed per minute"), use_container_width=True)
    g2.plotly_chart(charts.single_line(ts.index, ts["utilization"], "Market utilization", "utilization", ".0%",
                                       charts.SERIES[0], u_trig, "agent trigger"), use_container_width=True)
    g3, g4 = st.columns(2)
    g3.plotly_chart(charts.single_line(ts.index, ts["slippage_required"],
                                       "Slippage required to clear liquidatable collateral", "s_req", ".1%",
                                       charts.SERIES[1], lo.critical_slippage(lif), "s* (LIF break-even)"),
                    use_container_width=True)
    g4.plotly_chart(charts.single_line(ts.index, ts["kpk_supply"], "KPK supply remaining in market", "KPK", "$,.3s",
                                       charts.SERIES[2]), use_container_width=True)

    st.markdown("#### Outcome by market")
    rows = []
    for name, r in single["markets"].items():
        sm = r["summary"]
        rows.append({"market": name, "realized bad debt": sm["realized_bad_debt"], "residual MTM bad debt": sm["mtm_bad_debt"],
                     "total bad debt": sm["total_bad_debt"], "KPK loss": sm["kpk_loss"], "KPK exited": sm["kpk_withdrawn"],
                     "liquidated": sm["liquidated_usd"], "max s_req": sm["max_slippage_required"],
                     "agent trigger (min)": sm["agent_trigger_minute"]})
    tbl = pd.DataFrame(rows).set_index("market")
    fmt = {c: "${:,.0f}" for c in tbl.columns if c not in ("max s_req", "agent trigger (min)")}
    fmt["max s_req"] = "{:.2%}"
    st.dataframe(tbl.style.format(fmt, na_rep="–"), use_container_width=True)

    st.session_state["tab2"] = {"breach": sw["slippage_breach_shock"], "bad": sw["bad_debt_shock"],
                                "kpk_loss": loss_on, "lif": lif}
