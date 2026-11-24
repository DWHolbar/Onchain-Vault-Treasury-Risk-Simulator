"""Tab 1: Portfolio concentration & correlation stress-test."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PortfolioState
from src.data_fetcher import MarketData
from src.models import amm_backtester as amm
from src.models import portfolio_var as pv
from src.ui import charts


@st.cache_data(show_spinner="Running VaR engines…")
def _risk(prices: pd.DataFrame, ex: pv.Exposures, regime: str, vol_mult: float, target: float, lam: float,
          horizon: int, seed: int) -> dict:
    return pv.compute_portfolio_risk(prices, ex, regime, vol_mult, target, lam, horizon, 20_000, seed)


def lp_daily_fee_estimate(md: MarketData, capital: float, width: float) -> float:
    h = md.hourly_eth
    p = float(h["price"].iloc[-1])
    pa, pb = amm.range_bounds(p, width)
    L = amm.liquidity_for_value(capital, p, pa, pb) if capital > 0 else 0.0
    Lp = float(amm.pool_liquidity_from_tvl(md.pool_tvl_usd, p))
    daily_vol = float(h["volume_usd"].iloc[-24 * 30:].sum() / 30)
    return 0.0005 * daily_vol * L / (L + Lp) if L > 0 else 0.0


def build_exposures(state: PortfolioState, md: MarketData, lp_width: float, utilization: float,
                    ltv: float, slip: float) -> pv.Exposures:
    a = state.allocation()
    return pv.Exposures(
        morpho_wsteth_usd=a["Morpho wstETH/USDC"], morpho_wbtc_usd=a["Morpho WBTC/USDC"],
        lp_capital_usd=a["Uniswap v3 ETH/USDC"], wsteth_reserve_usd=a["wstETH reserve"],
        sdai_reserve_usd=a["sDAI reserve"], eth_price=float(md.daily_prices["ETH"].iloc[-1]),
        lp_width=lp_width, lp_daily_fee_usd=lp_daily_fee_estimate(md, a["Uniswap v3 ETH/USDC"], lp_width),
        morpho_apy=float(md.yields.get("morpho_usdc_supply_apy", 0.06)),
        utilization=utilization, borrower_ltv=ltv, liq_slippage=slip,
    )


def render(state: PortfolioState, md: MarketData) -> None:
    st.subheader("Portfolio Concentration & Correlation Stress-Test")
    st.caption("Daily log-returns of ETH, wstETH, WBTC and sDAI (USDC numeraire). Every position is fully revalued: "
               "the Uniswap v3 leg through the exact concentrated-liquidity value function, and the Morpho lender "
               "as a short put on collateral gap risk.")

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    regime_label = c1.radio("Market regime", ["Normal Market", "Contagion Shock"], horizontal=True)
    regime = "contagion" if regime_label.startswith("Contagion") else "normal"
    vol_mult = c2.number_input("Contagion vol multiplier", 1.0, 5.0, 2.5, 0.25, disabled=regime == "normal")
    target = c3.number_input("Stressed correlation target", 0.0, 0.99, 0.90, 0.05, disabled=regime == "normal")
    lam = c4.slider("Stress intensity λ", 0.0, 1.0, 1.0, 0.05, disabled=regime == "normal",
                    help="ρ' = (1−λ)ρ + λ·target within the crypto block")

    with st.expander("Position assumptions"):
        e1, e2, e3, e4, e5 = st.columns(5)
        lp_width = e1.slider("LP range ±%", 2, 50, 10, 1) / 100
        util = e2.slider("Morpho utilization", 0.5, 1.0, 0.90, 0.01)
        ltv = e3.slider("Borrower LTV (gap-risk)", 0.50, 0.86, 0.86, 0.01, help="0.86 = all borrowers at LLTV (worst case)")
        slip = e4.slider("Liquidation slippage", 0.0, 0.10, 0.02, 0.005)
        horizon = e5.selectbox("Horizon (days, √h scaling = Gaussian assumption)", [1, 5, 10], 0)

    ex = build_exposures(state, md, lp_width, util, ltv, slip)
    res = _risk(md.daily_prices, ex, regime, vol_mult, target, lam, horizon, state.seed)
    t = res["table"]
    aum = ex.total

    hist = t.loc["Historical"]
    m = st.columns(4)
    m[0].metric(f"Historical VaR 95% · {hist['VaR 95%'] / aum:.2%} of AUM", f"${hist['VaR 95%'] / 1e6:,.2f}M")
    m[1].metric(f"Historical VaR 99% · {hist['VaR 99%'] / aum:.2%} of AUM", f"${hist['VaR 99%'] / 1e6:,.2f}M")
    m[2].metric(f"Historical ES 99% · {hist['ES 99%'] / aum:.2%} of AUM", f"${hist['ES 99%'] / 1e6:,.2f}M")
    m[3].metric(f"P&L excess kurtosis · skew {res['loss_skew']:.2f}", f"{res['loss_exkurt']:.1f}",
                help="0 for a Gaussian")

    st.plotly_chart(charts.var_es_bars(t, ["VaR 95%", "ES 95%", "VaR 99%", "ES 99%"],
                                       f"VaR & Expected Shortfall by method ({regime_label}, {horizon}d)"),
                    use_container_width=True)
    show = t.copy()
    for c in show.columns:
        show[c] = show[c].map(lambda v: f"${v / 1e6:,.2f}M ({v / aum:.2%})")
    st.dataframe(show, use_container_width=True)
    st.caption(f"{res['n_obs']} daily observations. The gap between delta-normal and full-revaluation/historical "
               "results is the non-linear (short-gamma LP, short-put lender) and fat-tail risk that a Gaussian "
               "VaR misses.")

    h1, h2 = st.columns(2)
    h1.plotly_chart(charts.correlation_heatmap(res["corr_normal"], "Sample correlation (normal)"), use_container_width=True)
    h2.plotly_chart(charts.correlation_heatmap(res["corr_used"], f"Correlation used: {regime_label}"),
                    use_container_width=True)
    vol = pd.DataFrame({"Annualised vol (sample)": res["vol_normal_ann"], "Annualised vol (used)": res["vol_used_ann"]})
    st.dataframe(vol.map(lambda v: f"{v:.1%}").T, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.loss_histogram(res["loss_mc"], t.loc["Monte Carlo (full reval)", "VaR 99%"],
                                              t.loc["Monte Carlo (full reval)", "ES 99%"],
                                              "Monte Carlo full-revaluation loss distribution"),
                        use_container_width=True)
    with c2:
        comp = res["component_es"]["component_ES99"]
        st.plotly_chart(charts.horizontal_bars(comp, "Contribution to 99% ES (Monte Carlo) by position"),
                        use_container_width=True)

    st.markdown("#### Hidden counterparty & protocol dependencies")
    st.caption("Each dollar is mapped to every smart contract, oracle and issuer it relies on. The rows overlap, so "
               "gross dependency exposure adds up to more than AUM. A single failure point can hit several "
               "positions that look diversified.")
    a = state.allocation()
    p = float(md.daily_prices["ETH"].iloc[-1])
    usdc_share = 1 - amm.eth_value_share(p, *amm.range_bounds(p, lp_width))
    with st.expander("Loss-given-failure assumptions"):
        cols = st.columns(len(pv.DEFAULT_LGD))
        lgd = {k: cols[i].number_input(k, 0.0, 1.0, v, 0.05, key=f"lgd_{k}") for i, (k, v) in enumerate(pv.DEFAULT_LGD.items())}
    dep = pv.dependency_exposure(a, usdc_share, lgd)
    d1, d2 = st.columns([1.2, 1])
    with d1:
        st.plotly_chart(charts.horizontal_bars(dep["summary"]["gross_exposure_usd"],
                                               "Gross exposure by dependency"), use_container_width=True)
    with d2:
        k = st.columns(2)
        k[0].metric("Dependency overlap", f"{dep['overlap_ratio']:.2f}× AUM")
        k[1].metric("Dependency HHI", f"{dep['hhi']:.3f}", help="Σ share², 1/7≈0.143 = perfectly spread")
        s = dep["summary"].copy()
        st.dataframe(pd.DataFrame({
            "Exposure": s["gross_exposure_usd"].map(lambda v: f"${v / 1e6:,.1f}M"),
            "% AUM": s["pct_of_aum"].map(lambda v: f"{v:.0%}"),
            "LGD": s["lgd"].map(lambda v: f"{v:.0%}"),
            "Loss if fails": s["loss_if_failure_usd"].map(lambda v: f"${v / 1e6:,.1f}M"),
        }), use_container_width=True)
    with st.expander("Position × dependency matrix (USD)"):
        st.dataframe(dep["matrix_usd"].map(lambda v: f"${v / 1e6:,.1f}M" if v else "–"), use_container_width=True)

    st.session_state["tab1"] = {"regime": regime_label, "hist_var99": hist["VaR 99%"],
                                "param_var99": t.loc["Parametric (delta-normal)", "VaR 99%"],
                                "kurt": res["loss_exkurt"], "aum": aum}
