"""Tab 3: Uniswap v3 concentrated liquidity backtester."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import PortfolioState
from src.data_fetcher import MarketData
from src.models import amm_backtester as amm
from src.ui import charts


@st.cache_data(show_spinner="Backtesting LP strategies…")
def _bt(price: pd.Series, vol: pd.Series, capital: float, width: float, tvl: float, n: int, gas: float,
        slip: float, ref_w: float) -> dict:
    return amm.run_backtest(price, vol, capital, width, tvl, n, gas, slip, 0.0005, ref_w)


def render(state: PortfolioState, md: MarketData) -> None:
    st.subheader("Uniswap v3 Concentrated Liquidity Backtester")
    capital = state.univ3_usd
    if capital <= 0:
        st.info("Allocate capital to the Uniswap v3 LP in the sidebar to run this tab.")
        return
    st.caption(f"${capital / 1e6:,.1f}M deployed in ETH/USDC 0.05% over {len(md.hourly_eth)} hourly bars. "
               "Exact v3 reserve maths; fees accrue pro rata to in-range liquidity versus the pool's competing "
               "liquidity implied by TVL.")

    c = st.columns(5)
    width = c[0].slider("Range width ±%", 1, 50, 10, 1) / 100
    n = int(c[1].number_input("Rebalance after N bars out of range", 1, 168, 6, 1))
    gas = c[2].number_input("Gas per rebalance ($)", 0.0, 5000.0, 40.0, 10.0)
    slip = c[3].number_input("Swap slippage (bps)", 0.0, 100.0, 5.0, 1.0)
    ref_w = c[4].slider("Competing LP reference width ±%", 5, 100, 20, 5,
                        help="Pool TVL is assumed to be concentrated at this width (sets the competing liquidity)") / 100
    h = md.hourly_eth
    res = _bt(h["price"], h["volume_usd"], capital, width, md.pool_tvl_usd, n, gas, slip, ref_w)
    summ, ser = res["summary"], res["series"]

    k = st.columns(3)
    for i, name in enumerate(summ.index):
        r = summ.loc[name]
        k[i].metric(name, f"${r['net_pnl_usd'] / 1e6:,.2f}M", f"{r['net_return']:+.2%} net",
                    delta_color="normal")

    st.plotly_chart(charts.strategy_lines(ser, "net_pnl", "Cumulative net PnL (value + fees − initial capital)"),
                    use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(charts.fee_vs_il_bars(summ, "Fees vs. impermanent loss vs. theoretical LVR"), use_container_width=True)
    b.plotly_chart(charts.price_with_ranges(ser["Passive LP"], ser["Auto-Rebalanced LP"],
                                            "ETH/USDC with LP ranges and rebalances"), use_container_width=True)

    tbl = pd.DataFrame({
        "Net PnL": summ["net_pnl_usd"].map(lambda v: f"${v:,.0f}"),
        "Fee APR": summ["fee_apr"].map(lambda v: f"{v:.1%}"),
        "IL (vs. LP tokens held)": summ["il_pct"].map(lambda v: f"{v:.2%}"),
        "Theoretical LVR": summ["lvr_usd"].map(lambda v: f"${v:,.0f}"),
        "Time in range": summ["time_in_range"].map(lambda v: f"{v:.1%}"),
        "Rebalances": summ["rebalances"],
        "Rebalance costs": summ["costs_usd"].map(lambda v: f"${v:,.0f}"),
    })
    st.dataframe(tbl, use_container_width=True)
    st.caption("IL is measured against holding the LP's initial token amounts. It is zero at r = 1 and negative for "
               "any move. For the rebalanced strategy it also includes realised losses from re-centring and costs. "
               "LVR = σ²·L·√p/4 per bar while in range: the theoretical loss to arbitrageurs that fees must cover.")

    st.session_state["tab3"] = {"summary": summ}
