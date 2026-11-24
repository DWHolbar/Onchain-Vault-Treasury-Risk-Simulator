"""Shared sidebar: portfolio state + global data source."""
from __future__ import annotations

import streamlit as st

from src.config import PortfolioState
from src.data_fetcher import MarketData, get_market_data


@st.cache_data(ttl=6 * 3600, show_spinner="Loading market data…")
def load_market_data(source: str, seed: int) -> MarketData:
    return get_market_data(source, seed)


def render_sidebar() -> tuple[PortfolioState, MarketData]:
    sb = st.sidebar
    sb.header("Portfolio state")
    capital = sb.number_input("Total vault / treasury capital (USD)", min_value=1_000_000.0,
                              max_value=5_000_000_000.0, value=50_000_000.0, step=5_000_000.0, format="%.0f")
    w_m = sb.slider("Morpho curated lending vault (%)", 0, 100, 50, 5)
    w_u = sb.slider("Uniswap v3 ETH/USDC 0.05% LP (%)", 0, 100 - w_m, min(20, 100 - w_m), 5)
    w_r = 100 - w_m - w_u
    sb.caption(f"Liquid staked reserves: **{w_r}%** (remainder)")
    with sb.expander("Sub-allocations"):
        m_split = st.slider("Morpho supply in wstETH/USDC market (%)", 0, 100, 60, 5,
                            help="Remainder supplied to the WBTC/USDC market") / 100
        r_split = st.slider("Reserves held as wstETH (%)", 0, 100, 50, 5, help="Remainder held as sDAI") / 100

    sb.header("Data source")
    choice = sb.radio("Global market data", ["Offline Deterministic Stress Dataset",
                                            "Live Market Data (DefiLlama/CoinGecko with cache)"], index=0)
    source = "live" if choice.startswith("Live") else "offline"
    seed = int(sb.number_input("Synthetic dataset seed", 0, 10_000, 42, 1))
    if source == "live" and sb.button("Refresh live data"):
        load_market_data.clear()

    state = PortfolioState(total_capital=capital, w_morpho=w_m / 100, w_univ3=w_u / 100,
                           reserve_split_wsteth=r_split, morpho_split_wsteth=m_split,
                           data_source=source, seed=seed)
    md = load_market_data(source, seed)

    badge = {"live": ":green[● LIVE]", "cache": ":orange[● CACHE]", "offline": ":blue[● OFFLINE SYNTHETIC]"}
    sb.markdown(f"**Data in use:** {badge.get(md.source_used, md.source_used)}")
    sb.caption(md.note)

    sb.header("Allocation")
    for k, v in state.allocation().items():
        sb.markdown(f"<div style='display:flex;justify-content:space-between;font-size:0.9rem'>"
                    f"<span>{k}</span><span><b>${v / 1e6:,.1f}M</b></span></div>", unsafe_allow_html=True)
    return state, md
