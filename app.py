"""KPK Risk & Stress-Testing Sandbox — Streamlit entry point.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.ui import tab_amm, tab_lending, tab_memo, tab_var
from src.ui.sidebar import render_sidebar

st.set_page_config(page_title="KPK Risk Sandbox", page_icon="🛡️", layout="wide")

st.title("Onchain Vault & Treasury Risk Simulator")
st.caption("Institutional DeFi risk and stress-testing sandbox: curated Morpho lending, Uniswap v3 liquidity and "
           "liquid-staked treasury reserves.")

state, md = render_sidebar()

t1, t2, t3, t4 = st.tabs([
    "1 · Portfolio VaR & Correlation Stress",
    "2 · Morpho Oracle Lag & Liquidation Cascade",
    "3 · Uniswap v3 LP Backtester",
    "4 · Risk Memo: Where This Model Breaks",
])
with t1:
    tab_var.render(state, md)
with t2:
    tab_lending.render(state, md)
with t3:
    tab_amm.render(state, md)
with t4:
    tab_memo.render(state)
