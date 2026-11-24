"""Financial-invariant tests for the KPK risk sandbox models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PortfolioState
from src.data_fetcher import get_market_data
from src.models import amm_backtester as amm
from src.models import lending_oracle as lo
from src.models import portfolio_var as pv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def market():
    return get_market_data("offline", seed=42)


@pytest.fixture(scope="module")
def exposures(market):
    st = PortfolioState()
    a = st.allocation()
    return pv.Exposures(
        morpho_wsteth_usd=a["Morpho wstETH/USDC"], morpho_wbtc_usd=a["Morpho WBTC/USDC"],
        lp_capital_usd=a["Uniswap v3 ETH/USDC"], wsteth_reserve_usd=a["wstETH reserve"],
        sdai_reserve_usd=a["sDAI reserve"], eth_price=float(market.daily_prices["ETH"].iloc[-1]),
        lp_daily_fee_usd=2_000.0,
    )


@pytest.fixture(scope="module")
def vault_markets(market):
    px = market.daily_prices.iloc[-1]
    st = PortfolioState()
    a = st.allocation()
    m1 = lo.MarketParams("wstETH/USDC", a["Morpho wstETH/USDC"], float(px["wstETH"]), dex_depth_usd=25e6)
    m2 = lo.MarketParams("WBTC/USDC", a["Morpho WBTC/USDC"], float(px["WBTC"]), dex_depth_usd=40e6)
    return m1, m2


# ---------------------------------------------------------------------------
# (a) VaR 99% >= VaR 95%
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("regime", ["normal", "contagion"])
def test_var99_ge_var95(market, exposures, regime):
    res = pv.compute_portfolio_risk(market.daily_prices, exposures, regime=regime, n_mc=5_000)
    t = res["table"]
    for method, row in t.iterrows():
        assert row["VaR 99%"] >= row["VaR 95%"] - 1e-9, method
        assert row["ES 99%"] >= row["ES 95%"] - 1e-9, method
        assert row["ES 95%"] >= row["VaR 95%"] - 1e-9, method
        assert row["ES 99%"] >= row["VaR 99%"] - 1e-9, method


def test_contagion_increases_risk(market, exposures):
    n = pv.compute_portfolio_risk(market.daily_prices, exposures, "normal", n_mc=5_000)["table"]
    c = pv.compute_portfolio_risk(market.daily_prices, exposures, "contagion", n_mc=5_000)["table"]
    assert (c["VaR 99%"] > n["VaR 99%"]).all()


# ---------------------------------------------------------------------------
# (b) Bad debt is non-decreasing in shock severity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["eth_crash", "wsteth_depeg"])
def test_bad_debt_monotone_in_shock(vault_markets, scenario):
    m1, m2 = vault_markets
    sw = lo.shock_sweep(m1, m2, lo.VaultScenario(scenario=scenario))
    df = sw["table"].sort_values("shock", ascending=False)  # -5% -> -40%
    bad = df["total_bad_debt"].to_numpy()
    assert np.all(np.diff(bad) >= -1e-6), bad
    assert bad[-1] >= bad[0]


def test_bad_debt_appears_beyond_threshold(vault_markets):
    m1, m2 = vault_markets
    sw = lo.shock_sweep(m1, m2, lo.VaultScenario())
    assert sw["slippage_breach_shock"] is not None
    assert sw["bad_debt_shock"] is not None
    # slippage breaches s* no later (i.e. at a milder shock) than bad debt appears
    assert sw["slippage_breach_shock"] >= sw["bad_debt_shock"]


def test_exit_agent_reduces_kpk_loss(vault_markets):
    m1, m2 = vault_markets
    scen = lo.VaultScenario(shock=-0.40)
    off = lo.simulate_vault(m1, m2, scen)["summary"]["kpk_loss"]
    on = lo.simulate_vault(m1, m2, scen, agent=lo.AgentParams(enabled=True))["summary"]["kpk_loss"]
    assert on <= off


# ---------------------------------------------------------------------------
# (c) Uniswap v3 IL at r = 1 is zero
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [0.02, 0.05, 0.10, 0.25, 0.50])
def test_il_zero_at_r1(width):
    p0 = 3000.0
    pa, pb = amm.range_bounds(p0, width)
    assert abs(float(amm.impermanent_loss(p0, p0, pa, pb))) < 1e-12
    assert abs(float(amm.il_closed_form_in_range(1.0, p0, pa, pb))) < 1e-12


# ---------------------------------------------------------------------------
# (d) zero-fee LP strictly underperforms HODL whenever price moves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [0.05, 0.10, 0.30])
@pytest.mark.parametrize("r", [0.5, 0.8, 0.95, 0.99, 1.01, 1.05, 1.2, 2.0])
def test_zero_fee_lp_underperforms_hodl(width, r):
    p0 = 3000.0
    pa, pb = amm.range_bounds(p0, width)
    L = amm.liquidity_for_value(1_000_000.0, p0, pa, pb)
    x0, y0 = amm.token_amounts(L, p0, pa, pb)
    p = r * p0
    v_lp = float(amm.position_value(L, p, pa, pb))
    v_hold = float(x0 * p + y0)
    assert v_lp < v_hold
    assert float(amm.impermanent_loss(p, p0, pa, pb)) < 0


def test_zero_fee_backtest_underperforms_hodl_of_lp_tokens(market):
    h = market.hourly_eth
    res = amm.run_backtest(h["price"], h["volume_usd"] * 0.0, 10e6, width=0.10)
    assert res["summary"].loc["Passive LP", "il_usd"] < 0
    assert res["summary"].loc["Passive LP", "fees_usd"] == 0


# ---------------------------------------------------------------------------
# Supporting invariants
# ---------------------------------------------------------------------------
def test_morpho_lif_value():
    assert lo.morpho_lif(0.86) == pytest.approx(1.0 / 0.958, rel=1e-12)
    assert lo.critical_slippage(lo.morpho_lif(0.86)) == pytest.approx(0.042, abs=1e-9)
    assert lo.morpho_lif(0.0) == pytest.approx(1.15)  # capped at M


def test_oracle_respects_deviation_and_heartbeat():
    spot = 3000.0 * (1 - 0.004 * np.minimum(np.arange(200), 2))  # -0.8% move then flat
    df = lo.simulate_oracle(spot, deviation=0.01, heartbeat=60, latency=0)
    assert (df["oracle"].iloc[:60] == 3000.0).all()           # below threshold, before heartbeat
    assert df["oracle"].iloc[60] == pytest.approx(spot[60])    # heartbeat refresh
    fast = 3000.0 * (1 - 0.02 * (np.arange(10) >= 3))
    df2 = lo.simulate_oracle(fast, deviation=0.01, heartbeat=60, latency=1)
    assert df2["oracle"].iloc[3] == 3000.0 and df2["oracle"].iloc[4] == pytest.approx(fast[3])


def test_stressed_correlation_psd(market):
    rets = pv.log_returns(market.daily_prices)
    c = pv.stress_correlation(rets.corr(), 0.90, 1.0).to_numpy()
    assert np.allclose(np.diag(c), 1.0)
    assert np.linalg.eigvalsh(c).min() > -1e-10
    assert c[0, 1] == pytest.approx(0.90, abs=0.05)


def test_v3_closed_form_il_matches_direct():
    p0 = 2500.0
    pa, pb = amm.range_bounds(p0, 0.2)
    r = np.linspace(0.81, 1.19, 25)
    np.testing.assert_allclose(amm.il_closed_form_in_range(r, p0, pa, pb),
                               amm.impermanent_loss(r * p0, p0, pa, pb), atol=1e-12)


def test_v3_invariant_in_range():
    p0, L = 3000.0, 1e5
    pa, pb = amm.range_bounds(p0, 0.1)
    for p in (2750.0, 3000.0, 3250.0):
        x, y = amm.token_amounts(L, p, pa, pb)
        assert (x + L / np.sqrt(pb)) * (y + L * np.sqrt(pa)) == pytest.approx(L**2, rel=1e-12)


def test_offline_data_deterministic():
    a = get_market_data("offline", seed=7)
    b = get_market_data("offline", seed=7)
    pd.testing.assert_frame_equal(a.daily_prices, b.daily_prices)
    pd.testing.assert_frame_equal(a.hourly_eth, b.hourly_eth)
    assert a.source_used == "offline"
