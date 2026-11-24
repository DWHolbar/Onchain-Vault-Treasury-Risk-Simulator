# Onchain Vault & Treasury Risk Simulator (KPK Risk Sandbox)

An institutional DeFi risk and stress-testing sandbox, modelled on a KPK/Karpatkey-style onchain mandate:

- a curated **Morpho Blue lending vault** (USDC lent against wstETH and WBTC)
- a **Uniswap v3 ETH/USDC 0.05%** concentrated-liquidity position
- **liquid-staked reserves** (wstETH / sDAI)

```bash
pip install -r requirements.txt
streamlit run app.py
pytest
```

## Architecture

```
app.py                       Streamlit entry: shared sidebar state + 4 tabs
src/config.py                PortfolioState dataclass, Morpho / Uniswap constants
src/data_fetcher.py          CoinGecko + DefiLlama → JSON cache (6h) → deterministic synthetic fallback
src/models/portfolio_var.py  Tab 1: covariance, contagion stress, Historical / Parametric / Cornish–Fisher / MC VaR & ES, dependency map
src/models/lending_oracle.py Tab 2: push oracle, borrower book, CPMM slippage, liquidation cascade, exit agent, shock sweep
src/models/amm_backtester.py Tab 3: exact v3 reserve maths, IL, LVR, HODL / passive / auto-rebalanced backtests
src/ui/                      Plotly charts and per-tab rendering (no maths)
tests/test_models.py         Financial-invariant tests
```

All maths in `src/models/` is pure: there are no Streamlit imports, inputs are plain floats or DataFrames, and outputs are dicts or DataFrames. Randomness exists only in the synthetic data generator and in the Monte Carlo engine, both of which use seeds passed in as parameters.

## Data sources

The sidebar has a global data-source toggle:

- **Offline Deterministic Stress Dataset** (default): seeded fat-tailed (Student-t, ν=4) GBM with scripted crash episodes, including a −28% ETH crash and a stETH discount inside the 90-day backtest window.
- **Live Market Data**: CoinGecko prices and volumes plus DefiLlama yields and pool TVL/volume. The Uniswap pool is discovered by filter, not a hard-coded ID. Fetching falls back in order: live → cache (even if stale) → synthetic. The app never errors, and a badge shows which source is actually in use.

## Key formulas

| Area | Formula |
|---|---|
| Contagion stress | $\rho'_{ij}=(1-\lambda)\rho_{ij}+0.9\lambda$ on {ETH, wstETH, WBTC}; $\sigma'=2.5\sigma$; PSD repair by eigenvalue clipping |
| Historical stress | $r'=\mu+L'L^{-1}(r-\mu)$, where $L$ and $L'$ are the Cholesky factors of the sample and stressed covariance |
| Parametric VaR / ES | $-\delta^\top\mu+z_\alpha\sigma_p$, $\;-\delta^\top\mu+\sigma_p\varphi(z_\alpha)/(1-\alpha)$ |
| Morpho lender (Tab 1) | $b(r)=U\max(0,1-e^r(1-s)/\ell)$, which is a short put on collateral |
| Morpho LIF | $\min(1.15,\,1/(0.3\,\mathrm{LLTV}+0.7))=1.0438$ at LLTV 0.86 |
| Liquidation profitability | $\mathrm{LIF}\cdot(P/O)(1-s)\ge1\Rightarrow s^*=1-O/(\mathrm{LIF}\,P)$; per-step capacity $q_{max}=R(\mathrm{LIF}P/O-1)$ |
| DEX slippage | $s(q)=q/(R+q)$, impact $P(R/(R+q))^2$, panic depth $R_t=R_0e^{-\varphi\,DD_t}$ |
| Uniswap v3 | $x=L(1/\sqrt p-1/\sqrt{p_b})$, $y=L(\sqrt p-\sqrt{p_a})$, $L=V/(2\sqrt p-p/\sqrt{p_b}-\sqrt{p_a})$ |
| v3 IL | $\frac{2\sqrt r-1-r}{1+r-\sqrt{p_a/p_0}-r\sqrt{p_0/p_b}}$ |
| LVR | $\sigma^2L\sqrt p/4$ per unit time while in range |

### Note on the cascade model

With a deviation-threshold oracle, *when* the next oracle update happens relative to the crash can change the outcome more than the shock size itself. To handle this, each 120-minute crash is followed by a **settlement phase**: the price is held, the oracle keeps updating and liquidators clear the backlog. The bad debt reported is the amount that is unrecoverable once the market settles, not a snapshot at minute 120. With this in place, bad debt is non-decreasing in shock severity across the oracle, depth and recovery parameter grid that was tested.

## Tests

`pytest` covers:

- (a) VaR99 ≥ VaR95 for every method in both regimes
- (b) bad debt is non-decreasing from −5% to −40%, for both the ETH-crash and wstETH-depeg scenarios
- (c) v3 IL equals 0 at r = 1
- (d) a zero-fee LP strictly underperforms holding its own tokens for any r ≠ 1, both in and out of range

It also checks supporting invariants:

- the LIF value
- oracle deviation and heartbeat behaviour
- the stressed correlation matrix stays PSD
- the closed-form IL matches the direct calculation
- the v3 invariant $(x+L/\sqrt{p_b})(y+L\sqrt{p_a})=L^2$
- the exit agent lowers KPK's loss
- the offline dataset is reproducible
