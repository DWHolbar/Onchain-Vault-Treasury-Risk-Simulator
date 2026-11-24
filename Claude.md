# Project: Onchain Vault & Treasury Risk Simulator (KPK Risk Sandbox)

## Tech Stack & Structure
- Python 3.11+, Streamlit, Plotly, NumPy, pandas, SciPy, requests, pytest.
- Keep all quantitative math in `src/models/` (pure functions, zero Streamlit imports) and all UI/charts in `app.py` and `src/ui/`.
- Folder structure:
  - `app.py` (Main Streamlit entry point with shared sidebar portfolio state)
  - `src/data_fetcher.py` (DefiLlama / CoinGecko API fetchers with local JSON cache + deterministic synthetic fallback)
  - `src/models/portfolio_var.py` (Covariance matrix, regime-shift correlation stress, Historical & Parametric VaR/CVaR)
  - `src/models/lending_oracle.py` (Morpho v2 / Aave LTV liquidation cascade, DEX slippage curve, Chainlink oracle lag window)
  - `src/models/amm_backtester.py` (Uniswap v3 concentrated liquidity fee accrual, Impermanent Loss, threshold rebalancing)
  - `tests/test_models.py` (Unit tests validating financial invariants)

## Coding & Verification Rules
- Never leave placeholder `TODO` math or mock random numbers inside `src/models/`. Use exact closed-form DeFi equations (e.g., Uniswap v3 $x \cdot y = L^2$ concentrated liquidity formulas).
- Every model function must accept clean floats/DataFrames and return structured dicts or DataFrames.
- After implementing or modifying any file in `src/models/`, run `pytest` via Bash and fix any failures before finishing.
