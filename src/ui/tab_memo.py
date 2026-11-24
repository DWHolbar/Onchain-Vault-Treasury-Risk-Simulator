"""Tab 4: Quantitative risk memo — where this model breaks."""
from __future__ import annotations

import streamlit as st

from src.config import PortfolioState


def _live_numbers(state: PortfolioState) -> str:
    ss = st.session_state
    parts = []
    if "tab1" in ss:
        t = ss["tab1"]
        ratio = t["hist_var99"] / t["param_var99"] if t["param_var99"] else float("nan")
        parts.append(f"- **Tab 1 ({t['regime']}):** historical VaR99 is **${t['hist_var99'] / 1e6:,.2f}M** against "
                     f"delta-normal **${t['param_var99'] / 1e6:,.2f}M** ({ratio:.2f}×). The P&L excess kurtosis is "
                     f"**{t['kurt']:.1f}**; a Gaussian has 0.")
    if "tab2" in ss:
        t = ss["tab2"]
        b = f"{t['breach']:.1%}" if t["breach"] is not None else "outside −5…−40%"
        d = f"{t['bad']:.1%}" if t["bad"] is not None else "outside −5…−40%"
        parts.append(f"- **Tab 2:** liquidation slippage breaks the LIF break-even (s* = {1 - 1 / t['lif']:.2%}) at a "
                     f"**{b}** shock, and unrecoverable bad debt starts at **{d}**.")
    if "tab3" in ss:
        s = ss["tab3"]["summary"]
        if "Passive LP" in s.index:
            r = s.loc["Passive LP"]
            parts.append(f"- **Tab 3:** the passive LP earned **${r['fees_usd'] / 1e6:,.2f}M** in fees against a "
                         f"theoretical LVR of **${r['lvr_usd'] / 1e6:,.2f}M**, and was in range "
                         f"**{r['time_in_range']:.0%}** of the time.")
    if not parts:
        return "_Open Tabs 1–3 to populate the live figures quoted here._"
    return "\n".join(parts)


def render(state: PortfolioState) -> None:
    st.subheader("Quantitative Risk Memo: Where This Model Breaks")
    st.markdown(f"""
**To:** Investment & Risk Committee  
**From:** Quant Risk  
**Re:** Model limitations of the KPK Risk Sandbox  
**Mandate:** ${state.total_capital / 1e6:,.0f}M ({state.w_morpho:.0%} Morpho / {state.w_univ3:.0%} Uniswap v3 /
{state.w_reserves:.0%} LST & sDAI reserves)

---
### 1. Executive summary
The sandbox gives *conditional* answers: it tells you what happens if the stated assumptions hold. Each assumption
below fails in a predictable direction during an onchain panic, and nearly every failure **understates** risk.
Use the outputs as a lower bound on tail loss and as a map of which mechanism dominates. Do not read them as
point forecasts.

**Live figures from this session**

{_live_numbers(state)}
""")

    st.markdown(r"""
---
### 2. Tab 1: VaR / ES

| Assumption | Formula used | How live markets break it |
|---|---|---|
| Gaussian factor returns (parametric, MC) | $\mathrm{VaR}_\alpha=-\delta^\top\mu+z_\alpha\sqrt{\delta^\top\Sigma\delta}$, $\;\mathrm{ES}_\alpha=-\delta^\top\mu+\sigma_p\,\varphi(z_\alpha)/(1-\alpha)$ | Crypto daily returns have excess kurtosis well above 3. Event days such as the stETH discount, USDC's SVB depeg and FTX are 6–10σ moves that a normal model puts at essentially zero probability. |
| Cornish–Fisher fixes the tails | $z_{cf}=z+\tfrac{(z^2-1)S}{6}+\tfrac{(z^3-3z)K}{24}-\tfrac{(2z^3-5z)S^2}{36}$ | The expansion is only valid for moderate $S, K$. At the kurtosis this portfolio shows under contagion, the quantile becomes non-monotone (we force it monotone) and **overshoots**. It is a warning signal, not a better estimate. Use EVT/GPD on the tail instead. |
| Stationary covariance | $\hat\Sigma$ from 365 daily observations | Correlations are regime-dependent and spike toward 1 in a deleveraging. The contagion regime $\rho'=(1-\lambda)\rho+0.9\lambda$, $\sigma'=2.5\sigma$ is a *scenario*, not an estimate. Real contagion is also **asymmetric**: correlations rise in sell-offs only. |
| $\sqrt{h}$ scaling | $\mathrm{VaR}_h=\sqrt{h}\,\mathrm{VaR}_1$ | Volatility clusters (GARCH effects), and liquidation feedback makes returns autocorrelated in a crash. Both make multi-day loss larger than $\sqrt h$ scaling predicts. |
| Static gap-risk lender model | $b_k=U\cdot\max\!\big(0,1-e^{r}(1-s)/\ell\big)$ | Assumes no intraday liquidation and a fixed slippage $s$. Tab 2 shows that $s$ is endogenous: it rises exactly when $r$ is most negative. |
| Dependencies are additive exposures | gross $=\sum_i E_{ij}\,N_i$, fixed LGD | Failures are **correlated**. A Lido incident depegs wstETH, which triggers Morpho liquidations, which push Chainlink-reported prices down and drain Uniswap liquidity. The real dependency graph has feedback edges that a matrix does not capture. |

**Fat tails versus Gaussian VaR.** The ratio of historical to delta-normal VaR99 measures model error directly. When it is well
above 1, the risk budget should be set on historical or stressed ES, not on parametric VaR.
""")

    st.markdown(r"""
---
### 3. Tab 2: Morpho oracle lag and liquidation cascade

| Assumption | Formula used | How live markets break it |
|---|---|---|
| Liquidators are rational, instant and capital-unconstrained | act iff $\mathrm{LIF}\cdot\frac{P}{O}(1-s)\ge1$, capacity $q_{max}=R\,(\mathrm{LIF}\,P/O-1)$ | In a panic, gas spikes, builders congest and liquidators hit their USDC inventory limits. Several bots compete for the same depth. Effective capacity is lower than $q_{max}$ and is used later. |
| Constant-product depth that decays smoothly | $s(q)=q/(R+q)$, $\;R_t=R_0e^{-\varphi\,DD_t}$ | **Dynamic liquidity withdrawal** is discontinuous. Concentrated LPs pull ranges within blocks, so depth just below spot can vanish (the liquidity cliff). Real v3 depth is a tick distribution, not one reserve. |
| Exogenous, deterministic price path | $P_t=P_0(1+\text{shock}\cdot g(t))$ | In reality the path is endogenous to liquidation selling across *all* lending markets (Aave, Spark and other Morpho markets sell the same collateral). The sandbox only internalises our own two markets. |
| Push oracle with fixed deviation, heartbeat and latency | update iff $\lvert P-O\rvert/O\ge\delta$ or $t-t_0\ge H$ | Oracle rounds can be **delayed by MEV/ordering**, or front-run by searchers who back-run the update with liquidations. Aggregators can also lag CEX prices when venues halt. |
| Exchange-rate oracles are safe | $O=$ ETH/USD $\times$ redemption rate | They are safe against manipulation, but they **hide a market depeg**. No liquidations fire, while what the collateral can actually be sold for collapses. Run the depeg scenario with the exchange-rate oracle to see bad debt build without a single liquidation. |
| Bad debt is socialised pro rata at realisation | KPK loss $=b\cdot K/S$ | MetaMorpho vaults realise losses at different times. Withdrawal queues, timelocks and other lenders front-running exits (bank-run dynamics) change who bears the loss. The exit agent's edge assumes it acts **first** in the block. |

**Path dependence.** With a deviation-threshold oracle, where the next oracle update happens to fall relative to the
crash can change the outcome by more than the shock size does. We use a settlement phase so that the bad debt
reported is the unrecoverable amount, not a snapshot at minute 120. The underlying sensitivity is real and should be
stress-tested across oracle parameters.
""")

    st.markdown(r"""
---
### 4. Tab 3: Uniswap v3 backtest

| Assumption | Formula used | How live markets break it |
|---|---|---|
| Pro-rata fee share | $\text{fee}_t=\gamma\,V_t\,\frac{L}{L+L_{pool}}\,\mathbb 1_{\text{in range}}$ | **JIT liquidity sniping.** Searchers add very concentrated liquidity in the same block as a large swap and remove it right after. They capture most of the fee on the most valuable flow, so passive LPs earn materially less than pro rata on large trades. |
| Static competing liquidity | $L_{pool}$ implied from TVL at ±20% | Competing liquidity re-concentrates around spot after moves and leaves during volatility. Our fee share is lowest when volume is highest. |
| Volume is benign | historical volume × fee tier | **Toxic flow / LVR.** Informed arbitrageurs trade against stale pool prices. Expected LVR is $\tfrac{\sigma^2}{8}$ of pool value per unit time for a full-range v2 pool and $\tfrac{\sigma^2 L\sqrt p}{4}$ for a v3 position in range. Fees from uninformed flow have to exceed this for LPing to beat a rebalanced HODL. |
| Close-to-close range checks | in range iff $p_a\le p_t\le p_b$ at bar close | Price can leave and re-enter within a bar, so fees and adverse selection inside the bar are both missed. Hourly bars understate realised LVR. |
| Rebalance costs are fixed | gas $+$ notional$\cdot(\gamma+\text{slip})$ | Rebalancing in a trend crystallises IL (it sells low and buys high). Gas and slippage are highest during the volatility that triggers rebalances, and rebalance swaps are themselves sandwichable. |

**IL definition.** $\mathrm{IL}(r)=\frac{2\sqrt r-1-r}{1+r-\sqrt{p_a/p_0}-r\sqrt{p_0/p_b}}$ in range. It is 0 at
$r=1$ and strictly negative otherwise. Without fees, an LP therefore always underperforms holding its own initial
tokens, and the unit tests verify this.
""")

    st.markdown(r"""
---
### 5. Model-risk controls & recommended extensions
1. **Tails:** fit a GPD to losses beyond the 90th percentile (peaks-over-threshold) and report EVT-ES next to historical ES. Back-test VaR exceptions (Kupiec / Christoffersen).
2. **Liquidity:** replace the CPMM curve with tick-level v3 depth snapshots taken during past stress windows (Mar-2020, May-2022, Mar-2023, Aug-2024), and model the liquidity cliff explicitly.
3. **Liquidators:** use an agent-based model with liquidator inventory limits, a gas auction and competition across Aave, Spark and Morpho for the same DEX depth.
4. **Oracles:** run a Monte Carlo over round timing and latency, and report the *distribution* of bad debt across oracle parameters rather than a single path.
5. **LP:** apply a JIT haircut on fee share for swaps above a size threshold, and calibrate a markout-based toxicity measure from onchain swap data.
6. **Governance:** gate the exit agent's thresholds against MetaMorpho withdrawal queue depth and timelocks. Test it against a simultaneous run by other lenders, since being "first in the block" is not guaranteed.
""")
