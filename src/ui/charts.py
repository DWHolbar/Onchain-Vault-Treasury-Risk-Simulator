"""Plotly figure builders (validated default data-viz palette, light theme)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#f0efec"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, NEUTRAL], [0.75, "#ec8a8a"], [1.0, "#b3261e"]]


def _layout(fig: go.Figure, title: str = "", height: int = 380, yfmt: str | None = None,
            xtitle: str = "", ytitle: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT), x=0, xanchor="left"),
        height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=70),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT_2, size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(color=TEXT_2)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title=xtitle, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle, tickformat=yfmt)
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    z = corr.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, colorscale=DIVERGING,
        text=np.round(z, 2), texttemplate="%{text:.2f}", textfont=dict(size=13),
        xgap=2, ygap=2, colorbar=dict(thickness=10, len=0.8, tickvals=[-1, 0, 1]),
        hovertemplate="%{y} × %{x}: ρ = %{z:.3f}<extra></extra>",
    ))
    fig = _layout(fig, title, height=340)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def var_es_bars(table: pd.DataFrame, cols: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_bar(name=c, x=table.index, y=table[c], marker_color=SERIES[i],
                    marker_line=dict(color=SURFACE, width=2),
                    hovertemplate="%{x}<br>" + c + ": $%{y:,.0f}<extra></extra>")
    fig = _layout(fig, title, yfmt="$,.2s")
    fig.update_layout(barmode="group", hovermode="closest", bargap=0.25)
    return fig


def loss_histogram(losses: np.ndarray, var: float, es: float, title: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=losses, nbinsx=80, marker_color=SERIES[0],
                                 marker_line=dict(color=SURFACE, width=1),
                                 hovertemplate="Loss ≈ $%{x:,.0f}<br>count %{y}<extra></extra>"))
    for v, name, dash, pos in ((var, "VaR 99%", "dash", "top left"), (es, "ES 99%", "dot", "bottom right")):
        fig.add_vline(x=v, line=dict(color=TEXT, width=1.5, dash=dash), annotation_position=pos,
                      annotation_text=f"{name} ${v / 1e6:.2f}M", annotation_font_color=TEXT_2)
    fig = _layout(fig, title, height=320, xtitle="1-day loss (USD, positive = loss)", ytitle="scenarios")
    fig.update_xaxes(tickformat="$,.2s")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def horizontal_bars(s: pd.Series, title: str, fmt: str = "$,.3s", color: str = SERIES[0]) -> go.Figure:
    s = s.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=color,
                           marker_line=dict(color=SURFACE, width=2),
                           text=[f"${v / 1e6:.1f}M" for v in s.values], textposition="outside",
                           textfont=dict(color=TEXT_2),
                           hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
    fig = _layout(fig, title, height=max(260, 42 * len(s) + 80))
    fig.update_xaxes(tickformat=fmt, gridcolor=GRID, showgrid=True)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def spot_vs_oracle(ts: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    window = ts["divergence"] > 0
    # shade arbitrage window (oracle above spot) as contiguous bands
    start = None
    for t, w in zip(ts.index, window):
        if w and start is None:
            start = t
        if (not w) and start is not None:
            fig.add_vrect(x0=start - 0.5, x1=t - 0.5, fillcolor=WARNING, opacity=0.12, line_width=0)
            start = None
    if start is not None:
        fig.add_vrect(x0=start - 0.5, x1=ts.index[-1] + 0.5, fillcolor=WARNING, opacity=0.12, line_width=0)
    fig.add_scatter(x=ts.index, y=ts["spot"], name="DEX spot (incl. liquidation impact)",
                    line=dict(color=SERIES[0], width=2))
    fig.add_scatter(x=ts.index, y=ts["oracle"], name="Chainlink-style oracle", line=dict(color=SERIES[1], width=2, shape="hv"))
    upd = ts[ts["oracle_update"]]
    fig.add_scatter(x=upd.index, y=upd["oracle"], mode="markers", name="oracle round",
                    marker=dict(color=SERIES[1], size=8, line=dict(color=SURFACE, width=2)), showlegend=False,
                    hovertemplate="round posted<extra></extra>")
    return _layout(fig, title, yfmt="$,.0f", xtitle="minute")


def single_line(x, y, title: str, name: str, yfmt: str, color: str = SERIES[0], hline: float | None = None,
                hline_label: str = "", xtitle: str = "minute") -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, name=name, line=dict(color=color, width=2)))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=TEXT_2, width=1, dash="dash"),
                      annotation_text=hline_label, annotation_font_color=TEXT_2)
    fig = _layout(fig, title, height=300, yfmt=yfmt, xtitle=xtitle)
    fig.update_layout(showlegend=False)
    return fig


def liquidation_flow(ts: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=ts.index, y=ts["liq_demand_usd"], name="liquidatable demand", marker_color=SERIES[1], opacity=0.55)
    fig.add_bar(x=ts.index, y=ts["liq_executed_usd"], name="executed (profitable)", marker_color=SERIES[0])
    br = ts[ts["breach"]]
    fig.add_scatter(x=br.index, y=br["liq_demand_usd"], mode="markers", name="slippage > s* (LIF breached)",
                    marker=dict(color=CRITICAL, size=8, symbol="x"))
    fig = _layout(fig, title, height=320, yfmt="$,.2s", xtitle="minute")
    fig.update_layout(barmode="overlay", bargap=0.1)
    return fig


def bad_debt_curve(sweep: pd.DataFrame, sweep_agent: pd.DataFrame | None, breach_shock, bad_shock,
                   title: str) -> go.Figure:
    x = -sweep["shock"] * 100
    fig = go.Figure()
    fig.add_scatter(x=x, y=sweep["total_bad_debt"], name="vault bad debt", line=dict(color=SERIES[0], width=2))
    fig.add_scatter(x=x, y=sweep["kpk_loss"], name="KPK share (agent off)", line=dict(color=SERIES[1], width=2))
    if sweep_agent is not None:
        fig.add_scatter(x=-sweep_agent["shock"] * 100, y=sweep_agent["kpk_loss"], name="KPK share (exit agent on)",
                        line=dict(color=SERIES[2], width=2))
    if breach_shock is not None:
        fig.add_vline(x=-breach_shock * 100, line=dict(color=WARNING, width=1.5, dash="dash"),
                      annotation_text=f"slippage > s* at {breach_shock:.1%}", annotation_position="top left",
                      annotation_font_color=TEXT_2)
    if bad_shock is not None:
        fig.add_vline(x=-bad_shock * 100, line=dict(color=CRITICAL, width=1.5, dash="dash"),
                      annotation_text=f"bad debt from {bad_shock:.1%}", annotation_position="top right",
                      annotation_font_color=TEXT_2)
    fig = _layout(fig, title, yfmt="$,.2s", xtitle="crash severity (%)")
    fig.update_xaxes(ticksuffix="%")
    return fig


def strategy_lines(series: dict[str, pd.DataFrame], col: str, title: str, yfmt: str = "$,.3s") -> go.Figure:
    fig = go.Figure()
    for i, (name, df) in enumerate(series.items()):
        fig.add_scatter(x=df.index, y=df[col], name=name, line=dict(color=SERIES[i], width=2))
    return _layout(fig, title, yfmt=yfmt)


def price_with_ranges(passive: pd.DataFrame, rebal: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=passive.index, y=passive["range_high"], line=dict(color=SERIES[1], width=1, dash="dot"),
                    name="passive range", legendgroup="p")
    fig.add_scatter(x=passive.index, y=passive["range_low"], line=dict(color=SERIES[1], width=1, dash="dot"),
                    name="passive range", legendgroup="p", showlegend=False)
    fig.add_scatter(x=rebal.index, y=rebal["range_high"], line=dict(color=SERIES[2], width=1, shape="hv"),
                    name="rebalanced range", legendgroup="r")
    fig.add_scatter(x=rebal.index, y=rebal["range_low"], line=dict(color=SERIES[2], width=1, shape="hv"),
                    name="rebalanced range", legendgroup="r", showlegend=False)
    fig.add_scatter(x=passive.index, y=passive["price"], line=dict(color=SERIES[0], width=2), name="ETH/USDC")
    rb = rebal[rebal["rebalance"]]
    fig.add_scatter(x=rb.index, y=rb["price"], mode="markers", name="rebalance",
                    marker=dict(color=SERIES[2], size=9, line=dict(color=SURFACE, width=2)))
    return _layout(fig, title, yfmt="$,.0f")


def fee_vs_il_bars(summary: pd.DataFrame, title: str) -> go.Figure:
    s = summary.drop(index="HODL 50/50", errors="ignore")
    fig = go.Figure()
    fig.add_bar(name="Fees earned", x=s.index, y=s["fees_usd"], marker_color=SERIES[0],
                marker_line=dict(color=SURFACE, width=2))
    fig.add_bar(name="Impermanent loss (vs. holding LP tokens)", x=s.index, y=s["il_usd"], marker_color=SERIES[1],
                marker_line=dict(color=SURFACE, width=2))
    fig.add_bar(name="Theoretical LVR", x=s.index, y=-s["lvr_usd"], marker_color=SERIES[2],
                marker_line=dict(color=SURFACE, width=2))
    fig = _layout(fig, title, yfmt="$,.2s")
    fig.update_layout(barmode="group", hovermode="closest")
    fig.add_hline(y=0, line=dict(color=TEXT_2, width=1))
    return fig
