"""Market data acquisition.

Order of precedence for ``source="live"``:
    1. Live HTTP (CoinGecko prices/volumes + DefiLlama yields/pool stats)
    2. Local JSON cache (any age) if live fails
    3. Deterministic synthetic stress dataset

``source="offline"`` always returns the deterministic synthetic dataset, so the
app never fails when APIs are rate-limited or the network is unavailable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_TTL_S = 6 * 3600
HTTP_TIMEOUT_S = 10

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
DEFILLAMA_YIELDS = "https://yields.llama.fi"
COINGECKO_IDS = {
    "ETH": "ethereum",
    "wstETH": "wrapped-steth",
    "WBTC": "wrapped-bitcoin",
    "sDAI": "savings-dai",
}

DEFAULT_YIELDS = {"morpho_usdc_supply_apy": 0.06, "wsteth_apr": 0.03, "sdai_apy": 0.05}
DEFAULT_POOL_TVL = 150_000_000.0
DEFAULT_POOL_DAILY_VOLUME = 400_000_000.0


@dataclass
class MarketData:
    daily_prices: pd.DataFrame          # index: date, columns: ETH, wstETH, WBTC, sDAI (USD)
    hourly_eth: pd.DataFrame            # index: hourly ts, columns: price, volume_usd
    pool_tvl_usd: float
    yields: dict = field(default_factory=dict)
    source_used: str = "offline"        # "live" | "cache" | "offline"
    note: str = ""

    def to_json(self) -> dict:
        def df_js(df: pd.DataFrame) -> dict:
            return {
                "index": [ts.isoformat() for ts in df.index],
                "columns": list(df.columns),
                "data": df.to_numpy().tolist(),
            }

        return {
            "daily_prices": df_js(self.daily_prices),
            "hourly_eth": df_js(self.hourly_eth),
            "pool_tvl_usd": self.pool_tvl_usd,
            "yields": self.yields,
        }

    @classmethod
    def from_json(cls, js: dict, source_used: str, note: str = "") -> "MarketData":
        def js_df(d: dict) -> pd.DataFrame:
            return pd.DataFrame(d["data"], index=pd.to_datetime(d["index"]), columns=d["columns"])

        return cls(
            daily_prices=js_df(js["daily_prices"]),
            hourly_eth=js_df(js["hourly_eth"]),
            pool_tvl_usd=float(js["pool_tvl_usd"]),
            yields=dict(js["yields"]),
            source_used=source_used,
            note=note,
        )


# ---------------------------------------------------------------------------
# Deterministic synthetic dataset
# ---------------------------------------------------------------------------
def synthetic_market_data(seed: int = 42, days: int = 365) -> MarketData:
    """Seeded fat-tailed (Student-t, nu=4) GBM with scripted crash episodes."""
    rng = np.random.default_rng(seed)
    n = days * 24
    hours_per_year = 24 * 365
    nu = 4.0
    t_scale = np.sqrt(nu / (nu - 2.0))  # standardise t to unit variance

    z_eth = rng.standard_t(nu, n) / t_scale
    z_idio = rng.standard_t(nu, n) / t_scale
    rho_btc = 0.80
    z_btc = rho_btc * z_eth + np.sqrt(1 - rho_btc**2) * z_idio

    sig_eth = 0.65 / np.sqrt(hours_per_year)
    sig_btc = 0.50 / np.sqrt(hours_per_year)
    r_eth = -0.5 * sig_eth**2 + sig_eth * z_eth
    r_btc = -0.5 * sig_btc**2 + sig_btc * z_btc

    # Scripted crash episodes (log-return drift injected over a window).
    def inject(start_day: float, dur_h: int, total_move: float, beta_btc: float) -> None:
        s = int(start_day * 24)
        r_eth[s : s + dur_h] += np.log(1 + total_move) / dur_h
        r_btc[s : s + dur_h] += np.log(1 + beta_btc * total_move) / dur_h

    inject(days - 90 - 120, 36, -0.18, 0.6)      # earlier drawdown inside VaR window
    crash_day = days - 90 + 58                   # day 58 of the 90-day backtest window
    inject(crash_day, 48, -0.28, 0.7)            # -28% over 48h
    inject(crash_day + 2, 7 * 24, 0.12, 0.6)     # partial recovery

    eth = 3200.0 * np.exp(np.cumsum(r_eth))
    btc = 95_000.0 * np.exp(np.cumsum(r_btc))

    # stETH/ETH peg: tight around 0.9995, dips to 0.97 in the crash.
    t_years = np.arange(n) / hours_per_year
    peg = 0.9995 + 0.0005 * np.sin(np.arange(n) / 500.0)
    c0 = int(crash_day * 24)
    dip = np.zeros(n)
    hrs = np.arange(n) - c0
    dip_mask = hrs >= 0
    dip[dip_mask] = 0.0295 * (1 - np.exp(-hrs[dip_mask] / 12.0)) * np.exp(-np.maximum(hrs[dip_mask] - 48, 0) / 72.0)
    peg = peg - dip
    wsteth_rate = 1.18 * np.exp(0.03 * t_years)
    wsteth = eth * peg * wsteth_rate
    # sDAI: DSR accrual x small secondary-market basis (AR(1), ~3 bps daily noise).
    basis = np.zeros(n)
    e = rng.standard_normal(n) * 0.0003 / np.sqrt(24)
    for i in range(1, n):
        basis[i] = 0.995 * basis[i - 1] + e[i]
    sdai = 1.12 * np.exp(0.05 * t_years + basis)

    idx = pd.date_range(end=pd.Timestamp("2026-11-01"), periods=n, freq="h")
    hourly = pd.DataFrame({"ETH": eth, "wstETH": wsteth, "WBTC": btc, "sDAI": sdai}, index=idx)
    daily = hourly.resample("1D").last()

    vol_weight = 1.0 + 3.0 * np.abs(r_eth) / sig_eth
    vol_weight = vol_weight / vol_weight.mean()
    volume = DEFAULT_POOL_DAILY_VOLUME / 24.0 * vol_weight
    hourly_eth = pd.DataFrame({"price": eth, "volume_usd": volume}, index=idx).iloc[-90 * 24 :]

    return MarketData(
        daily_prices=daily,
        hourly_eth=hourly_eth,
        pool_tvl_usd=DEFAULT_POOL_TVL,
        yields=dict(DEFAULT_YIELDS),
        source_used="offline",
        note=f"Deterministic synthetic stress dataset (seed={seed})",
    )


# ---------------------------------------------------------------------------
# Live fetchers
# ---------------------------------------------------------------------------
def _get_json(url: str, params: dict | None = None) -> dict | list:
    for attempt in range(2):
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_S)
        if resp.status_code == 429 and attempt == 0:
            time.sleep(2.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"rate-limited: {url}")


def _cg_market_chart(coin_id: str, days: int) -> pd.DataFrame:
    js = _get_json(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", {"vs_currency": "usd", "days": days})
    px = pd.DataFrame(js["prices"], columns=["ts", "price"])
    vol = pd.DataFrame(js["total_volumes"], columns=["ts", "volume"])
    df = px.merge(vol, on="ts", how="left")
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["price", "volume"]]


def _llama_pools() -> list[dict]:
    js = _get_json(f"{DEFILLAMA_YIELDS}/pools")
    return js.get("data", [])


def _pick_pool(pools: list[dict], **crit) -> dict | None:
    cands = [p for p in pools if all(str(p.get(k, "")).lower() == str(v).lower() for k, v in crit.items())]
    if not cands:
        return None
    return max(cands, key=lambda p: p.get("tvlUsd") or 0)


def fetch_live_market_data() -> MarketData:
    daily_cols, hourly_eth = {}, None
    for sym, cid in COINGECKO_IDS.items():
        d = _cg_market_chart(cid, 365)
        daily_cols[sym] = d["price"].resample("1D").last()
        if sym == "ETH":
            h = _cg_market_chart(cid, 90)
            hourly_eth = h.resample("1h").last().ffill()
    daily = pd.DataFrame(daily_cols).ffill().dropna()

    yields = dict(DEFAULT_YIELDS)
    pool_tvl, pool_vol = DEFAULT_POOL_TVL, DEFAULT_POOL_DAILY_VOLUME
    try:
        pools = _llama_pools()
        uni = _pick_pool(pools, project="uniswap-v3", chain="Ethereum", symbol="USDC-WETH", poolMeta="0.05%")
        if uni:
            pool_tvl = float(uni.get("tvlUsd") or pool_tvl)
            pool_vol = float(uni.get("volumeUsd1d") or pool_vol)
        sdai = _pick_pool(pools, chain="Ethereum", symbol="SDAI")
        if sdai and sdai.get("apy") is not None:
            yields["sdai_apy"] = float(sdai["apy"]) / 100
        lido = _pick_pool(pools, project="lido", chain="Ethereum", symbol="STETH")
        if lido and lido.get("apy") is not None:
            yields["wsteth_apr"] = float(lido["apy"]) / 100
        morpho = [p for p in pools if str(p.get("project", "")).startswith("morpho")
                  and p.get("chain") == "Ethereum" and str(p.get("symbol", "")).upper() == "USDC"]
        if morpho:
            best = max(morpho, key=lambda p: p.get("tvlUsd") or 0)
            if best.get("apy") is not None:
                yields["morpho_usdc_supply_apy"] = float(best["apy"]) / 100
    except Exception:  # DefiLlama optional: keep defaults
        pass

    # Rescale CoinGecko aggregate ETH volume profile to the pool's daily volume.
    vol_share = hourly_eth["volume"] / hourly_eth["volume"].mean()
    hourly = pd.DataFrame(
        {"price": hourly_eth["price"], "volume_usd": pool_vol / 24.0 * vol_share.fillna(1.0)}
    ).dropna()

    return MarketData(daily, hourly, pool_tvl, yields, source_used="live",
                      note="CoinGecko prices + DefiLlama yields/pool stats")


# ---------------------------------------------------------------------------
# Cache + public entry point
# ---------------------------------------------------------------------------
def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _cache_save(key: str, md: MarketData) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"saved_at": time.time(), "data": md.to_json()}
    _cache_path(key).write_text(json.dumps(payload))


def _cache_load(key: str) -> tuple[dict, float] | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        return payload["data"], time.time() - float(payload["saved_at"])
    except Exception:
        return None


def get_market_data(source: str = "offline", seed: int = 42) -> MarketData:
    if source != "live":
        return synthetic_market_data(seed)

    cached = _cache_load("market_bundle")
    if cached and cached[1] < CACHE_TTL_S:
        return MarketData.from_json(cached[0], "cache", f"Fresh cache, age {cached[1] / 60:.0f} min")
    try:
        md = fetch_live_market_data()
        _cache_save("market_bundle", md)
        return md
    except Exception as exc:
        if cached:
            return MarketData.from_json(
                cached[0], "cache", f"Live fetch failed ({type(exc).__name__}); stale cache age {cached[1] / 3600:.1f} h"
            )
        md = synthetic_market_data(seed)
        md.note = f"Live fetch failed ({type(exc).__name__}); using deterministic synthetic dataset"
        return md
