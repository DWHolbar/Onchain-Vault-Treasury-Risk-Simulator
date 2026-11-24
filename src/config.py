"""Shared configuration: portfolio state and protocol constants."""
from __future__ import annotations

from dataclasses import dataclass

# --- Morpho Blue liquidation constants (Morpho Blue whitepaper) ---
MORPHO_LIF_MAX = 1.15       # M: maximum liquidation incentive factor
MORPHO_LIF_BETA = 0.30      # beta: cursor
DEFAULT_LLTV = 0.86

# --- Uniswap v3 ETH/USDC 0.05% ---
UNIV3_FEE_TIER = 0.0005

RISK_FACTORS = ["ETH", "wstETH", "WBTC", "sDAI"]
CRYPTO_FACTORS = ["ETH", "wstETH", "WBTC"]


@dataclass(frozen=True)
class PortfolioState:
    total_capital: float = 50_000_000.0
    w_morpho: float = 0.50
    w_univ3: float = 0.20
    reserve_split_wsteth: float = 0.50   # share of reserves held as wstETH (rest sDAI)
    morpho_split_wsteth: float = 0.60    # share of Morpho supply in wstETH/USDC market (rest WBTC/USDC)
    data_source: str = "offline"         # "live" | "offline"
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0 <= self.w_morpho <= 1 and 0 <= self.w_univ3 <= 1):
            raise ValueError("weights must lie in [0, 1]")
        if self.w_morpho + self.w_univ3 > 1 + 1e-12:
            raise ValueError("w_morpho + w_univ3 must be <= 1")

    @property
    def w_reserves(self) -> float:
        return max(0.0, 1.0 - self.w_morpho - self.w_univ3)

    @property
    def morpho_usd(self) -> float:
        return self.total_capital * self.w_morpho

    @property
    def univ3_usd(self) -> float:
        return self.total_capital * self.w_univ3

    @property
    def reserves_usd(self) -> float:
        return self.total_capital * self.w_reserves

    @property
    def wsteth_reserve_usd(self) -> float:
        return self.reserves_usd * self.reserve_split_wsteth

    @property
    def sdai_reserve_usd(self) -> float:
        return self.reserves_usd * (1.0 - self.reserve_split_wsteth)

    def allocation(self) -> dict[str, float]:
        return {
            "Morpho wstETH/USDC": self.morpho_usd * self.morpho_split_wsteth,
            "Morpho WBTC/USDC": self.morpho_usd * (1 - self.morpho_split_wsteth),
            "Uniswap v3 ETH/USDC": self.univ3_usd,
            "wstETH reserve": self.wsteth_reserve_usd,
            "sDAI reserve": self.sdai_reserve_usd,
        }
