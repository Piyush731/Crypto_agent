"""Provider-neutral market-data contract."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class MarketDataProvider(ABC):
    """Public market data only. No order methods belong in this interface."""

    @abstractmethod
    def test_connection(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 300,
        completed_only: bool = True,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_multi_tf_ohlcv(
        self,
        symbol: str,
        timeframes: tuple[str, ...] = ("5m", "15m", "1h"),
        limit: int = 300,
        completed_only: bool = True,
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def get_ticker(self, symbol: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def get_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def get_open_interest(self, symbol: str) -> dict[str, Any] | None:
        raise NotImplementedError
