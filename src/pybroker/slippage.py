"""Implements slippage models."""

"""Copyright (C) 2023 Edward West. All rights reserved.

This code is licensed under Apache 2.0 with Commons Clause license
(see LICENSE for details).
"""

import random
from pybroker.context import ExecContext
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

import pandas as pd
import ta.volatility


class SlippageModel(ABC):
    """Base class for implementing a slippage model."""

    @abstractmethod
    def apply_slippage(
        self,
        ctx: ExecContext,
        buy_shares: Optional[Decimal] = None,
        sell_shares: Optional[Decimal] = None,
    ):
        """Applies slippage to ``ctx``."""


class RandomSlippageModel(SlippageModel):
    """Implements a simple random slippage model.

    Args:
        min_pct: Min percentage of slippage.
        max_pct: Max percentage of slippage.
    """

    def __init__(self, min_pct: float, max_pct: float):
        if min_pct < 0 or min_pct > 100:
            raise ValueError(r"min_pct must be between 0% and 100%.")
        if max_pct < 0 or max_pct > 100:
            raise ValueError(r"max_pct must be between 0% and 100%.")
        if min_pct >= max_pct:
            raise ValueError("min_pct must be < max_pct.")
        self.min_pct = min_pct / 100.0
        self.max_pct = max_pct / 100.0

    def apply_slippage(
        self,
        ctx: ExecContext,
        buy_shares: Optional[Decimal] = None,
        sell_shares: Optional[Decimal] = None,
    ):
        if buy_shares or sell_shares:
            slippage_pct = Decimal(random.uniform(self.min_pct, self.max_pct))
            if buy_shares:
                ctx.buy_shares = buy_shares - slippage_pct * buy_shares
            if sell_shares:
                ctx.sell_shares = sell_shares - slippage_pct * sell_shares


class VolatilityVolumeSlippageModel(SlippageModel):
    """
    Applies slippage based on market volatility (ATR) and order size relative
    to average volume. Size factor is dynamically adjusted based on portfolio cash.
    Falls back to volatility-only if volume data is unavailable.

    Args:
        data (pd.DataFrame): DataFrame containing historical OHLCV data,
                             Needs 'date', 'symbol', 'high', 'low', 'close'.
                             'volume' is optional. Assumes date is index or column.
        atr_window (int): Window period for ATR calculation. Defaults to 14.
        vol_window (int): Window period for rolling average volume calculation. Defaults to 20.
        vol_factor (float): Scaling factor for the volatility component of slippage. Defaults to 0.1.
        cash_thresholds (List[float]): List of cash thresholds for adjusting size_factor.
                                       Defaults to [500_000, 1_000_000].
        size_factors (List[float]): List of size factors corresponding to cash thresholds.
                                    Must be one longer than cash_thresholds.
                                    Defaults to [0.01, 0.03, 0.05].
    """

    def __init__(
        self,
        data: pd.DataFrame,
        atr_window: int = 14,
        vol_window: int = 20,
        vol_factor: float = 0.1,
        cash_thresholds: list[float] = [500_000.0, 1_000_000.0],
        size_factors: list[float] = [0.01, 0.03, 0.05],
    ):

        if len(size_factors) != len(cash_thresholds) + 1:
            raise ValueError(
                "Length of size_factors must be one more than cash_thresholds."
            )

        self.atr_window = atr_window
        self.vol_window = vol_window
        self.vol_factor = Decimal(str(vol_factor))
        self.cash_thresholds_dec = [Decimal(str(t)) for t in sorted(cash_thresholds)]
        self.size_factors_dec = [Decimal(str(f)) for f in size_factors]
        self.metrics = {}

        if "date" in data.columns:
            data = data.set_index("date")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex.")

        required_cols = ["symbol", "high", "low", "close"]
        has_required = all(col in data.columns for col in required_cols)
        has_volume = "volume" in data.columns

        if not has_required:
            raise ValueError(
                "Input data must contain 'symbol', 'high', 'low', 'close' columns."
            )

        for symbol, group_df in data.groupby("symbol"):
            group_df = group_df.sort_index()

            symbol_metrics = pd.DataFrame(index=group_df.index)

            symbol_metrics["atr"] = ta.volatility.AverageTrueRange(
                high=group_df["high"],
                low=group_df["low"],
                close=group_df["close"],
                window=self.atr_window,
                fillna=True,
            ).average_true_range()

            if has_volume and not group_df["volume"].isnull().all():
                vol_filled = group_df["volume"].ffill()
                symbol_metrics["avg_volume"] = (
                    vol_filled.ewm(span=self.vol_window, min_periods=1, adjust=False)
                    .mean()
                    .fillna(0)
                )
            else:
                symbol_metrics["avg_volume"] = Decimal("0")

            symbol_metrics["atr"] = symbol_metrics["atr"].apply(
                lambda x: Decimal(str(x)) if pd.notna(x) else Decimal("0")
            )
            symbol_metrics["avg_volume"] = symbol_metrics["avg_volume"].apply(
                lambda x: Decimal(str(x)) if pd.notna(x) else Decimal("0")
            )

            self.metrics[symbol] = symbol_metrics

    def apply_slippage(
        self,
        ctx: ExecContext,
        buy_shares: Optional[Decimal] = None,
        sell_shares: Optional[Decimal] = None,
    ):
        if not (buy_shares or sell_shares):
            return

        symbol = ctx.symbol
        try:
            current_date = ctx.dt
            current_date = (
                current_date.tz_localize(None) if current_date.tzinfo else current_date
            )
            current_cash = ctx.cash

        except AttributeError:
            print(
                f"Slippage Warning: ctx.dt or ctx.cash not found for {symbol}. Cannot apply slippage."
            )
            if buy_shares:
                ctx.buy_shares = buy_shares
            if sell_shares:
                ctx.sell_shares = sell_shares
            return

        try:
            symbol_metrics = self.metrics.get(symbol)
            if symbol_metrics is None:
                print(
                    f"Slippage Warning: No metrics pre-calculated for symbol {symbol}. Applying zero slippage."
                )
                if buy_shares:
                    ctx.buy_shares = buy_shares
                if sell_shares:
                    ctx.sell_shares = sell_shares
                return

            metrics_today = (
                symbol_metrics.loc[:current_date].iloc[-1]
                if current_date >= symbol_metrics.index.min()
                else None
            )

            if metrics_today is None or metrics_today.name != current_date:
                print(
                    f"Slippage Warning: Metrics not found for {symbol} on or before {current_date}. Applying zero slippage."
                )
                if buy_shares:
                    ctx.buy_shares = buy_shares
                if sell_shares:
                    ctx.sell_shares = sell_shares
                return

            current_atr = metrics_today["atr"]
            current_avg_volume = metrics_today["avg_volume"]

            if hasattr(ctx, "fill_price") and ctx.fill_price:
                current_price = ctx.fill_price
            elif hasattr(ctx, "close") and len(ctx.close) > 0:
                current_price = Decimal(str(ctx.close[-1]))
            else:
                print(
                    f"Slippage Warning: Cannot determine price for {symbol} on {current_date}. Applying zero slippage."
                )
                if buy_shares:
                    ctx.buy_shares = buy_shares
                if sell_shares:
                    ctx.sell_shares = sell_shares
                return

            slippage_pct = Decimal("0")

            if not isinstance(current_price, Decimal):
                current_price = Decimal(str(current_price))

            if current_price > 0:
                relative_volatility = current_atr / current_price
                slippage_pct += relative_volatility * self.vol_factor

            dynamic_size_factor = self.size_factors_dec[-1]
            for i, threshold in enumerate(self.cash_thresholds_dec):
                if current_cash < threshold:
                    dynamic_size_factor = self.size_factors_dec[i]
                    break

            if current_avg_volume > 0:
                order_shares = buy_shares if buy_shares else sell_shares
                if order_shares:
                    if not isinstance(order_shares, Decimal):
                        order_shares = Decimal(str(order_shares))
                    size_impact = order_shares / current_avg_volume
                    slippage_pct += size_impact * dynamic_size_factor

            slippage_pct = max(Decimal("0"), slippage_pct)

            one_decimal = Decimal("1")
            if buy_shares:
                ctx.buy_shares = buy_shares * (one_decimal - slippage_pct)
            if sell_shares:
                ctx.sell_shares = sell_shares * (one_decimal - slippage_pct)

        except KeyError as e:
            print(
                f"Slippage Warning: Metric key error for {symbol} on {current_date}: {e}. Applying zero slippage."
            )
            if buy_shares:
                ctx.buy_shares = buy_shares
            if sell_shares:
                ctx.sell_shares = sell_shares
        except Exception as e:
            import traceback

            print(
                f"Slippage Error for {symbol} on {current_date}: {e}\n{traceback.format_exc()}. Applying zero slippage."
            )
            if buy_shares:
                ctx.buy_shares = buy_shares
            if sell_shares:
                ctx.sell_shares = sell_shares
