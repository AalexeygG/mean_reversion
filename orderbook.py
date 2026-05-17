"""
Order book reconstruction from MOEX full_orders_log.

Supports both spot (SBER) and futures (SBERF) formats.

MOEX ACTION codes:
  1 = new order added to book
  0 = order cancelled (volume = remaining volume at cancellation)
  2 = order partially/fully executed (volume = executed volume)

Key insight: ACTION=0 and ACTION=2 rows contain the original limit PRICE
and the exact volume to subtract, so we don't need to track orders by ID.
The book state at time T = sum of signed volumes per (side, price) level
for all events up to T.
"""

import pandas as pd
import numpy as np
from typing import Optional


def _parse_spot_time(df: pd.DataFrame) -> pd.Series:
    """Parse MOEX spot TIME column (HHMMSSSSSSSS, microseconds, leading zeros dropped)."""
    s = df["TIME"].astype(str).str.zfill(12)
    return pd.to_datetime(
        "2025-06-10 "
        + s.str[:2] + ":"
        + s.str[2:4] + ":"
        + s.str[4:6] + "."
        + s.str[6:]
    )


def load_spot(path: str) -> pd.DataFrame:
    """Load and preprocess SBER spot orders log."""
    df = pd.read_parquet(path)
    df["timestamp"] = _parse_spot_time(df)
    df["side"] = df["BUYSELL"]
    df = df[df["PRICE"] > 0].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def load_futures(path: str) -> pd.DataFrame:
    """Load and preprocess SBERF futures orders log."""
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["DateTime"])
    df["side"] = df["TYPE"]
    df = df[df["PRICE"] > 0].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


class OrderBook:
    """
    Reconstructs a limit order book from a preprocessed orders log dataframe.

    Usage:
        ob = OrderBook(df)
        snapshot = ob.get_snapshot("2025-06-10 12:30:00")
        ob.print_book(snapshot)
        mid_series = ob.build_mid_series(freq="1min")
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        df["_signed"] = np.where(df["ACTION"] == 1, df["VOLUME"], -df["VOLUME"])

    def get_snapshot(self, timestamp, depth: int = 10) -> dict:
        """
        Build order book at the given timestamp.

        Returns:
            bids: list[(price, volume)] sorted best-to-worst (descending price)
            asks: list[(price, volume)] sorted best-to-worst (ascending price)
            best_bid, best_ask, mid, bid_ask_spread
        """
        ts = pd.Timestamp(timestamp)
        sub = self.df[self.df["timestamp"] <= ts]

        book = (
            sub.groupby(["side", "PRICE"])["_signed"]
            .sum()
            .reset_index(name="volume")
        )
        book = book[book["volume"] > 0]

        bids_df = book[book["side"] == "B"].sort_values("PRICE", ascending=False).head(depth)
        asks_df = book[book["side"] == "S"].sort_values("PRICE").head(depth)

        bids = list(zip(bids_df["PRICE"], bids_df["volume"]))
        asks = list(zip(asks_df["PRICE"], asks_df["volume"]))

        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        spread = round(best_ask - best_bid, 4) if best_bid and best_ask else None

        return {
            "timestamp": ts,
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "bid_ask_spread": spread,
        }

    def print_book(self, snapshot: dict, depth: int = 10):
        """Pretty-print the order book snapshot."""
        ts = snapshot["timestamp"]
        print(f"\n{'='*50}")
        print(f"  Order Book at {ts}")
        print(f"{'='*50}")
        print(f"  {'PRICE':>10}  {'VOLUME':>12}  SIDE")
        print(f"  {'-'*38}")

        asks = snapshot["asks"][:depth]
        for price, vol in reversed(asks):
            print(f"  {price:>10.2f}  {vol:>12,}  ASK")

        mid = snapshot["mid"]
        spread = snapshot["bid_ask_spread"]
        print(f"  {'-'*38}")
        print(f"  {'mid':>10}  {mid:>10.4f}   spread={spread:.4f}")
        print(f"  {'-'*38}")

        bids = snapshot["bids"][:depth]
        for price, vol in bids:
            print(f"  {price:>10.2f}  {vol:>12,}  BID")
        print(f"{'='*50}\n")

    def build_mid_series(
        self,
        freq: str = "1min",
        start: str = "2025-06-10 10:00",
        end: str = "2025-06-10 18:45",
    ) -> pd.Series:
        """
        Efficiently build mid-price time series by single O(N) pass.

        Instead of calling get_snapshot() at each tick (which would be O(N*M)),
        we process events once and record mid-price at each snapshot moment.
        """
        snapshot_times = pd.date_range(start, end, freq=freq)

        df = self.df.reset_index(drop=True)
        n = len(df)
        event_idx = 0

        bids: dict[float, int] = {}
        asks: dict[float, int] = {}

        mids = []

        for snap_ts in snapshot_times:
            # Advance through events up to snap_ts
            while event_idx < n and df.at[event_idx, "timestamp"] <= snap_ts:
                side = df.at[event_idx, "side"]
                price = df.at[event_idx, "PRICE"]
                signed = df.at[event_idx, "_signed"]

                book_side = bids if side == "B" else asks
                book_side[price] = book_side.get(price, 0) + signed

                event_idx += 1

            # Filter for positive volumes only (handles orphan cancel/fill events
            # from prior sessions whose ACTION=1 is absent in today's data)
            best_bid = max((p for p, v in bids.items() if v > 0), default=None)
            best_ask = min((p for p, v in asks.items() if v > 0), default=None)
            mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
            mids.append(mid)

        return pd.Series(mids, index=snapshot_times, name="mid_price")

    def build_book_series(
        self,
        freq: str = "5min",
        start: str = "2025-06-10 10:00",
        end: str = "2025-06-10 18:45",
        depth: int = 12,
    ) -> list[dict]:
        """
        Single O(N) pass — returns full book snapshot at every freq interval.
        Each entry: {ts, bids: [(price, vol)], asks: [(price, vol)], mid}
        """
        snapshot_times = pd.date_range(start, end, freq=freq)
        df = self.df.reset_index(drop=True)
        n = len(df)
        event_idx = 0
        bids: dict[float, int] = {}
        asks: dict[float, int] = {}
        snapshots = []

        for snap_ts in snapshot_times:
            while event_idx < n and df.at[event_idx, "timestamp"] <= snap_ts:
                side  = df.at[event_idx, "side"]
                price = df.at[event_idx, "PRICE"]
                signed = df.at[event_idx, "_signed"]
                book_side = bids if side == "B" else asks
                book_side[price] = book_side.get(price, 0) + signed
                event_idx += 1

            bid_levels = sorted([(p, v) for p, v in bids.items() if v > 0], reverse=True)[:depth]
            ask_levels = sorted([(p, v) for p, v in asks.items() if v > 0])[:depth]

            best_bid = bid_levels[0][0] if bid_levels else None
            best_ask = ask_levels[0][0] if ask_levels else None
            mid = round((best_bid + best_ask) / 2, 4) if best_bid and best_ask else None

            snapshots.append({
                "ts":   snap_ts.strftime("%H:%M"),
                "bids": bid_levels,
                "asks": ask_levels,
                "mid":  mid,
            })

        return snapshots
