"""
BKS Test Assignment · SBER Spot vs SBERF Futures · 10.06.2025
===============================================================
1. Build order book (биржевой стакан) at arbitrary timestamp
2. Compute intraday spread dynamics
3. Run basis mean-reversion backtest (★)

Usage:
    python main.py
"""

import pandas as pd
from orderbook import OrderBook, load_spot, load_futures
from analysis import compute_spread, print_spread_stats, plot_spread
from backtest import run_backtest, print_backtest_stats, plot_backtest


SPOT_PATH = "data/20250610_SBER.parquet"
FUT_PATH = "data/SBERF_2025_06_10.parquet"

DEMO_TIMESTAMP = "2025-06-10 12:30:00"
FREQ = "1min"
SESSION_START = "2025-06-10 10:00"
SESSION_END = "2025-06-10 18:45"


def main():
    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    print("Loading data...")
    spot_df = load_spot(SPOT_PATH)
    fut_df = load_futures(FUT_PATH)
    print(f"  SBER rows  (filtered, price>0): {len(spot_df):,}")
    print(f"  SBERF rows (filtered, price>0): {len(fut_df):,}")

    # ------------------------------------------------------------------ #
    # 2. Build order books at a demo timestamp                             #
    # ------------------------------------------------------------------ #
    print(f"\nBuilding order book snapshots at {DEMO_TIMESTAMP} ...")
    spot_ob = OrderBook(spot_df)
    fut_ob = OrderBook(fut_df)

    spot_snap = spot_ob.get_snapshot(DEMO_TIMESTAMP, depth=10)
    fut_snap = fut_ob.get_snapshot(DEMO_TIMESTAMP, depth=10)

    print("\n--- SBER (spot) ---")
    spot_ob.print_book(spot_snap)

    print("--- SBERF (futures) ---")
    fut_ob.print_book(fut_snap)

    print(f"  Spot  mid: {spot_snap['mid']:.4f} руб")
    print(f"  Fut   mid: {fut_snap['mid']:.4f} руб")
    print(f"  Spread at {DEMO_TIMESTAMP}: {fut_snap['mid'] - spot_snap['mid']:.4f} руб")

    # ------------------------------------------------------------------ #
    # 3. Mid-price time series (O(N) pass through data)                   #
    # ------------------------------------------------------------------ #
    print(f"\nBuilding mid-price series (freq={FREQ}, {SESSION_START}–{SESSION_END}) ...")
    spot_mid = spot_ob.build_mid_series(freq=FREQ, start=SESSION_START, end=SESSION_END)
    fut_mid = fut_ob.build_mid_series(freq=FREQ, start=SESSION_START, end=SESSION_END)
    print(f"  Snapshot count: {len(spot_mid)}")

    # ------------------------------------------------------------------ #
    # 4. Spread analysis                                                   #
    # ------------------------------------------------------------------ #
    spread_df = compute_spread(spot_mid, fut_mid)
    print_spread_stats(spread_df)
    plot_spread(spread_df, output_path="outputs/spread_analysis.html")

    # ------------------------------------------------------------------ #
    # 5. Backtest ★                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "="*72)
    print("STRATEGY: Basis Mean-Reversion (SBERF − SBER) · risk-filtered")
    print("="*72)
    print("""
  Idea: the basis (futures − spot) fluctuates around a slow-moving fair
  value. Sharp deviations are typically caused by order-flow imbalances
  on one leg and revert within minutes. We fade those deviations.

  Signal: rolling stats (window=60 min) give local mean & σ of the spread.
          z = (spread − rolling_mean) / rolling_std.

  Entry: z > +2σ → SHORT spread (sell SBERF / buy SBER)
         z < −2σ → LONG  spread (buy  SBERF / sell SBER)

  Exit:  natural — when z crosses back through the mean to −0.5σ on the
         opposite side (≈2.5σ swing, ≈0.21 руб; bigger than 0.12 руб of
         round-trip costs).  This is a milder version of the pure
         mean-revert exit (it captures a small overshoot).

  Risk filters (the heart of the "less risky" variant):
    • Trading hours 10:15 — 18:15  — skip first/last 30 min of session
      (wide BA spreads, low liquidity, "exit poisoning" near close).
    • Skip entries when |5-min SBER return| > 0.35%
      (fast directional move → mean-reversion logic breaks).
    • Hard stop when |z| ≥ 4σ
      (caps tail-loss if dislocation is structural, not noise).
    • Max holding time 60 min
      (stale signals = noise; force-exit and free the capital).

  Cost model: 1 bps per leg = 2 bps round-trip ≈ 0.12 руб / trade.
  Passive (limit-order) execution on liquid SBER/SBERF.
""")
    bt_df = run_backtest(
        spread_df,
        window=60,
        entry_k=2.0,
        exit_k=-0.5,
        cost_bps=1.0,
        trade_start="10:15",
        trade_end="18:15",
        skip_5m_return=0.0035,
        stop_z=4.0,
        max_hold_min=60,
    )
    print_backtest_stats(bt_df, cost_bps=1.0)
    plot_backtest(bt_df, output_path="outputs/backtest.html")

    print("\nDone! Charts saved in outputs/")


if __name__ == "__main__":
    main()
