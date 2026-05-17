"""
Basis Mean-Reversion Strategy Backtest
========================================
SBERF (futures) vs SBER (spot) · 10.06.2025

Strategy logic:
  - Signal uses a rolling window to estimate the "fair" spread level.
  - Entry: when spread deviates more than k*σ from rolling mean.
  - Exit:  when spread returns within exit_threshold*σ of rolling mean,
           or at end of session (forced close).

Positions:
  +1 = long spread  (buy futures, sell spot)   → bet on spread widening
  -1 = short spread (sell futures, buy spot)   → bet on spread narrowing
   0 = flat

Assumptions:
  - Execution at mid-price (optimistic; add slippage in production).
  - Transaction cost: cost_bps basis points per leg (2 legs per trade).
  - No margin / funding costs modelled.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_backtest(
    spread_df: pd.DataFrame,
    window: int = 60,
    entry_k: float = 2.0,
    exit_k: float = -0.5,
    cost_bps: float = 1.0,
    trade_start: str = "10:15",
    trade_end: str = "18:15",
    skip_5m_return: float = 0.0035,
    stop_z: float = 3.5,
    max_hold_min: int = 40,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    spread_df       : DataFrame with columns spot_mid, futures_mid, spread
    window          : rolling window in bars (minutes) for mean/std estimation
    entry_k         : entry threshold in σ units
    exit_k          : exit threshold in σ units (negative = overshoot past mean)
    cost_bps        : transaction cost per leg (basis points)
    trade_start/end : trading-hours filter (HH:MM, MSK).  Entries outside are skipped;
                      open positions are forced flat at trade_end.
    skip_5m_return  : skip entries when |5-min SBER return| exceeds this (e.g. 0.0035 = 0.35%).
                      Filters out fast directional moves where mean-reversion logic breaks.
    stop_z          : emergency stop when |z| crosses this (in σ).  Caps tail loss.
    max_hold_min    : close any open trade after this many minutes.  Stale signals = noise.
    """
    df = spread_df.copy()

    df["roll_mean"] = df["spread"].rolling(window, min_periods=window // 2).mean()
    df["roll_std"] = df["spread"].rolling(window, min_periods=window // 2).std()
    df["z_score"] = (df["spread"] - df["roll_mean"]) / df["roll_std"].replace(0, np.nan)
    df["spot_5m_ret"] = df["spot_mid"].pct_change(5).abs()

    t_start = pd.Timestamp(f"2025-06-10 {trade_start}").time()
    t_end   = pd.Timestamp(f"2025-06-10 {trade_end}").time()

    position = 0
    positions = []
    entry_price_spot = 0.0
    entry_price_fut = 0.0
    entry_time = None
    entry_cost = 0.0      # cost paid on entry, attributed to round-trip P&L
    exit_reasons = []     # per-bar reason tag (or "")
    trade_pnl_list = []   # round-trip net P&L, recorded on exit bar (else 0)

    cost_factor = cost_bps / 10_000

    gross_pnl_list = []
    net_pnl_list = []
    costs_list = []

    for ts, row in df.iterrows():
        z = row["z_score"]
        spot = row["spot_mid"]
        fut = row["futures_mid"]
        big_move = row["spot_5m_ret"]
        t = ts.time()

        trade_cost = 0.0
        bar_pnl = 0.0
        exit_reason = ""
        round_trip_pnl = 0.0

        if pd.isna(z):
            positions.append(position)
            gross_pnl_list.append(0.0)
            net_pnl_list.append(0.0)
            costs_list.append(0.0)
            exit_reasons.append("")
            trade_pnl_list.append(0.0)
            continue

        # ── Exit logic ────────────────────────────────────────────────────
        if position != 0:
            held_min = (ts - entry_time).total_seconds() / 60 if entry_time else 0
            forced_close = (t >= t_end)              # end-of-session window
            stopped      = abs(z) >= stop_z          # emergency stop
            timed_out    = held_min >= max_hold_min  # stale trade

            # natural mean-reversion exit (z crossed back through threshold)
            natural = ((position == 1 and z >= -exit_k) or
                       (position == -1 and z <= exit_k))

            if natural or forced_close or stopped or timed_out:
                if position == 1:
                    bar_pnl = (fut - entry_price_fut) - (spot - entry_price_spot)
                else:
                    bar_pnl = (entry_price_fut - fut) - (entry_price_spot - spot)
                trade_cost = (spot + fut) * cost_factor
                round_trip_pnl = bar_pnl - entry_cost - trade_cost   # full round-trip net
                exit_reason = ("stop"      if stopped else
                               "timeout"   if timed_out else
                               "end-of-day" if forced_close else "natural")
                position = 0
                entry_time = None
                entry_cost = 0.0

        # ── Entry logic (only when flat AND within trading hours AND no big move) ──
        if position == 0 and t_start <= t < t_end and (pd.isna(big_move) or big_move <= skip_5m_return):
            if z > entry_k:
                position = -1
                entry_price_spot, entry_price_fut = spot, fut
                entry_time = ts
                entry_cost = (spot + fut) * cost_factor
                trade_cost += entry_cost
            elif z < -entry_k:
                position = 1
                entry_price_spot, entry_price_fut = spot, fut
                entry_time = ts
                entry_cost = (spot + fut) * cost_factor
                trade_cost += entry_cost

        positions.append(position)
        gross_pnl_list.append(bar_pnl)
        costs_list.append(trade_cost)
        net_pnl_list.append(bar_pnl - trade_cost)
        exit_reasons.append(exit_reason)
        trade_pnl_list.append(round_trip_pnl)

    # Force-close any still-open position at very end (covers trade_end == session_end)
    if position != 0:
        last = df.iloc[-1]
        spot, fut = last["spot_mid"], last["futures_mid"]
        if position == 1:
            final_pnl = (fut - entry_price_fut) - (spot - entry_price_spot)
        else:
            final_pnl = (entry_price_fut - fut) - (entry_price_spot - spot)
        cost = (spot + fut) * cost_factor
        net_pnl_list[-1] += final_pnl - cost
        gross_pnl_list[-1] += final_pnl
        costs_list[-1] += cost
        exit_reasons[-1] = "end-of-day"
        trade_pnl_list[-1] = final_pnl - entry_cost - cost

    df["position"] = positions
    df["gross_pnl"] = gross_pnl_list
    df["trade_cost"] = costs_list
    df["net_pnl"] = net_pnl_list
    df["trade_pnl"] = trade_pnl_list
    df["exit_reason"] = exit_reasons
    df["cum_net_pnl"] = df["net_pnl"].cumsum()
    df["cum_gross_pnl"] = df["gross_pnl"].cumsum()

    return df


def print_backtest_stats(df: pd.DataFrame, cost_bps: float = 1.0):
    """Print summary statistics of the backtest, counted per round-trip."""
    # One row per closed round-trip (entry+exit combined)
    rt = df[df["trade_pnl"] != 0]["trade_pnl"] if "trade_pnl" in df.columns else df[df["net_pnl"] != 0]["net_pnl"]
    n_trades = len(rt)
    wins   = int((rt > 0).sum())
    losses = int((rt < 0).sum())

    total_net   = df["net_pnl"].sum()
    total_gross = df["gross_pnl"].sum()
    total_cost  = df["trade_cost"].sum()

    avg_win  = rt[rt > 0].mean() if wins else 0
    avg_loss = rt[rt < 0].mean() if losses else 0
    pf       = abs(avg_win * wins / (avg_loss * losses)) if losses and avg_loss else float("inf")

    # Sharpe annualised assuming 252 trading days × 510 minutes per day
    sharpe = (rt.mean() / rt.std() * np.sqrt(252 * 510)
              if len(rt) > 1 and rt.std() > 0 else 0)

    max_dd = (df["cum_net_pnl"] - df["cum_net_pnl"].cummax()).min()

    print("\n=== Backtest Results ===")
    print(f"  Strategy:        Basis Mean-Reversion (SBERF − SBER)")
    print(f"  Cost per leg:    {cost_bps} bps  (round-trip ≈ {4*cost_bps/100:.2f}%)")
    print(f"  Total trades:    {n_trades}  (closed round-trips)")
    if n_trades:
        print(f"  Wins / Losses:   {wins} / {losses}  (win rate {wins/n_trades*100:.1f}%)")
        print(f"  Avg win:         {avg_win:+.4f} руб")
        print(f"  Avg loss:        {avg_loss:+.4f} руб")
        print(f"  Profit factor:   {pf:.2f}")
    print(f"  Gross P&L:       {total_gross:+.4f} руб")
    print(f"  Total costs:     {total_cost:.4f} руб")
    print(f"  Net P&L:         {total_net:+.4f} руб")
    print(f"  Max Drawdown:    {max_dd:.4f} руб")
    print(f"  Sharpe (ann.):   {sharpe:.2f}")

    if "exit_reason" in df.columns:
        breakdown = df[df["exit_reason"] != ""]["exit_reason"].value_counts()
        if len(breakdown):
            print(f"  Exit breakdown:  " +
                  ", ".join(f"{k}={v}" for k, v in breakdown.items()))


def plot_backtest(df: pd.DataFrame, output_path: str = "outputs/backtest.html"):
    """Create interactive backtest visualisation."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Спред и z-score с сигналами",
            "Позиция",
            "Накопленный P&L (нетто, руб.)",
        ],
        vertical_spacing=0.10,
        row_heights=[0.45, 0.20, 0.35],
    )

    # ── Panel 1: spread + ±entry σ band (one legend entry) ───────────────
    fig.add_trace(
        go.Scatter(x=df.index, y=df["spread"], name="Спред",
                   line=dict(color="#2ca02c", width=1.5),
                   legendgroup="spread", legendgrouptitle_text="Спред"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["roll_mean"], name="Скользящее среднее",
                   line=dict(color="#e74c3c", width=1, dash="dash"),
                   legendgroup="spread"),
        row=1, col=1,
    )
    # σ band: upper trace invisible in legend, lower trace owns the legend item
    fig.add_trace(
        go.Scatter(x=df.index, y=df["roll_mean"] + 2.0 * df["roll_std"],
                   line=dict(color="rgba(128,128,128,0)", width=0),
                   showlegend=False, hoverinfo="skip",
                   legendgroup="spread"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["roll_mean"] - 2.0 * df["roll_std"],
                   name="±2σ полоса входа",
                   line=dict(color="rgba(128,128,128,0)", width=0),
                   fill="tonexty", fillcolor="rgba(128,128,128,0.12)",
                   hoverinfo="skip", legendgroup="spread"),
        row=1, col=1,
    )

    # ── Trade markers (entries/exits) ────────────────────────────────────
    entries = df[df["position"] != df["position"].shift(1).fillna(0)]
    long_entries  = entries[entries["position"] == 1]
    short_entries = entries[entries["position"] == -1]
    exits         = entries[entries["position"] == 0]

    fig.add_trace(
        go.Scatter(x=long_entries.index, y=long_entries["spread"],
                   mode="markers", name="Вход LONG",
                   marker=dict(symbol="triangle-up", color="#2980b9", size=11,
                               line=dict(color="white", width=1)),
                   legendgroup="trades", legendgrouptitle_text="Сделки"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=short_entries.index, y=short_entries["spread"],
                   mode="markers", name="Вход SHORT",
                   marker=dict(symbol="triangle-down", color="#e67e22", size=11,
                               line=dict(color="white", width=1)),
                   legendgroup="trades"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=exits.index, y=exits["spread"],
                   mode="markers", name="Выход",
                   marker=dict(symbol="x", color="#2c3e50", size=9),
                   legendgroup="trades"),
        row=1, col=1,
    )

    # ── Panel 2: position ────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=df.index, y=df["position"], name="Позиция",
                   line=dict(color="#8e44ad", width=1.5),
                   fill="tozeroy", fillcolor="rgba(142,68,173,0.18)",
                   legendgroup="pos", legendgrouptitle_text="Позиция"),
        row=2, col=1,
    )

    # ── Panel 3: cumulative P&L ──────────────────────────────────────────
    fig.add_trace(
        go.Scatter(x=df.index, y=df["cum_gross_pnl"], name="Gross P&L",
                   line=dict(color="#17becf", width=1, dash="dot"),
                   legendgroup="pnl", legendgrouptitle_text="P&L"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["cum_net_pnl"], name="Net P&L",
                   line=dict(color="#d62728", width=2),
                   legendgroup="pnl"),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5, row=3, col=1)

    fig.update_layout(
        title=dict(
            text="Backtest: Basis Mean-Reversion · SBERF − SBER · 10.06.2025",
            x=0.5, xanchor="center", y=0.985, yanchor="top",
            font=dict(size=15),
        ),
        height=900,
        template="plotly_white",
        margin=dict(t=70, b=130, l=70, r=30),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#ddd", borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
            tracegroupgap=22,
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Спред (руб.)", row=1, col=1)
    fig.update_yaxes(title_text="Позиция", row=2, col=1, tickvals=[-1, 0, 1])
    fig.update_yaxes(title_text="P&L (руб.)", row=3, col=1)
    fig.update_xaxes(
        tickformat="%H:%M", dtick=30 * 60 * 1000, tickangle=-45, row=3, col=1,
    )
    fig.update_xaxes(tickformat="%H:%M", dtick=30 * 60 * 1000, row=1, col=1)
    fig.update_xaxes(tickformat="%H:%M", dtick=30 * 60 * 1000, row=2, col=1)

    fig.write_html(output_path)
    print(f"Chart saved: {output_path}")
    return fig
