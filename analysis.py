"""
Spread dynamics between SBER (spot) and SBERF (futures).

Spread = futures_mid - spot_mid

This represents the basis between the two instruments.
Theoretically, basis ≈ Spot * r * (T/365) - D,
where r = risk-free rate, T = days to expiration, D = expected dividends.
On expiry day the basis converges to zero.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_spread(
    spot_mid: pd.Series,
    fut_mid: pd.Series,
    label_spot: str = "SBER (spot)",
    label_fut: str = "SBERF (futures)",
) -> pd.DataFrame:
    """
    Align mid-price series by timestamp and compute spread.

    Returns a DataFrame with columns:
        spot_mid, futures_mid, spread, spread_pct
    """
    df = pd.DataFrame({"spot_mid": spot_mid, "futures_mid": fut_mid}).dropna()
    df["spread"] = df["futures_mid"] - df["spot_mid"]
    df["spread_pct"] = df["spread"] / df["spot_mid"] * 100
    return df


def print_spread_stats(df: pd.DataFrame):
    """Print basic statistics of the spread."""
    sp = df["spread"]
    print("\n=== Spread Statistics (SBERF - SBER) ===")
    print(f"  Mean:        {sp.mean():.4f} руб")
    print(f"  Std:         {sp.std():.4f} руб")
    print(f"  Min:         {sp.min():.4f} руб")
    print(f"  Max:         {sp.max():.4f} руб")
    print(f"  Mean (%):    {df['spread_pct'].mean():.4f}%")
    print(f"  Std (%):     {df['spread_pct'].std():.4f}%")
    print(f"  Observations: {len(df)}")


def plot_spread(df: pd.DataFrame, output_path: str = "outputs/spread_analysis.html"):
    """
    Create interactive Plotly chart with:
      1. Spot and futures mid-price
      2. Spread (absolute and ±2σ bands)
      3. Spread as % of spot
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Mid-цена SBER и SBERF (10:00–18:45)",
            "Спред = SBERF − SBER (руб.)",
            "Спред (% от цены спота)",
        ],
        vertical_spacing=0.08,
        row_heights=[0.45, 0.30, 0.25],
    )

    # --- Panel 1: mid prices ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df["spot_mid"], name="SBER (спот)",
                   line=dict(color="#1f77b4", width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["futures_mid"], name="SBERF (фьючерс)",
                   line=dict(color="#ff7f0e", width=1.5)),
        row=1, col=1,
    )

    # --- Panel 2: spread with ±2σ bands ---
    mean_sp = df["spread"].mean()
    std_sp = df["spread"].std()

    fig.add_trace(
        go.Scatter(x=df.index, y=df["spread"], name="Спред",
                   line=dict(color="#2ca02c", width=1.5)),
        row=2, col=1,
    )
    for k, style in [(1, "dot"), (2, "dash")]:
        for sign, label in [(1, f"+{k}σ"), (-1, f"−{k}σ")]:
            fig.add_hline(
                y=mean_sp + sign * k * std_sp,
                line_dash=style, line_color="gray", line_width=1,
                annotation_text=label, annotation_position="right",
                row=2, col=1,
            )
    fig.add_hline(y=mean_sp, line_dash="solid", line_color="red",
                  line_width=1, annotation_text="mean", row=2, col=1)

    # --- Panel 3: spread % ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df["spread_pct"], name="Спред %",
                   line=dict(color="#9467bd", width=1.2),
                   fill="tozeroy", fillcolor="rgba(148,103,189,0.1)"),
        row=3, col=1,
    )

    fig.update_layout(
        title="Динамика спреда SBERF − SBER · 10.06.2025",
        height=800,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Цена (руб.)", row=1, col=1)
    fig.update_yaxes(title_text="Спред (руб.)", row=2, col=1)
    fig.update_yaxes(title_text="Спред (%)", row=3, col=1)
    fig.update_xaxes(
        title_text="",
        tickformat="%H:%M",
        dtick=30 * 60 * 1000,  # tick every 30 minutes (ms)
        tickangle=-45,
        row=3, col=1,
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=30 * 60 * 1000,
        tickangle=-45,
        row=1, col=1,
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        dtick=30 * 60 * 1000,
        tickangle=-45,
        row=2, col=1,
    )

    fig.write_html(output_path)
    print(f"Chart saved: {output_path}")
    return fig
