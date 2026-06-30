"""Render XAUUSD candlestick charts for Gemini vision input."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def render_candlestick_chart(
    bars: List[List[float]],
    *,
    timeframe: str,
    session: Optional[str] = None,
    symbol: str = "XAUUSD",
) -> Optional[bytes]:
    """Return PNG bytes for the last N OHLC bars."""
    if not HAS_MATPLOTLIB or not bars:
        return None

    opens, highs, lows, closes = [], [], [], []
    for bar in bars:
        if len(bar) < 5:
            continue
        opens.append(float(bar[1]))
        highs.append(float(bar[2]))
        lows.append(float(bar[3]))
        closes.append(float(bar[4]))

    if len(closes) < 5:
        return None

    bg = "#1a1a2e"
    last_close = closes[-1]
    sess = (session or "—").upper()
    title = f"{symbol} · {timeframe} · {sess} · {last_close:.2f}"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#2a2a3e")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.grid(True, color="#2a2a3e", alpha=0.4, linewidth=0.5)

    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        col = "#00e676" if c >= o else "#ff1744"
        ax.plot([i, i], [l, h], color=col, linewidth=0.8, zorder=2)
        body = abs(c - o) or (h - l) * 0.01
        ax.bar(i, body, bottom=min(o, c), color=col, width=0.6, alpha=0.88, zorder=2)

    ax.set_title(title, color="#ddddff", fontsize=11, pad=8)
    ax.set_ylabel("Price", color="#aaaacc", fontsize=8)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
