"""Render XAUUSD candlestick charts for Gemini vision input."""
from __future__ import annotations

import logging
from io import BytesIO
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_MIN_PNG_BYTES = 2_500
_MIN_PIXEL_STD = 8.0

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    MaxNLocator = None  # type: ignore


def _parse_ohlc(bars: Sequence[Sequence[float]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    opens: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    for bar in bars:
        if len(bar) < 5:
            continue
        try:
            o = float(bar[1])
            h = float(bar[2])
            l = float(bar[3])
            c = float(bar[4])
        except (TypeError, ValueError):
            continue
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            continue
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
    return opens, highs, lows, closes


def summarize_bars_for_prompt(
    bars: Sequence[Sequence[float]],
    *,
    label: str,
    tail: int = 5,
) -> str:
    """Compact OHLC tail for the text prompt (vision fallback)."""
    opens, highs, lows, closes = _parse_ohlc(bars)
    if len(closes) < 1:
        return f"{label}: no usable OHLC"
    start = max(0, len(closes) - max(1, tail))
    lines = [f"{label} ({len(closes)} bars):"]
    for i in range(start, len(closes)):
        lines.append(
            f"  #{i}: O={opens[i]:.2f} H={highs[i]:.2f} L={lows[i]:.2f} C={closes[i]:.2f}"
        )
    lines.append(f"  range: {min(lows):.2f} – {max(highs):.2f} | last={closes[-1]:.2f}")
    return "\n".join(lines)


def chart_png_is_valid(png: Optional[bytes]) -> bool:
    """Reject empty/corrupt/flat PNGs before burning a Gemini vision call."""
    if not png or len(png) < _MIN_PNG_BYTES:
        return False
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    try:
        from PIL import Image
        import numpy as np

        arr = np.array(Image.open(BytesIO(png)).convert("RGB"))
        if arr.size == 0:
            return False
        if float(arr.std()) < _MIN_PIXEL_STD:
            return False
        return True
    except Exception:
        # Pillow optional — size + PNG header is a weak fallback.
        return len(png) >= 8_000


def render_candlestick_chart(
    bars: List[List[float]],
    *,
    timeframe: str,
    session: Optional[str] = None,
    symbol: str = "XAUUSD",
) -> Optional[bytes]:
    """Return PNG bytes for the last N OHLC bars (light theme for vision models)."""
    if not HAS_MATPLOTLIB or not bars:
        return None

    opens, highs, lows, closes = _parse_ohlc(bars)
    if len(closes) < 5:
        logger.warning(
            "[gemini-gold] chart render skipped tf=%s — only %s valid OHLC bars",
            timeframe,
            len(closes),
        )
        return None

    bg = "#ffffff"
    grid = "#e0e0e0"
    text = "#222222"
    last_close = closes[-1]
    sess = (session or "—").upper()
    title = f"{symbol} {timeframe} {sess} close={last_close:.2f}"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg, dpi=120)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    ax.tick_params(colors=text, labelsize=9)
    ax.grid(True, color=grid, alpha=0.8, linewidth=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
    ax.set_xlabel("Bars (oldest → newest)", color=text, fontsize=8)
    ax.set_ylabel("Price (USD)", color=text, fontsize=9)

    xs = list(range(len(closes)))
    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        col = "#1b8f3a" if c >= o else "#c62828"
        ax.plot([i, i], [l, h], color=col, linewidth=1.0, zorder=2, solid_capstyle="round")
        body = abs(c - o) or max((h - l) * 0.15, last_close * 0.00002)
        ax.bar(i, body, bottom=min(o, c), color=col, width=0.65, alpha=0.92, zorder=3)

    ymin = min(lows)
    ymax = max(highs)
    pad = max((ymax - ymin) * 0.08, last_close * 0.0005)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.8, len(closes) - 0.2)

    ax.set_title(title, color=text, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.canvas.draw()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    data = buf.getvalue()
    if not chart_png_is_valid(data):
        logger.warning(
            "[gemini-gold] chart render produced invalid/flat PNG tf=%s bytes=%s",
            timeframe,
            len(data) if data else 0,
        )
        return None
    return data
