"""
Tweet card generator — creates professional 1200x675 PNG cards for Twitter posts.
Uses Pillow (PIL) with DejaVu fonts. No external dependencies beyond Pillow.

Brand palette mirrors tradehubmarkets.com:
  accent   #3b5bdb  (indigo)
  accent2  #6366f1  (purple-indigo)
  green    #16a34a
  red      #dc2626
  yellow   #f59e0b
"""

from __future__ import annotations
import io
import math
import random
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

FONT_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

W, H = 1200, 675

# ── palette ──────────────────────────────────────────────────────────────────
BG          = (10,  13,  26)          # dark navy
CARD        = (18,  22,  42)          # card surface
CARD2       = (26,  31,  56)          # secondary card
ACCENT      = (59,  91, 219)          # #3b5bdb indigo
ACCENT2     = (99, 102, 241)          # #6366f1
GREEN       = (22, 163,  74)          # #16a34a
RED         = (220,  38,  38)         # #dc2626
YELLOW      = (245, 158,  11)         # #f59e0b
WHITE       = (255, 255, 255)
OFF_WHITE   = (226, 232, 240)
MUTED       = (100, 116, 139)
BORDER      = (40,  50,  80)


# ── font helpers ──────────────────────────────────────────────────────────────
def _f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)


def _text_w(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def _centered_x(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    return (W - _text_w(draw, text, font)) // 2


# ── drawing primitives ────────────────────────────────────────────────────────
def _draw_rect(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    fill=None, outline=None, radius: int = 12,
):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=outline)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    cx: int, cy: int, text: str,
    fill: Tuple, text_color: Tuple = WHITE,
    font_size: int = 28, pad_x: int = 20, pad_y: int = 10,
) -> Tuple[int, int, int, int]:
    """Returns bounding box (x1, y1, x2, y2)."""
    font = _f(font_size, bold=True)
    tw = _text_w(draw, text, font)
    bw = tw + pad_x * 2
    bh = font_size + pad_y * 2
    x1, y1 = cx - bw // 2, cy - bh // 2
    _draw_rect(draw, x1, y1, bw, bh, fill=fill, radius=8)
    draw.text((x1 + pad_x, y1 + pad_y), text, font=font, fill=text_color)
    return x1, y1, x1 + bw, y1 + bh


def _gradient_bg(img: Image.Image):
    """Vertical gradient overlay from BG to a slightly lighter shade."""
    pix = img.load()
    for y in range(H):
        t = y / H
        r = int(BG[0] + (CARD[0] - BG[0]) * t * 0.5)
        g = int(BG[1] + (CARD[1] - BG[1]) * t * 0.5)
        b = int(BG[2] + (CARD[2] - BG[2]) * t * 0.5)
        for x in range(W):
            pix[x, y] = (r, g, b)


def _accent_line(draw: ImageDraw.ImageDraw, y: int, color=ACCENT, thickness: int = 4):
    """Full-width horizontal accent line."""
    draw.rectangle([0, y, W, y + thickness], fill=color)


def _dot_grid(draw: ImageDraw.ImageDraw, color=(ACCENT[0], ACCENT[1], ACCENT[2], 15)):
    """Subtle dot pattern in background."""
    for x in range(0, W, 40):
        for y in range(0, H, 40):
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)


def _sparkline(
    draw: ImageDraw.ImageDraw,
    values: list,
    x: int, y: int, w: int, h: int,
    color=GREEN, line_width: int = 3,
):
    if len(values) < 2:
        return
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    pts = []
    for i, v in enumerate(values):
        px = x + int(i / (len(values) - 1) * w)
        py = y + h - int((v - mn) / rng * h)
        pts.append((px, py))
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=color, width=line_width)


def _header(draw: ImageDraw.ImageDraw, tag: str = "TRADEHUB MARKETS"):
    """Top header bar with branding."""
    _draw_rect(draw, 0, 0, W, 60, fill=CARD, radius=0)
    draw.rectangle([0, 60, W, 62], fill=ACCENT)

    font = _f(20, bold=True)
    draw.text((28, 20), "⬡ TRADEHUB MARKETS", font=font, fill=WHITE)

    tag_font = _f(17)
    tag_w = _text_w(draw, tag, tag_font)
    draw.text((W - tag_w - 28, 22), tag, font=tag_font, fill=(ACCENT2[0], ACCENT2[1], ACCENT2[2]))

    url_font = _f(15)
    url = "tradehubmarkets.com"
    url_w = _text_w(draw, url, url_font)
    # right side hint already has the tag; put URL in footer instead


def _footer(draw: ImageDraw.ImageDraw, label: str = "Build & automate strategies free"):
    """Bottom footer bar."""
    draw.rectangle([0, H - 54, W, H], fill=CARD)
    draw.rectangle([0, H - 56, W, H - 54], fill=ACCENT)

    f_url  = _f(18, bold=True)
    f_text = _f(16)
    url = "tradehubmarkets.com"
    draw.text((28, H - 36), url, font=f_url, fill=(ACCENT2[0], ACCENT2[1], ACCENT2[2]))
    label_x = 28 + _text_w(draw, url, f_url) + 18
    draw.text((label_x, H - 34), f"— {label}", font=f_text, fill=MUTED)


def _stat_block(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    label: str, value: str, value_color=WHITE,
):
    """Small label/value stat block inside a card."""
    _draw_rect(draw, x, y, w, h, fill=CARD2, radius=10)
    lf = _f(15)
    vf = _f(22, bold=True)
    lw = _text_w(draw, label, lf)
    vw = _text_w(draw, value, vf)
    draw.text((x + (w - lw) // 2, y + 10), label, font=lf, fill=MUTED)
    draw.text((x + (w - vw) // 2, y + 32), value, font=vf, fill=value_color)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CARD BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def make_gainer_card(
    symbol: str,
    price: float,
    change_pct: float,
    volume_usd: float,
    market_cap: float = 0,
    rank: int = 0,
    extra_label: str = "",
    sparkline_values: list = None,
) -> bytes:
    """
    Card for featured_coin / early_gainer posts.
    Returns PNG bytes ready to upload to Twitter.
    """
    img = Image.new("RGB", (W, H), BG)
    _gradient_bg(img)
    draw = ImageDraw.Draw(img)
    _dot_grid(draw)

    _header(draw, "TOP GAINER · 24H")

    # ── main content ──────────────────────────────────────────────
    is_up = change_pct >= 0
    sign  = "+" if is_up else ""
    chg_color = GREEN if is_up else RED

    # Rank badge (top left content area)
    if rank > 0:
        rank_txt = f"#{rank} TODAY"
        rf = _f(17, bold=True)
        _draw_rect(draw, 28, 78, _text_w(draw, rank_txt, rf) + 24, 34, fill=ACCENT, radius=6)
        draw.text((40, 85), rank_txt, font=rf, fill=WHITE)

    # Coin symbol — big
    sym_font = _f(96, bold=True)
    sym_text = f"${symbol}"
    sym_x = _centered_x(draw, sym_text, sym_font)
    draw.text((sym_x, 100), sym_text, font=sym_font, fill=WHITE)

    # Price
    if price < 0.01:
        price_str = f"${price:.8f}"
    elif price < 1:
        price_str = f"${price:.6f}"
    elif price < 100:
        price_str = f"${price:,.4f}"
    else:
        price_str = f"${price:,.2f}"

    pf = _f(40, bold=True)
    pw = _text_w(draw, price_str, pf)
    draw.text(((W - pw) // 2, 210), price_str, font=pf, fill=OFF_WHITE)

    # Change badge — centered
    chg_txt = f"  {sign}{change_pct:.2f}%  "
    _draw_badge(draw, W // 2, 290, chg_txt, fill=chg_color, font_size=36, pad_x=24, pad_y=14)

    # Arrow
    arrow = "▲" if is_up else "▼"
    af = _f(28, bold=True)
    draw.text((W // 2 + 130, 279), arrow, font=af, fill=chg_color)

    # Stat row
    vol_str = f"${volume_usd/1e6:.1f}M" if volume_usd < 1e9 else f"${volume_usd/1e9:.2f}B"
    mcap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else (f"${market_cap/1e6:.0f}M" if market_cap > 0 else "—")
    gap = 20
    sw = (W - 56 - gap * 2) // 3

    _stat_block(draw, 28, 340, sw, 72, "24H VOLUME", vol_str)
    _stat_block(draw, 28 + sw + gap, 340, sw, 72, "MARKET CAP", mcap_str)

    third_label = extra_label or ("SIGNAL" if is_up else "ALERT")
    third_val   = "BULLISH ↑" if is_up else "BEARISH ↓"
    third_vc    = GREEN if is_up else RED
    _stat_block(draw, 28 + (sw + gap) * 2, 340, sw, 72, third_label, third_val, value_color=third_vc)

    # Sparkline
    if sparkline_values and len(sparkline_values) >= 4:
        sl_color = GREEN if is_up else RED
        _sparkline(draw, sparkline_values, 28, 430, W - 56, 90, color=sl_color, line_width=3)
        # Shade under sparkline
        mn = min(sparkline_values)
        mx = max(sparkline_values)
        rng = mx - mn or 1
        pts = []
        for i, v in enumerate(sparkline_values):
            px = 28 + int(i / (len(sparkline_values) - 1) * (W - 56))
            py = 430 + 90 - int((v - mn) / rng * 90)
            pts.append((px, py))
        pts.append((pts[-1][0], 520))
        pts.append((28, 520))
        draw.polygon(pts, fill=(sl_color[0], sl_color[1], sl_color[2], 30))
    else:
        # Decorative divider if no sparkline
        draw.rectangle([28, 440, W - 28, 442], fill=BORDER)
        hint_f = _f(20)
        hint = "strategy running on this setup"
        hw = _text_w(draw, hint, hint_f)
        draw.text(((W - hw) // 2, 455), hint, font=hint_f, fill=MUTED)

    _footer(draw, "automate moves like this free")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_promo_card(
    feature_headline: str = "Build Your Strategy",
    sub: str = "No code. No guesswork. Just results.",
    stats: list = None,           # list of (label, value) tuples
    cta: str = "Free to start — tradehubmarkets.com",
) -> bytes:
    """
    Promotional card for tradehub_promo posts.
    Mimics the website's feature showcase layout.
    """
    img = Image.new("RGB", (W, H), BG)
    _gradient_bg(img)
    draw = ImageDraw.Draw(img)
    _dot_grid(draw)

    _header(draw, "STRATEGY PLATFORM")

    # Top accent stripe
    _accent_line(draw, 62, color=ACCENT, thickness=4)

    # Headline
    hf = _f(64, bold=True)
    hw = _text_w(draw, feature_headline, hf)
    draw.text(((W - hw) // 2, 90), feature_headline, font=hf, fill=WHITE)

    # Sub-headline
    sf = _f(26)
    sw_ = _text_w(draw, sub, sf)
    draw.text(((W - sw_) // 2, 172), sub, font=sf, fill=MUTED)

    # Feature pills
    features = [
        "🧠  AI Strategy Builder",
        "📊  Backtester  (30d / 90d)",
        "🏆  Strategy Leaderboard",
        "⚡  Live Bitunix Automation",
        "🛒  Strategy Marketplace",
        "📈  Paper Trading  (Free)",
    ]
    pill_f = _f(20, bold=True)
    px, py = 60, 225
    col_w = (W - 120) // 2
    for i, feat in enumerate(features):
        col = i % 2
        row = i // 2
        fx = px + col * (col_w + 10)
        fy = py + row * 68
        pw_ = _text_w(draw, feat, pill_f) + 32
        _draw_rect(draw, fx, fy, pw_, 46, fill=CARD2, radius=10)
        draw.rectangle([fx, fy, fx + 5, fy + 46], fill=ACCENT)
        draw.text((fx + 18, fy + 13), feat, font=pill_f, fill=OFF_WHITE)

    # Stats row (if provided)
    if stats:
        gap = 20
        n = len(stats)
        bw = (W - 56 - gap * (n - 1)) // n
        for i, (lbl, val) in enumerate(stats):
            _stat_block(draw, 28 + i * (bw + gap), 440, bw, 72, lbl, val, value_color=ACCENT2)

    # CTA
    cta_f = _f(22, bold=True)
    cta_w = _text_w(draw, cta, cta_f)
    draw.text(((W - cta_w) // 2, H - 96), cta, font=cta_f, fill=ACCENT2)

    _footer(draw, "strategy leaderboard live now")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_memecoin_card(
    symbol: str,
    price: float,
    change_pct: float,
    volume_usd: float,
    tagline: str = "",
) -> bytes:
    """Vibrant card for memecoin posts."""
    is_up = change_pct >= 0
    bg_color = (12, 6, 26) if is_up else (26, 6, 10)

    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)

    # Animated feel — concentric faint rings
    for r in range(50, 350, 60):
        alpha = max(10, 40 - r // 10)
        col = (ACCENT[0], ACCENT[1], ACCENT[2]) if is_up else (RED[0], RED[1], RED[2])
        draw.ellipse(
            [W // 2 - r, H // 2 - r, W // 2 + r, H // 2 + r],
            outline=(*col, alpha),
            width=2,
        )

    _header(draw, "MEME SEASON 🐸")
    _accent_line(draw, 62, color=ACCENT if is_up else RED, thickness=4)

    chg_color = GREEN if is_up else RED
    sign = "+" if is_up else ""

    # Symbol
    sym_font = _f(110, bold=True)
    sym_text = f"${symbol}"
    sym_x = _centered_x(draw, sym_text, sym_font)
    draw.text((sym_x, 88), sym_text, font=sym_font, fill=WHITE)

    # Change — huge
    chg_txt = f"{sign}{change_pct:.1f}%"
    cf = _f(72, bold=True)
    cw = _text_w(draw, chg_txt, cf)
    draw.text(((W - cw) // 2, 215), chg_txt, font=cf, fill=chg_color)

    # Price
    if price < 0.000001:
        price_str = f"${price:.10f}"
    elif price < 0.01:
        price_str = f"${price:.8f}"
    elif price < 1:
        price_str = f"${price:.6f}"
    else:
        price_str = f"${price:,.4f}"

    pf = _f(32)
    pw = _text_w(draw, price_str, pf)
    draw.text(((W - pw) // 2, 300), price_str, font=pf, fill=OFF_WHITE)

    # Volume
    vol_str = f"24h vol: ${volume_usd/1e6:.1f}M" if volume_usd < 1e9 else f"24h vol: ${volume_usd/1e9:.2f}B"
    vf = _f(24)
    vw = _text_w(draw, vol_str, vf)
    draw.text(((W - vw) // 2, 350), vol_str, font=vf, fill=MUTED)

    # Tagline
    if tagline:
        tf = _f(22)
        tw = _text_w(draw, tagline, tf)
        draw.text(((W - tw) // 2, 400), tagline, font=tf, fill=(ACCENT2[0], ACCENT2[1], ACCENT2[2]))

    # CTA box
    cta = "⚡ Automate plays like this → tradehubmarkets.com"
    ctaf = _f(20, bold=True)
    ctaw = _text_w(draw, cta, ctaf)
    box_x = (W - ctaw - 40) // 2
    _draw_rect(draw, box_x, 450, ctaw + 40, 48, fill=CARD, radius=10)
    draw.text((box_x + 20, 463), cta, font=ctaf, fill=ACCENT2)

    _footer(draw, "meme strategy builder — free")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_ta_card(
    symbol: str,
    price: float,
    change_pct: float,
    volume_usd: float,
    ta_lines: list = None,        # list of strings for TA bullets
    bias: str = "BULLISH",        # "BULLISH" | "BEARISH" | "NEUTRAL"
) -> bytes:
    """Quick-TA card — shows TA analysis bullets."""
    img = Image.new("RGB", (W, H), BG)
    _gradient_bg(img)
    draw = ImageDraw.Draw(img)
    _dot_grid(draw)

    bias_upper = bias.upper()
    bias_color = GREEN if "BULL" in bias_upper else (RED if "BEAR" in bias_upper else YELLOW)

    _header(draw, "TECHNICAL ANALYSIS")
    _accent_line(draw, 62, color=bias_color, thickness=4)

    is_up = change_pct >= 0
    sign = "+" if is_up else ""
    chg_color = GREEN if is_up else RED

    # Left panel — coin info
    panel_w = 380
    _draw_rect(draw, 28, 78, panel_w, H - 78 - 60, fill=CARD, radius=14)
    draw.rectangle([28, 78, 32, H - 60], fill=bias_color)

    sym_f = _f(56, bold=True)
    draw.text((50, 90), f"${symbol}", font=sym_f, fill=WHITE)

    pf = _f(26, bold=True)
    if price < 1:
        price_str = f"${price:.6f}"
    elif price < 100:
        price_str = f"${price:,.4f}"
    else:
        price_str = f"${price:,.2f}"
    draw.text((50, 158), price_str, font=pf, fill=OFF_WHITE)

    chg_f = _f(30, bold=True)
    draw.text((50, 198), f"{sign}{change_pct:.2f}%", font=chg_f, fill=chg_color)

    # Volume
    vol_str = f"Vol: ${volume_usd/1e6:.1f}M" if volume_usd < 1e9 else f"Vol: ${volume_usd/1e9:.2f}B"
    vf = _f(20)
    draw.text((50, 242), vol_str, font=vf, fill=MUTED)

    # Bias badge
    bias_f = _f(22, bold=True)
    bias_text = f"  {bias_upper}  "
    bw = _text_w(draw, bias_text, bias_f)
    _draw_rect(draw, 50, 280, bw + 16, 40, fill=bias_color, radius=8)
    draw.text((58, 289), bias_text, font=bias_f, fill=WHITE)

    # Divider
    draw.rectangle([50, 335, 28 + panel_w - 22, 337], fill=BORDER)

    # Strategy CTA inside left panel
    cta_lines = ["Strategy running on", "this setup →", "tradehubmarkets.com"]
    clf = _f(17)
    for i, ln in enumerate(cta_lines):
        color = (ACCENT2[0], ACCENT2[1], ACCENT2[2]) if i == 2 else MUTED
        draw.text((50, 348 + i * 26), ln, font=clf, fill=color)

    # Right panel — TA bullets
    rp_x = 28 + panel_w + 20
    rp_w = W - rp_x - 28
    _draw_rect(draw, rp_x, 78, rp_w, H - 78 - 60, fill=CARD, radius=14)

    ta_header_f = _f(20, bold=True)
    draw.text((rp_x + 20, 90), "ANALYSIS", font=ta_header_f, fill=ACCENT2)

    bullets = ta_lines or [
        f"RSI reading suggests {bias_upper.lower()} momentum",
        "Volume above 20-period average",
        "Price above 50-period MA",
        "Support held at key level",
        "MACD crossover pending",
    ]
    bf = _f(20)
    dot_f = _f(18, bold=True)
    by = 122
    for bullet in bullets[:7]:
        dot_color = bias_color
        draw.text((rp_x + 18, by), "●", font=dot_f, fill=dot_color)
        draw.text((rp_x + 38, by + 1), bullet[:62], font=bf, fill=OFF_WHITE)
        by += 36
        if by > H - 90:
            break

    _footer(draw, "automate TA-based strategies — build free")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — generate fake sparkline for when real OHLCV not available
# ─────────────────────────────────────────────────────────────────────────────
def _fake_sparkline(base: float, change_pct: float, n: int = 24) -> list:
    """Plausible price path ending at base with overall trend matching change_pct."""
    start = base / (1 + change_pct / 100)
    values = [start]
    for _ in range(n - 2):
        drift = (base - values[-1]) / (n - len(values)) * random.uniform(0.5, 1.5)
        noise = start * random.gauss(0, 0.008)
        values.append(values[-1] + drift + noise)
    values.append(base)
    return values


def make_gainer_card_auto(
    symbol: str, price: float, change_pct: float, volume_usd: float,
    market_cap: float = 0, rank: int = 0,
) -> bytes:
    """make_gainer_card with auto-generated sparkline."""
    sl = _fake_sparkline(price, change_pct)
    return make_gainer_card(
        symbol=symbol, price=price, change_pct=change_pct,
        volume_usd=volume_usd, market_cap=market_cap, rank=rank,
        sparkline_values=sl,
    )
