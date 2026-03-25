"""
screenshot_card.py — Renders a PNG leaderboard card that looks like the actual
TradeHub Markets website (light theme: white cards, #f7f9ff background, indigo accent).

Uses cairosvg to convert SVG → PNG. No headless browser required.
"""
from __future__ import annotations
import html as _html_mod
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CARD_W = 1200
CARD_H = 630

# ── Website design tokens (copied exactly from strategy_portal.html) ────────
BG        = "#f7f9ff"   # page background
SURFACE   = "#f0f2fb"   # surface / card2
WHITE     = "#ffffff"   # card background
BORDER    = "#e2e5f0"   # border
BORDER2   = "#c8ccde"   # border2
TEXT      = "#111827"   # primary text
MUTED     = "#6b7280"   # muted text
ACCENT    = "#3b5bdb"   # indigo accent (blue)
ACCENT2   = "#6366f1"   # accent2
GREEN     = "#16a34a"   # green (positive / live)
GREEN_L   = "#22c55e"   # lighter green (fills)
RED       = "#dc2626"   # red (negative)
YELLOW    = "#d97706"   # yellow / warning
PAPER_C   = "#0891b2"   # paper mode blue


def _esc(s: object) -> str:
    return _html_mod.escape(str(s))


def _pnl_color(v: float) -> str:
    return GREEN if v >= 0 else RED


def _fmt_pnl(v: float) -> str:
    sign = "+" if v >= 0 else ""
    # Large numbers → compact: 1701.5% → "+1701%"
    if abs(v) >= 1000:
        return f"{sign}{v:.0f}%"
    return f"{sign}{v:.1f}%"


def _tag_color(tag: str) -> str:
    return GREEN if "+" in tag else RED


def _tag_bg(color: str) -> str:
    # Semi-transparent fill for tags
    if color == GREEN:
        return "rgba(34,197,94,.10)"
    return "rgba(239,68,68,.10)"


# ─────────────────────────────────────────────────────────────────────────────
# SVG builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_svg(strategies: List[Dict]) -> bytes:
    lines: List[str] = []

    def L(s: str) -> None:
        lines.append(s)

    top = strategies[0] if strategies else {}
    top_pnl  = float(top.get("total_pnl", 0))
    top_wr   = float(top.get("win_rate",  0))
    n_strats = len(strategies)

    # ── SVG root ──────────────────────────────────────────────────────────────
    L(f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_W}" height="{CARD_H}"'
      f' viewBox="0 0 {CARD_W} {CARD_H}">')

    # ── Defs ──────────────────────────────────────────────────────────────────
    L('<defs>')
    L(f'  <linearGradient id="gAccent" x1="0%" y1="0%" x2="100%" y2="0%">'
      f'<stop offset="0%" stop-color="{ACCENT}"/>'
      f'<stop offset="100%" stop-color="{ACCENT2}"/>'
      f'</linearGradient>')
    L(f'  <linearGradient id="gGreen" x1="0%" y1="0%" x2="0%" y2="100%">'
      f'<stop offset="0%" stop-color="#15803d"/>'
      f'<stop offset="100%" stop-color="{GREEN_L}"/>'
      f'</linearGradient>')
    L('</defs>')

    # ── Page background ───────────────────────────────────────────────────────
    L(f'<rect width="{CARD_W}" height="{CARD_H}" fill="{BG}"/>')

    # ── Top accent bar ────────────────────────────────────────────────────────
    L(f'<rect x="0" y="0" width="{CARD_W}" height="4" fill="url(#gAccent)"/>')

    # ── White topbar (y=4, h=58) ──────────────────────────────────────────────
    L(f'<rect x="0" y="4" width="{CARD_W}" height="58" fill="{WHITE}"/>')
    L(f'<line x1="0" y1="62" x2="{CARD_W}" y2="62" stroke="{BORDER}" stroke-width="1"/>')

    # Logo mark (32×32 indigo rounded square)
    L(f'<rect x="20" y="16" width="32" height="32" rx="8" fill="{ACCENT}"/>')
    L(f'<text x="36" y="37" font-family="DejaVu Sans,Arial,sans-serif" font-size="11"'
      f' font-weight="bold" fill="white" text-anchor="middle">TH</text>')
    # Logo wordmark
    L(f'<text x="60" y="32" font-family="DejaVu Sans,Arial,sans-serif" font-size="16"'
      f' font-weight="bold" fill="{TEXT}">TradeHub</text>')
    L(f'<text x="60" y="47" font-family="DejaVu Sans,Arial,sans-serif" font-size="10"'
      f' fill="{MUTED}">Strategy Builder &amp; Marketplace</text>')

    # Right: header label + PRO badge
    label_x = CARD_W - 20
    L(f'<text x="{label_x}" y="29" font-family="DejaVu Sans,Arial,sans-serif"'
      f' font-size="13" fill="{MUTED}" text-anchor="end">Strategy Leaderboard</text>')
    # PRO badge
    L(f'<rect x="{CARD_W - 58}" y="34" width="38" height="20" rx="4"'
      f' fill="{ACCENT}" />')
    L(f'<text x="{CARD_W - 39}" y="48" font-family="DejaVu Sans,Arial,sans-serif"'
      f' font-size="11" font-weight="bold" fill="white" text-anchor="middle">PRO</text>')

    # ── Summary row (y=62, h=52) ──────────────────────────────────────────────
    summary_y = 62
    summary_h = 52
    L(f'<rect x="0" y="{summary_y}" width="{CARD_W}" height="{summary_h}" fill="{SURFACE}"/>')
    L(f'<line x1="0" y1="{summary_y + summary_h}" x2="{CARD_W}"'
      f' y2="{summary_y + summary_h}" stroke="{BORDER}" stroke-width="1"/>')

    # 4 summary pills
    pill_items = [
        (_fmt_pnl(top_pnl), "TOP P&L",       _pnl_color(top_pnl)),
        (f"{top_wr:.1f}%",   "WIN RATE",       TEXT),
        (str(n_strats),      "STRATEGIES",     ACCENT),
        ("tradehubmarkets.com", "BUILD FREE",  ACCENT),
    ]
    pill_w   = 270
    pill_gap = 20
    total_pill_w = pill_w * 4 + pill_gap * 3
    pill_start_x = (CARD_W - total_pill_w) // 2

    for i, (val, lbl, color) in enumerate(pill_items):
        px = pill_start_x + i * (pill_w + pill_gap)
        py = summary_y + 8
        ph = summary_h - 16
        L(f'<rect x="{px}" y="{py}" width="{pill_w}" height="{ph}"'
          f' rx="6" fill="{WHITE}" stroke="{BORDER}" stroke-width="1"/>')
        font_sz = 15 if i < 3 else 12
        L(f'<text x="{px + pill_w // 2}" y="{py + 20}" font-family="DejaVu Sans,Arial,sans-serif"'
          f' font-size="{font_sz}" font-weight="bold" fill="{color}" text-anchor="middle">'
          f'{_esc(val)}</text>')
        L(f'<text x="{px + pill_w // 2}" y="{py + 32}" font-family="DejaVu Sans,Arial,sans-serif"'
          f' font-size="9" fill="{MUTED}" text-anchor="middle">{_esc(lbl)}</text>')

    # ── 3 strategy cards (y=122, h to footer) ────────────────────────────────
    CARDS_Y    = summary_y + summary_h + 12
    FOOTER_Y   = CARD_H - 44
    CARDS_H    = FOOTER_Y - CARDS_Y
    MARGIN     = 20
    CARD_GAP   = 12
    n_cards    = min(len(strategies), 3)
    total_w    = CARD_W - 2 * MARGIN
    cw         = (total_w - CARD_GAP * (n_cards - 1)) // n_cards

    for ci, s in enumerate(strategies[:3]):
        cx = MARGIN + ci * (cw + CARD_GAP)
        cy = CARDS_Y

        name   = _esc(s.get("name", "—")[:24])
        dirn   = str(s.get("direction", ""))
        lev    = s.get("leverage", "")
        tp     = s.get("tp_pct", "")
        sl     = s.get("sl_pct", "")
        trades = int(s.get("total_trades", 0))
        wr     = float(s.get("win_rate", 0))
        pnl    = float(s.get("total_pnl", 0))
        recent = s.get("recent_tags", [])[:5]

        # Card shadow (simulate with slightly larger gray rect)
        L(f'<rect x="{cx+1}" y="{cy+2}" width="{cw}" height="{CARDS_H}"'
          f' rx="10" fill="rgba(17,24,39,.06)"/>')
        # Card white background
        L(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{CARDS_H}"'
          f' rx="10" fill="{WHITE}" stroke="{BORDER}" stroke-width="1"/>')
        # Left accent bar (green for top, accent for others)
        bar_color = "url(#gGreen)" if ci == 0 else ACCENT
        L(f'<rect x="{cx}" y="{cy + 10}" width="3" height="{CARDS_H - 20}"'
          f' rx="2" fill="{bar_color}"/>')

        iy = cy + 16   # inner y cursor, accounting for accent bar
        ix = cx + 14   # inner x (after accent bar)
        iw = cw - 18   # usable inner width

        # ── Card header: dot + name + status pill ─────────────────────────────
        # Status dot (green = live, indigo = active)
        dot_color = GREEN_L if ci == 0 else ACCENT
        L(f'<circle cx="{ix + 4}" cy="{iy + 7}" r="3" fill="{dot_color}"/>')

        # Name
        L(f'<text x="{ix + 12}" y="{iy + 13}" font-family="DejaVu Sans,Arial,sans-serif"'
          f' font-size="14" font-weight="bold" fill="{TEXT}">{name}</text>')

        # ACTIVE pill (right side)
        pill_label = "ACTIVE"
        pill_bg    = "rgba(34,197,94,.08)"
        pill_col   = GREEN
        pill_bdr   = "rgba(34,197,94,.2)"
        pill_right = cx + cw - 12
        pill_pw    = 52
        pill_py    = iy
        L(f'<rect x="{pill_right - pill_pw}" y="{pill_py}" width="{pill_pw}" height="18"'
          f' rx="4" fill="{pill_bg}" stroke="{pill_bdr}" stroke-width="1"/>')
        L(f'<text x="{pill_right - pill_pw // 2}" y="{pill_py + 12}"'
          f' font-family="DejaVu Sans,Arial,sans-serif" font-size="9" font-weight="bold"'
          f' fill="{pill_col}" text-anchor="middle">{pill_label}</text>')

        iy += 22

        # ── Meta line: direction · leverage · TP · SL ────────────────────────
        meta_parts = []
        if dirn:
            meta_parts.append(dirn)
        if lev:
            meta_parts.append(f"{lev}× lev")
        if tp:
            meta_parts.append(f"TP {tp}%")
        if sl:
            meta_parts.append(f"SL {sl}%")
        if not meta_parts:
            meta_parts = ["Automated strategy"]
        meta_str = " · ".join(meta_parts)[:44]
        L(f'<text x="{ix}" y="{iy + 11}" font-family="DejaVu Sans,Arial,sans-serif"'
          f' font-size="11" fill="{MUTED}">{_esc(meta_str)}</text>')

        iy += 18

        # ── Metrics grid (4 cells, light gray bg with 1px gap border) ─────────
        grid_h = 66
        grid_y = iy
        # Grid background (#e2e5f0 = border color, creates "1px gap" effect)
        L(f'<rect x="{ix}" y="{grid_y}" width="{iw}" height="{grid_h}"'
          f' rx="6" fill="{BORDER}"/>')

        metrics = [
            (str(trades),      "TRADES",    TEXT),
            (f"{wr:.1f}%",     "WIN RATE",  GREEN if wr >= 50 else RED),
            (_fmt_pnl(pnl),    "TOTAL P&L", _pnl_color(pnl)),
            ("0",              "OPEN",      TEXT),
        ]
        cell_w = (iw - 3) // 4  # 3 gaps of 1px
        for mi, (mv, ml, mc) in enumerate(metrics):
            cx2 = ix + mi * (cell_w + 1)
            cy2 = grid_y + 1
            ch2 = grid_h - 2
            is_last = mi == 3
            radius_str = ""
            if mi == 0:
                radius_str = 'rx="5"'
            elif mi == 3:
                radius_str = 'rx="5"'
            L(f'<rect x="{cx2}" y="{cy2}" width="{cell_w}" height="{ch2}"'
              f' {radius_str} fill="{SURFACE}"/>')
            # Value
            mv_font = 17 if len(str(mv)) <= 7 else 13
            L(f'<text x="{cx2 + cell_w // 2}" y="{cy2 + 28}" font-family="DejaVu Sans,Arial,sans-serif"'
              f' font-size="{mv_font}" font-weight="bold" fill="{mc}" text-anchor="middle">'
              f'{_esc(mv)}</text>')
            # Label
            L(f'<text x="{cx2 + cell_w // 2}" y="{cy2 + 42}" font-family="DejaVu Sans,Arial,sans-serif"'
              f' font-size="8" fill="{MUTED}" text-anchor="middle">{_esc(ml)}</text>')
            # Win rate bar under WIN RATE cell
            if mi == 1:
                bar_track_y = cy2 + ch2 - 8
                bar_fill_w  = int((cell_w - 12) * min(wr, 100) / 100)
                L(f'<rect x="{cx2 + 6}" y="{bar_track_y}" width="{cell_w - 12}" height="3"'
                  f' rx="2" fill="{BORDER2}"/>')
                if bar_fill_w > 0:
                    fill_c = GREEN_L if wr >= 40 else RED
                    L(f'<rect x="{cx2 + 6}" y="{bar_track_y}" width="{bar_fill_w}" height="3"'
                      f' rx="2" fill="{fill_c}"/>')

        iy += grid_h + 10

        # ── Recent mini-trade tags ─────────────────────────────────────────────
        if recent:
            tag_x = ix
            for tag in recent:
                tc     = _tag_color(tag)
                tbg    = _tag_bg(tc)
                label  = _esc(tag[:14])
                tw     = max(len(tag) * 7 + 10, 50)
                if tag_x + tw > cx + cw - 12:
                    break  # don't overflow
                L(f'<rect x="{tag_x}" y="{iy}" width="{tw}" height="20"'
                  f' rx="4" fill="{tbg}"/>')
                L(f'<text x="{tag_x + tw // 2}" y="{iy + 13}"'
                  f' font-family="DejaVu Sans,Arial,sans-serif" font-size="10" font-weight="bold"'
                  f' fill="{tc}" text-anchor="middle">{label}</text>')
                tag_x += tw + 5

    # ── Footer ────────────────────────────────────────────────────────────────
    L(f'<rect x="0" y="{FOOTER_Y}" width="{CARD_W}" height="{CARD_H - FOOTER_Y}"'
      f' fill="{WHITE}"/>')
    L(f'<line x1="0" y1="{FOOTER_Y}" x2="{CARD_W}" y2="{FOOTER_Y}"'
      f' stroke="{BORDER}" stroke-width="1"/>')
    L(f'<text x="20" y="{FOOTER_Y + 27}" font-family="DejaVu Sans,Arial,sans-serif"'
      f' font-size="13" font-weight="bold" fill="{ACCENT}">tradehubmarkets.com</text>')
    L(f'<text x="{CARD_W - 20}" y="{FOOTER_Y + 27}" font-family="DejaVu Sans,Arial,sans-serif"'
      f' font-size="12" fill="{MUTED}" text-anchor="end">'
      f'Build, test &amp; automate your strategy — free</text>')
    # Right bottom accent dot
    L(f'<circle cx="{CARD_W - 18}" cy="{FOOTER_Y + 12}" r="4" fill="{ACCENT}"/>')

    L('</svg>')
    return "\n".join(lines).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def screenshot_leaderboard_card_sync(strategies: List[Dict]) -> Optional[bytes]:
    """Render the SVG leaderboard card to PNG bytes using cairosvg."""
    try:
        import cairosvg

        svg_bytes = _build_svg(strategies)
        png_bytes = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=CARD_W,
            output_height=CARD_H,
        )
        logger.info(f"[ScreenshotCard] cairosvg render OK — {len(png_bytes):,} bytes")
        return png_bytes

    except Exception as e:
        logger.error(f"[ScreenshotCard] cairosvg render failed: {e}", exc_info=True)
        return None


async def screenshot_leaderboard_card(strategies: List[Dict]) -> Optional[bytes]:
    """Async wrapper (runs sync cairosvg in a thread)."""
    import asyncio
    import concurrent.futures

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, screenshot_leaderboard_card_sync, strategies
        )
