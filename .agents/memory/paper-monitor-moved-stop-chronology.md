---
name: Paper monitor moved-stop chronology
description: Why the paper sweep must not retro-apply a moved stop to earlier candles, and how the sleff guard + live-spot candle work.
---

# Paper-trade moved-stop chronology guard

The paper monitor (`_evaluate_paper_position_against_candles`) re-scans candles
from entry on EVERY sweep. It mutates and persists `ex.sl_price` in place (auto
breakeven → entry, partial-close → entry, trailing → tighter). So on the next
sweep the persisted tightened stop is tested against EARLIER candles that
predate the move → a pre-activation dip to entry is read as a stop-out and a
winner is closed flat. Confirmed on a real gold exec (BE trigger first reached
late, price never returned to entry after, yet closed BREAKEVEN minutes later).

**Rule:** a moved stop may only apply to price action AT/AFTER the move.

**How it's enforced:** `sl_eff_ms` = timestamp the CURRENT stop level became
effective. Set `sl_eff_ms = _ts + 1` wherever the stop is tightened (BE, partial,
trailing) — `+1ms` excludes the move candle (no same-candle look-ahead) and
includes the next. Per candle, `sl_hit &= not (sl_eff_ms is not None and _ts <
sl_eff_ms)`. Persisted as a `sleff=<ms>` token in `notes`.

**Why it's provably safe:** an OPEN trade never hit its looser earlier stop, so
skipping the stop-test on pre-move candles can never miss a real loss; and it
can never invent a false close. Fixes breakeven, trailing and partial-to-entry
uniformly with one timestamp (the timestamp always tracks the LAST tightening,
which is the tightest level, so candles after it are correctly tested against it).

**Notes is a shared junk-drawer** (`pos=`, `acct=`, `partial_close_pct`,
`partial_close_done`, `Live→Paper`/fallback text, the live-pnl suffix). The
sweep's notes rewrite must PRESERVE the metadata base (everything before
" | open"/" | unrealised") — the old `else: ex.notes = pnl_note` branch silently
wiped `partial_close_done` (latent bug, now fixed). When rewriting: strip the
old `sleff=` token, re-add the current one, re-add `partial_close_done`, keep the
base. Stamp `sleff` in the partial-close mid-loop commit too for crash durability.

**Latency fix (same change):** paper SL/TP was only seen on 1m candle CLOSE (+
slow yfinance fallback) → ~9 min late. `_fetch_for_bucket` now appends a
synthetic `[now_ms, px,px,px,px]` candle for forex/index from
`tradfi_prices.get_price` (the broker spot feed the live path uses; returns None
in dev → synthetic skipped, consistent with prod-only executor). Flat o=h=l=c
can't trip the combined TP+SL block, drives BE/trailing like a tick, and as
`relevant[-1]` makes the unrealised-pnl note reflect the freshest price.

Executor is prod-only/disabled in dev → user must REPUBLISH for prod to pick it up.
