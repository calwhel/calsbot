---
name: Breakeven (scratch) outcome classification
description: A stop hit at entry must be classified BREAKEVEN, not LOSS — and the check must read persisted sl_price vs entry, never a per-cycle activation flag.
---

# Breakeven (scratch) outcome classification

When a stop-loss is hit at (essentially) the entry price the trade is a
**BREAKEVEN/scratch**, never a LOSS. This happens after auto-breakeven (or
partial-close-to-entry) moves the stop to entry and price later returns to it.

## The trap: per-cycle activation flags lie across monitor cycles

The paper monitor (`_evaluate_paper_position_against_candles` in
`strategy_executor.py`) re-scans candles from entry every cycle. Breakeven
classification must NOT depend on a per-cycle local like `be_activated`, because:
- BE activation only sets `be_activated=True` inside the block guarded by
  `if not be_activated and ex.sl_price != ex.entry_price`.
- Once BE is persisted (sl_price already == entry from a PRIOR cycle), that
  block is skipped, so `be_activated` stays False on the cycle that finally hits
  the stop → the scratch exit gets mislabeled LOSS (the reported gold bug:
  entry==exit==4446.6, +0.0 pips, shown as "STRATEGY LOSS / SL hit").

**Fix:** classify off the persisted state — `outcome = BREAKEVEN` when
`abs(sl_price - entry_price)` is tick-level, else LOSS.

## Tolerance must be tick-level, not a wide band

BE (and partial-close-to-entry) set `ex.sl_price = ex.entry_price` EXACTLY
(identical Python float, bit-stable across the Neon double round-trip). A genuine
stop is always ≥ ~1 pip away (smallest pip ≈ 2e-5 relative for gold, ≈6e-5+ for
FX pairs). So use a tight epsilon (`<= entry*1e-7`). A wide band (e.g.
`entry*1e-4` ≈ 4 pips on gold) mislabels small genuine losses as BREAKEVEN —
this exact regression was caught in review.

**Why:** breakeven is excluded from the win-rate denominator, so mislabeling
distorts both the user's loss count and their displayed win rate.

## Live forex path already correct

The live cTrader reconcile path classifies via `w["be_moved"]` (persisted notes
flag) + near-entry tolerance — robust, needs no change. Only the paper path had
the per-cycle-flag bug.
