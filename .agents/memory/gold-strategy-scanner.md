---
name: Gold strategy discovery scanner
description: How the Claude-driven XAUUSD backtest scanner finds the most profitable strategy across timeframes/sessions; the design constraints that keep it honest.
---

# Gold strategy discovery scanner

`app/services/gold_strategy_scanner.py::run_gold_discovery(days, direction_mode)`
backtests many candidate strategies on gold and ranks by profitability across
trading sessions. Endpoint: `POST /api/backtest/gold-discovery` (pro/admin gated).

## Hard design rules (don't regress)
- **Candidates use ONLY backtest-supported signals** (`SUPPORTED_PRIMARY` = the
  set `backtest_engine.eval_condition_bt` actually evaluates). ICT `fx_*` signals
  are EXCLUDED: `eval_condition_bt` passes unknown types through as `True`
  (no-op), so including them silently inflates every result to look amazing.
  **Why:** a no-op condition is always satisfied → fake edge.
- **Claude never rebuilds config.** Claude's two jobs are: propose extra
  candidates (validated, invalid dropped) and PICK the winner (returns
  `{index, name, rationale}` only). The saved/displayed config always comes from
  the leaderboard entry we built. **Why:** LLM-emitted strategy JSON is the
  single most common silently-never-fires bug class in this codebase.
- **Sessions are bucketed post-hoc by entry_ts**, not by re-running per session.
  Each candidate is backtested once per (TF × risk); trades are then bucketed by
  the UTC session of their entry. The engine-level session gate
  (`_bt_session_active` / `forex_session` condition in `eval_condition_bt`) exists
  ONLY for faithfully replaying a *saved* session-restricted strategy — the scan
  itself does NOT add or use it. So there is no double-gating / boundary mismatch
  inside the scan. (`run_backtest` trades carry `entry_ts` + `pip_move` in pip-mode.)
- **Scoring is credibility-weighted with a `MIN_TRADES` floor.** Composite score
  blends profit factor + total pips + win rate + return − drawdown, scaled by
  `min(1, trades/25)`, and buckets under `MIN_TRADES` are dropped. **Why:** raw PF
  on 3 trades will always top an honest 30-trade strategy otherwise.
- **`coverage_days` reports ACTUAL fetched history per TF.** The gold data source
  (`tradfi_prices.get_klines`) caps depth (~15m: 1–2 months, 1h: 4–5 months), so
  a 90/180-day request is NOT fully covered on 15m. Report real coverage rather
  than implying the full window. Forex backtests run leverage=1 (app FX convention).
- **Leaderboard dedupes by `(label, direction)` GLOBALLY** → diverse ideas, each
  carrying its own best session/TF/RR, not 10 permutations of one idea. An idea
  appears ONCE on the whole board even though it's tested in both scalp and swing
  risk profiles (see scalp quota below) — never list the same idea twice.

## Scalp vs swing risk variants (don't regress)
- `RISK_VARIANTS` is labelled `(sl_pips, tp_pips, style)` 3-tuples — 3 `scalp`
  (~100-pip TP: 50/100, 60/150, 40/120) + 3 `swing` (200/400, 250/500, 150/450).
  Each result row carries its `style`. **Why:** users want fast scalps on the
  board, not only swing runners.
- **Swings out-score scalps on raw pips**, so a single score-ranked board hides
  every scalp. The board therefore RESERVES ~half its slots for scalps: take best
  scalp-per-idea up to `LEADERBOARD_SIZE//2`, then fill remainder with best
  swing-per-idea, then backfill leftover scalps — all sharing ONE `used_ideas`
  set so no idea appears as both scalp and swing. **Why:** without the quota the
  scan looks like it ignored the scalp request.
- **Backtest budget is bounded by the 180s inline gunicorn timeout.** Keep
  `MAX_CANDIDATES × len(TIMEFRAMES) × len(RISK_VARIANTS)` ≈ 320 (proven safe; dev
  full run ~82–91s, but Claude propose+pick latency adds variance). Adding a risk
  variant or timeframe means lowering `MAX_CANDIDATES` in lockstep — a run that
  exceeds 180s is killed mid-request.

## Stop management (breakeven / trailing) — backtest↔live fidelity (don't regress)
- `RISK_VARIANTS` are `(sl, tp, style, mgmt)` 4-tuples; `mgmt` is one of
  `fixed` / `breakeven` / `trail`. `_eval_candidate` emits the engine keys
  `breakeven_at_pct`, `trailing_stop`, `trailing_stop_pct` (the last derived from
  `sl_pips × pip_sz / ref_close`); result rows + `_build_prompt_for` /
  `_build_name_for` carry the management so the built strategy replays as scanned.
- **TP targets are realistic** (40–200 pip swings, ~100-pip scalps) — no 500-pip
  "blue moon" targets. **Why:** unsustainable targets backtest as rare giant wins
  but never fill live.
- **`run_backtest` models management itself** (default OFF): top-level
  `breakeven_at_pct` (% of entry→TP distance → SL ratchets to entry),
  `trailing_stop` + `trailing_stop_pct` (% of price). The stop is moved AFTER each
  bar's exit checks (next-bar only — **no lookahead**). Mirrors executor semantics
  (`_compute_be_trigger_price`).
- **SL-side exits are three-way** by realised price move vs entry with tick-level
  eps (`entry × 1e-7`): WIN if stop ratcheted beyond entry, **BREAKEVEN** at
  entry, LOSS below. Applied in BOTH the normal-SL and SL-gap blocks. Mirrors the
  shared `strategy_executor._classify_sl_outcome` so test == live.
- **BREAKEVEN is its own outcome everywhere** (`_compute_stats`, scanner
  `_bucket_stats`, the pip-mode aggregate in `run_backtest`): it COUNTS as a
  closed trade for equity / pnl / drawdown / total_pips but is **neither win nor
  loss** → `win_rate = wins / (wins + losses)`. **Why:** folding a scratch into
  WIN inflated the displayed win-rate. For strategies WITHOUT management (the
  default) zero breakevens are produced, so every metric is bit-identical to the
  pre-management behavior (parity-tested). Don't fold BREAKEVEN back into WIN/LOSS.

## UI
Gold-finder button + `modal-gold` in `strategy_portal.html`. Leaderboard has a
Style column (⚡ scalp / 🌊 swing). Two build paths:
- "Build this in AI Builder" prefills the chat builder (forex market) with the
  winner's NL `build_prompt` — reuse the proven compiler, don't auto-save.
- "⚡ Build all (paper)" (`goldBuildAll`): per leaderboard entry POST
  `/api/build-strategy` (compile NL `build_prompt`) → force
  `asset_class='forex'`, `universe={specific, [XAUUSD]}`, `_build_mode='paper'`,
  `name=build_name` → POST `/api/save-strategy`. 3 concurrent workers, confirm
  first, saves as PAPER only (nothing auto-goes-live). Every leaderboard entry
  therefore carries `build_prompt` + `build_name` (`_build_name_for`), not just
  the winner. Reuses the compiler — never hand-builds config (same rule as above).
