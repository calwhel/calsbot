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
- **Leaderboard dedupes by `(label, direction)`** → diverse ideas, each carrying
  its own best session/TF/RR, not 10 permutations of one idea.

## UI
Gold-finder button + `modal-gold` in `strategy_portal.html`. "Build this in AI
Builder" prefills the chat builder (forex market) with the winner's NL
`build_prompt` — reuse the proven compiler, don't auto-save a hand-built config.
