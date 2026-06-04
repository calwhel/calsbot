---
name: AI builder strategy-config emission
description: How the AI chat-builder/compiler emits & sanitizes top-level strategy CONFIG (sessions, trading_days, risk limits), distinct from entry SIGNALS.
---

The AI strategy builder must be able to set top-level CONFIGURATION (not just entry signals) from natural language: `filters.session`, `filters.trading_days`, and `risk.*` (position_size_pct, max_trades_per_day, max_open_positions, cooldown_minutes, daily_loss_limit_pct). These are read directly by the executor and are editable in the wizard.

**Why:** the compiler/STRATEGY_SCHEMA originally never emitted session/trading_days/risk-limit fields, so "only trade the NY session" or "max 2 open positions" had no effect even though the wizard + executor already supported them.

**How to apply — four synced layers (like signal-type-layers.md, but for config):**
1. `strategy_builder.py` STRATEGY_SCHEMA: expose `filters.session` + `filters.trading_days`.
2. `strategy_builder.py` COMPILER_SYSTEM_PROMPT: SESSION & DAY CONFIGURATION + RISK CONFIGURATION mapping rules, and chat-field parse hints.
3. `strategy_portal_server.py` chat-builder `###STRATEGY###` grammar + FIELD RULES: Sessions / Trading Days / Daily Loss Limit / Max Open Positions / Cooldown fields.
4. `strategy_portal_server.py` `_describe_existing_config()` (improve-mode serializer): MUST round-trip these fields or an "improve" edit silently drops them.

**Canonical values:** session ids = asian / london / new_york / overlap (executor `_SESSION_HOURS` also tolerates ny/tokyo/europe aliases). Day names = lowercase full (`monday`…). Session stored as `{"type":"session","sessions":[...]}`.

**Robustness rules (LLMs emit messy shapes):**
- Post-compile, run a normalizer (`_normalize_compiled_config` + `_parse_session_ids`/`_parse_trading_days`) that coerces dict/list/**string** (incl. `"none"`→drop, `"Mon-Fri"` range expand, comma lists), drops unknown tokens, and clamps risk knobs. Apply it to the output of `compile_strategy_from_conversation` (covers BOTH the Anthropic and Gemini compiler paths).
- Executor gates `_check_trading_days` / `_check_time_filter` must FAIL OPEN (allow trading) on malformed types (a raw string would otherwise iterate char-by-char and block every day, or `.get()` on a non-dict would raise).
- Improve-mode serializer must tolerate legacy non-dict `filters.session` (dict/list/string) without raising.

**Prompt precedence (don't double-gate):** plain "only trade during X session" → top-level `filters.session` (cross-asset, wizard-editable). Reserve the `forex_session` `in_session` CONDITION for ICT signal combos or session_open/session_close sub-windows only. Keep this consistent across ALL prompt layers (strategy_builder FOREX rules + NLP mapping line + portal SIGNAL RECOGNITION) — they drifted and reintroduced the contradiction twice in review.
