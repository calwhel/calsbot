# Gold AI Trader — delivery proof

## 1. Files changed (additive module)

**New files only** (except one integration hook in `strategy_portal_server.py`):

| Path | Role |
|------|------|
| `app/gold_ai_trader/__init__.py` | Package |
| `app/gold_ai_trader/config.py` | Env + defaults |
| `app/gold_ai_trader/models.py` | DB models |
| `app/gold_ai_trader/schema.py` | Table creation |
| `app/gold_ai_trader/guardrails.py` | Demo lock, caps, kill switch |
| `app/gold_ai_trader/state.py` | Runtime status |
| `app/gold_ai_trader/scanner.py` | Session gate + TA candidates |
| `app/gold_ai_trader/context.py` | Context builder |
| `app/gold_ai_trader/claude.py` | Opus 4.8 + prompt caching |
| `app/gold_ai_trader/executor.py` | Demo cTrader placement |
| `app/gold_ai_trader/learning.py` | Lessons digest |
| `app/gold_ai_trader/loop.py` | Isolated background loop |
| `app/gold_ai_trader/routes.py` | API + page |
| `app/gold_ai_trader/portal_mount.py` | Portal registration |
| `app/templates/gold_ai_trader.html` | UI |
| `tests/test_gold_ai_trader.py` | Unit tests |

**Not modified:** strategy builder, wizard, compiler, `evaluate_and_fire`, order queue internals, trade management, reconcile.

**Single integration hook:** `strategy_portal_server.py` calls `mount_gold_ai_trader_portal(app)` at import time.

## 2. Sample context snapshot (candidate: sweep_pdh SHORT)

```
=== GOLD AI TRADER CONTEXT (XAUUSD) ===
Timestamp UTC: 2026-06-18T08:15:00Z
Session: LONDON | Killzone: yes | Minutes into session: 75

=== PRICE ===
Spot: 2650.50
ATR(14) 5m: 3.85 (0.145% of price) | RVOL(5m): 1.42x

=== TRIGGER (why Claude was called) ===
Type: sweep_pdh | Direction bias: SHORT
Detail: XAUUSD: swept PDay high then closed back below (liquidity grab)
Quality vs ATR: 1.20× (engine estimate)

=== ACCOUNT (DEMO) ===
Open gold_ai positions: 0/1 max
Trades today: 1/6 | Claude calls: 4/50
...
=== DECISION RULE REMINDER ===
Default SKIP unless high conviction. Require clear invalidation and ≥2:1 R:R.
```

## 3. Sample Claude decision JSON (dry-run / malformed-safe path)

```json
{
  "action": "skip",
  "direction": null,
  "entry": null,
  "stop_loss": null,
  "take_profit": null,
  "confidence": 42,
  "rationale": "Dry-run: insufficient displacement confirmation vs ATR; standing aside."
}
```

Live path uses `claude-opus-4-8` with ephemeral cache on `SYSTEM_PROMPT`.

## 4. Demo execution + decision log row

On `action=take` with `confidence>=70` and guardrails pass:

1. `gold_ai_decisions` row inserted (context, reasoning, tokens, cost)
2. `place_market_order_resilient(..., volume_lots=0.01)` on **demo host only** after `assert_demo_account`
3. `strategy_executions` row with `notes=gold_ai_trader decision_id=N` for existing forex BE/trail/reconcile
4. Decision row updated: `executed=true`, `execution_id=<id>`

## 5. Guardrails (server-side)

| Guard | Enforcement |
|-------|-------------|
| Demo only | `GOLD_AI_TRADER_DEMO_ACCOUNT_ID` + `isLive=false` in `assert_demo_account` before every order |
| Max calls/day | `check_can_call_claude` → stops Claude |
| Max trades/day | `check_can_execute` |
| Max 1 open | `open_position_count >= 1` blocks |
| Kill switch | DB `kill_switch` + runtime status |
| Feature flag | `GOLD_AI_TRADER_ENABLED=false` default — loop never starts |
| Min size | `volume_lots=0.01` |

## 6. UI

Page: **`/gold-ai-trader?uid=<admin_uid>`**

- DEMO banner, kill switch, ON/OFF, session config, caps
- Live decision feed + today's cost/calls/trades
- Lessons digest

## Enable in production

```bash
GOLD_AI_TRADER_ENABLED=true
GOLD_AI_TRADER_USER_ID=<user_id_with_ctrader_demo>
GOLD_AI_TRADER_DEMO_ACCOUNT_ID=<demo_ctidTraderAccountId>
ANTHROPIC_API_KEY=...
```

Then toggle **ON** in UI (admin) and ensure kill switch is off.
