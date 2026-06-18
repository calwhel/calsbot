# Gold AI Trader — Live Mirror Proof

## Model

| Leg | Role | Default |
|-----|------|---------|
| **Demo** | Source of truth — P&L, stats, decision feed | Always active when trader enabled |
| **Live mirror** | Optional copy of TAKE orders to a real cTrader account | **OFF** until explicit enable + confirm |

## Flow (unchanged Claude logic)

```
Claude TAKE (conf ≥ 70)
  ├─► execute_take()          → demo ctid (assert_demo_account, isLive=false)
  └─► execute_live_mirror_take()  → live ctid IF enabled (assert_live_account, isLive=true)
        └─► enqueue_ctrader_order() → standard live forex queue (reconcile + failure notify)
```

## Safety checks

1. **Demo lock unchanged** — `assert_demo_account()` still requires configured demo ctid + `isLive=false`.
2. **Live lock separate** — `assert_live_account()` requires configured live ctid + `isLive=true`.
3. **Enable requires confirm** — API rejects `live_mirror_enabled=true` without `confirm_real_money: true`.
4. **Kill switch** — `check_can_execute_live_mirror()` returns `kill_switch` when active (stops both legs).
5. **Independent caps** — demo `max_trades_day` vs live `max_live_trades_day`; live lot size default 0.01.

## Linking demo ↔ live

- Demo execution notes: `gold_ai_trader decision_id=N`
- Live execution notes: `gold_ai_trader_live_mirror decision_id=N demo_exec=D`
- `GoldAiDecision.live_mirror_execution_id` + `live_mirror_status` (pending/filled/failed/skipped)

## Example decision log entry

```json
{
  "id": 42,
  "action": "take",
  "executed": true,
  "execution_id": 9001,
  "live_mirror_execution_id": 9002,
  "live_mirror_status": "pending",
  "live_mirror_error": null
}
```

## Disconnect live (demo keeps running)

`POST /api/gold-ai-trader/disconnect-live?uid=...` → sets `live_mirror_enabled=false` only.

## UI

- Red **Live mirror · real money** section (separate from demo stats)
- Demo P&L tile labelled **Source of truth**
- Live P&L + live trades shown separately
- Confirm dialog before enable
- **Disconnect live** button (does not stop demo)
