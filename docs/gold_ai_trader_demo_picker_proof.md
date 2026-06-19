# Gold AI Trader — Demo Account Picker Proof

## Behaviour

1. **Dropdown** lists only connected cTrader accounts with `isLive=false` from `user_preferences.ctrader_accounts` (same JSON source as Live Forex).
2. **Live accounts excluded** — `is_confirmed_demo_account()` requires `isLive is False`; live rows never appear in `demo_accounts`.
3. **Persistence** — selection POSTs `demo_ctrader_account_id` to `/api/gold-ai-trader/config` → saved on `gold_ai_config.demo_ctrader_account_id` (survives restarts; overrides env when set in DB).
4. **Demo lock intact** — `assert_demo_account()` still requires configured ctid + `isLive=false` before any order (`executor.py` unchanged path).
5. **No demo → no run** — `check_can_call_claude()` / `check_can_execute()` return `no_demo_account`; API rejects `enabled:true` without a selected demo; UI disables trader toggle.

## API (`GET /api/gold-ai-trader/status`)

```json
{
  "demo_accounts": [
    {"ctid": "47465772", "label": "#47465772 · login 12345 · Demo", "trader_login": 12345, "balance": 10000}
  ],
  "demo_account_selected": {"ctid": "47465772", "label": "#47465772 · login 12345 · Demo"},
  "demo_account_ready": true,
  "config": { "demo_ctrader_account_id": "47465772", ... }
}
```

## UI

- **Trading demo account** bar (gold) under demo pill — shows selected ctid + label, or warning.
- **Settings → Demo account** dropdown + Save (demo-only list).
- Trader switch disabled until `demo_account_ready`.

## Tests

- `test_demo_accounts_filters_live_only` — live ctid excluded from list
- `test_validate_demo_ctid_rejects_live` — cannot save live ctid as demo
- `test_no_demo_account_blocks_claude` — guardrail `no_demo_account`
- `test_demo_account_lock_rejects_live` — `assert_demo_account` unchanged

## Files changed (gold_ai_trader module only)

- `app/gold_ai_trader/accounts.py` (new)
- `app/gold_ai_trader/routes.py`
- `app/gold_ai_trader/guardrails.py`
- `app/templates/gold_ai_trader.html`
- `tests/test_gold_ai_trader.py`
