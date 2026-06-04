---
name: cTrader account list is token-authorization-scoped
description: Why a newly created cTrader account doesn't appear until the user re-links (re-authorizes OAuth).
---

# cTrader account list only shows accounts the access token was authorized for

`get_accounts_for_token` (ProtoOAGetAccountListByAccessToken) returns ONLY the
trading accounts that were granted to that access token during the OAuth consent.
A cTrader account created AFTER the user linked is NOT auto-included — the user
must RE-LINK (re-authorize via the granting-access page) to grant the token access
to the new account. This is true for demo accounts too (demo + live both appear
once authorized; the list is token-scoped, not host-scoped — querying the LIVE
host still returns authorized demo accounts).

**Why:** A user "made a new demo account but it's not coming up." Two layers:
- The picker endpoint (`/api/ctrader/accounts`) originally returned only the JSON
  snapshot captured at link time and never re-fetched → even already-authorized
  new accounts wouldn't show. Fixed to live re-fetch each load (fallback to cached
  on failure).
- Even after live re-fetch, cTrader still didn't return the brand-new account
  because it wasn't in the token's OAuth grant → re-link is the only fix.

**How to apply:**
- When a user says a new cTrader account isn't appearing: the live re-fetch will
  surface any TOKEN-AUTHORIZED account automatically; if it still doesn't show,
  tell them to re-link/reconnect cTrader (re-authorize) so the new account is
  granted to the token.
- `get_accounts_for_token` returns `[]` on BOTH success-empty AND internal failure
  (conflated). The picker treats `[]` as "keep cached" so it never shows an empty
  picker on a transient outage — acceptable since a linked user always has ≥1
  account; the cost is a legitimately-emptied list stays stale until next success.

Related: stuck OPEN live forex execs (no `pos=` positionId in notes) can't be
reconciled and block the max_open gate until the 48h `close_stale_open_executions`
sweep — clear them sooner if a user reports "says open but isn't / not firing".
See forex-live-sl-management.md + ghost-cleanup-broker-id-trap.md.
