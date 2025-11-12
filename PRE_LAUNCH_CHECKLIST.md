# 🚀 PRE-LAUNCH CHECKLIST - Tradehub AI Bot

## ✅ PAYMENT SYSTEM

### NOWPayments Integration
- [x] API Key configured: `NOWPAYMENTS_API_KEY` ✅
- [x] IPN Secret configured: `NOWPAYMENTS_IPN_SECRET` ✅
- [x] API connection tested: `{"message":"OK"}` ✅
- [x] Webhook endpoint: `/webhooks/nowpayments` ✅
- [x] HMAC-SHA512 signature verification ✅
- [ ] **ACTION REQUIRED:** Set `WEBHOOK_BASE_URL` secret to your Railway URL
  - Example: `https://your-app-name.railway.app`
  - This enables automatic subscription activation

### Subscription Tiers
- [x] Scan Mode: $25/month ✅
- [x] Manual Signals: $100/month ✅
- [x] Auto-Trading: $200/month ✅
- [x] Order ID format: `sub_{plan_type}_{telegram_id}_{timestamp}` ✅
- [x] 30-day subscription duration ✅
- [x] Instant activation on payment confirmation ✅

### Payment Flow
1. User clicks subscription tier → Payment link generated
2. User pays with crypto (BTC/ETH/USDT/200+ coins)
3. NOWPayments sends webhook to `/webhooks/nowpayments`
4. Bot verifies HMAC signature
5. Bot activates subscription + sends confirmation
6. **Referral payout triggered if Auto-Trading subscription**

---

## ✅ REFERRAL SYSTEM

### Referral Code Generation
- [x] Format: `TH-XXXXXX` (6 alphanumeric chars) ✅
- [x] Uniqueness check in database ✅
- [x] Auto-generated on user registration ✅
- [x] Stored in `users.referral_code` ✅

### Referral Tracking
- [x] `referred_by` field stores referrer's code ✅
- [x] Referral link format: `t.me/AISIGNALPERPBOT?start=ref_TH-XXXXXX` ✅
- [x] Deep linking handled by `/start ref_{code}` ✅

### Cash Reward System ($50 Auto-Trading Only)
- [x] Reward amount: $50 USD (configurable) ✅
- [x] Triggers only on Auto-Trading subscriptions ($200/mo) ✅
- [x] Duplicate prevention: `paid_referrals` JSON list ✅
- [x] Pending earnings tracked in `referral_earnings` ✅
- [x] Wallet address required: `crypto_wallet` field ✅

### Payout Flow
1. User subscribes to Auto-Trading via referral link
2. Bot checks if referral already paid (prevents duplicates)
3. Bot adds $50 to `referrer.referral_earnings`
4. Bot notifies referrer with pending balance
5. Bot notifies admins with payout details
6. **Manual payout**: Admin sends crypto to `referrer.crypto_wallet`

### Notifications
- [x] Referrer notification: Pending $50 reward ✅
- [x] Wallet address reminder if not set ✅
- [x] Admin notification: Payout request with wallet address ✅
- [x] User confirmation: Subscription activated ✅

---

## ✅ DATABASE SCHEMA

### Users Table
- [x] `referral_code` (unique, TH-XXXXXX format) ✅
- [x] `referred_by` (who referred this user) ✅
- [x] `referral_earnings` (pending $50 payouts) ✅
- [x] `paid_referrals` (JSON list of paid user IDs) ✅
- [x] `crypto_wallet` (for receiving payouts) ✅
- [x] `subscription_type` (scan/manual/auto) ✅
- [x] `subscription_end` (expiry date) ✅

### Subscriptions Table
- [x] Payment history tracking ✅
- [x] Transaction IDs from NOWPayments ✅
- [x] Amount and duration logging ✅

---

## ✅ BOT COMMANDS

### User Commands
- [x] `/start` - Registration + referral tracking ✅
- [x] `/subscribe` - Legacy subscription (manual tier) ✅
- [x] Menu button: "💎 Subscribe" - 3-tier selection ✅
- [x] `/referral` - Referral stats and link ✅
- [x] `/setwallet [address]` - Set crypto wallet for payouts ✅

### Admin Commands
- [x] View pending payouts ✅
- [x] User management ✅
- [x] System health monitoring ✅

---

## ⚠️ CRITICAL PRE-LAUNCH ACTIONS

1. **Set WEBHOOK_BASE_URL Secret**
   ```bash
   # On Railway, add this secret:
   WEBHOOK_BASE_URL=https://your-app-name.railway.app
   ```
   **Why:** Without this, NOWPayments can't notify the bot about completed payments!

2. **Test Payment Flow (Live Test)**
   - Subscribe to Scan Mode ($25) with a test wallet
   - Verify webhook is received
   - Confirm subscription activates instantly
   - Check database for proper tier assignment

3. **Test Referral Flow**
   - Generate referral link: `/referral`
   - Open bot with referral link in incognito
   - Subscribe to Auto-Trading ($200)
   - Verify $50 added to referrer's `referral_earnings`
   - Verify admin notification received

4. **Verify Wallet Command**
   - `/setwallet 0x1234...` (Ethereum address)
   - Confirm wallet saved in database
   - Verify wallet appears in admin payout notifications

5. **Check NOWPayments Dashboard**
   - Login to NOWPayments
   - Verify IPN callback URL is set
   - Check that webhook secret matches `NOWPAYMENTS_IPN_SECRET`

---

## 📊 MONITORING CHECKLIST

After Launch:
- [ ] Monitor `/webhooks/nowpayments` endpoint logs
- [ ] Track successful vs failed payments
- [ ] Verify all subscriptions activate automatically
- [ ] Monitor referral earnings accumulation
- [ ] Process manual payouts weekly/monthly
- [ ] Check for duplicate referral payouts

---

## 🔧 TECHNICAL DETAILS

### Webhook Security
- HMAC-SHA512 signature verification prevents spoofed payments
- Only "finished" or "confirmed" payments activate subscriptions
- Invalid signatures return 401 Unauthorized

### Error Handling
- Failed payment notifications logged but don't activate subscriptions
- Duplicate referral payouts prevented via `paid_referrals` JSON list
- Wallet address optional during signup, required for payout

### Scalability
- Referral earnings accumulate (support multiple referrals)
- Subscription renewals extend existing expiry date
- Admins can manually adjust subscription status

---

## 🎯 SUCCESS METRICS

Track these post-launch:
- Conversion rate: Free users → Paid subscribers
- Referral performance: How many referrals per user?
- Revenue breakdown: Scan ($25) vs Manual ($100) vs Auto ($200)
- Payout efficiency: Pending earnings → Completed payouts

---

**Status:** 95% Ready for Launch
**Blocker:** Set `WEBHOOK_BASE_URL` secret on Railway
**ETA:** Ready to launch after webhook URL configured! 🚀
