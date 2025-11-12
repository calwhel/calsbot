# 🔧 NOWPayments Complete Setup Guide

## Overview
This guide will walk you through setting up NOWPayments for your Tradehub AI bot, enabling crypto payments for 3 subscription tiers.

---

## 📋 Prerequisites

1. **NOWPayments Account**: https://account.nowpayments.io/
2. **Railway Deployment**: Your bot must be deployed to Railway
3. **API Credentials**: API Key and IPN Secret from NOWPayments

---

## Part 1: Get Your NOWPayments Credentials

### Step 1: Create NOWPayments Account
1. Go to https://nowpayments.io/
2. Click "Sign Up" and create account
3. Complete KYC verification (required for crypto payments)

### Step 2: Get API Key
1. Login to NOWPayments dashboard
2. Go to **Settings** → **API Keys**
3. Click **"Generate API Key"**
4. Copy the key (looks like: `ABC123-DEF456-GHI789`)
5. **Save this** - you'll need it for Railway

### Step 3: Get IPN Secret Key
1. In NOWPayments dashboard, go to **Settings** → **IPN**
2. Enable IPN (Instant Payment Notifications)
3. Click **"Generate IPN Secret"**
4. Copy the secret (long random string)
5. **Save this** - you'll need it for Railway

---

## Part 2: Configure Railway Secrets

### Step 1: Add API Credentials to Railway
1. Go to Railway dashboard: https://railway.app/
2. Open your bot project
3. Click **"Variables"** tab
4. Add these 3 secrets:

**Secret 1: NOWPAYMENTS_API_KEY**
```
Variable: NOWPAYMENTS_API_KEY
Value: [Paste your API key from NOWPayments]
```

**Secret 2: NOWPAYMENTS_IPN_SECRET**
```
Variable: NOWPAYMENTS_IPN_SECRET
Value: [Paste your IPN secret from NOWPayments]
```

**Secret 3: WEBHOOK_BASE_URL**
```
Variable: WEBHOOK_BASE_URL
Value: https://your-app-name.up.railway.app
```
**How to get your Railway URL:**
- Go to **Settings** → **Domains**
- Copy your public URL (e.g., `tradehub-ai-production.up.railway.app`)
- Add `https://` prefix
- **No trailing slash!**

### Step 2: Redeploy Bot
After adding secrets, Railway will auto-redeploy. If not:
1. Go to **"Deployments"** tab
2. Click **"Redeploy"** on latest commit
3. Wait 2-3 minutes for deployment

---

## Part 3: Configure NOWPayments Dashboard

### Step 1: Set IPN Callback URL
1. Login to NOWPayments: https://account.nowpayments.io/
2. Go to **Settings** → **IPN Settings**
3. Enable **"IPN Callbacks"**
4. Set **IPN Callback URL:**
   ```
   https://your-railway-url.up.railway.app/webhooks/nowpayments
   ```
   Replace with your actual Railway URL!

5. **Enable** these notifications:
   - ✅ Payment finished
   - ✅ Payment confirmed
   - ✅ Payment failed
   - ✅ Payment expired

6. Click **"Save IPN Settings"**

### Step 2: Configure Payment Settings
1. Go to **Settings** → **Payment Settings**
2. Enable these currencies (recommended):
   - ✅ BTC (Bitcoin)
   - ✅ ETH (Ethereum)
   - ✅ USDT (Tether - TRC20 and ERC20)
   - ✅ USDC (USD Coin)
   - ✅ BNB (Binance Coin)
   - ✅ SOL (Solana)
   - ✅ LTC (Litecoin)
   - ✅ Add any others you want to accept

3. Set **Auto-conversion** (optional):
   - Convert all payments to stable coin (USDT/USDC)
   - This protects you from crypto volatility

4. Click **"Save"**

---

## Part 4: Test the Payment System

### Test 1: Generate Payment Link
1. Open your Telegram bot
2. Send `/start`
3. Click **"💎 Subscribe"**
4. Select **"📊 Scan Mode"** ($25)
5. **Expected:** Payment link appears with crypto options

**If link doesn't generate:**
- Check Railway logs for errors
- Verify `NOWPAYMENTS_API_KEY` is correct
- Ensure API key is active in NOWPayments dashboard

### Test 2: Test Payment (Small Amount)
1. Click the payment link from Test 1
2. Select **USDT (TRC20)** (lowest fees)
3. Send $25 worth of USDT to the address
4. **Expected:** Payment confirmed within 1-2 minutes
5. **Expected:** Bot sends "Subscription activated!" message

**If subscription doesn't activate:**
- Check Railway logs: Look for webhook received
- Verify `WEBHOOK_BASE_URL` is correct
- Check `NOWPAYMENTS_IPN_SECRET` matches dashboard
- Ensure IPN callbacks are enabled in NOWPayments

### Test 3: Test Referral System
1. Generate referral link: `/referral`
2. Open bot in incognito browser/different device
3. Use referral link to start bot
4. Subscribe to **Auto-Trading** ($200)
5. **Expected:** Referrer gets "$50 pending!" notification
6. **Expected:** Admin gets payout notification with wallet

---

## Part 5: Verify Webhooks are Working

### Method 1: Railway Logs
1. Go to Railway dashboard
2. Click **"Deployments"** → **"View Logs"**
3. Make a test payment
4. Search logs for: `nowpayments_webhook`
5. **Expected:** You should see:
   ```
   INFO - Received NOWPayments webhook
   INFO - Payment status: finished
   INFO - Activating subscription for user...
   ```

### Method 2: NOWPayments Dashboard
1. Go to NOWPayments → **"Payments"**
2. Find your test payment
3. Click on it → Check **"IPN Callbacks"** tab
4. **Expected:** Shows successful webhook delivery (200 OK)

**If showing errors (4xx/5xx):**
- Webhook URL is wrong
- Bot is not deployed/running
- HMAC signature mismatch (check IPN secret)

---

## 🔒 Security Checklist

✅ **API Key Security**
- Never commit API keys to Git
- Store only in Railway environment variables
- Rotate keys every 6 months

✅ **Webhook Security**
- HMAC-SHA512 signature verification enabled
- Only process "finished" or "confirmed" payments
- Invalid signatures return 401 Unauthorized

✅ **Payment Validation**
- Amount verification (prevent underpayment)
- Order ID validation (prevent replay attacks)
- User verification (prevent unauthorized access)

---

## 📊 Subscription Tiers (What Users Get)

### 📊 Scan Mode - $25/month
- Top Gainers scanner (real-time)
- Volume surge detection
- New coin alerts
- Pump analysis

### 💎 Manual Signals - $100/month
- All Scan Mode features
- Manual signal notifications
- Entry, TP, SL levels
- LONGS + SHORTS strategies
- PnL tracking

### 🤖 Auto-Trading - $200/month
- All Manual Signals features
- Automated 24/7 execution
- Bitunix integration
- Advanced risk management
- **$50 referral reward** (referrer gets cash!)

---

## 🎯 Testing Checklist

Before going live, test these scenarios:

- [ ] $25 Scan Mode payment → Activates correctly
- [ ] $100 Manual Signals payment → Activates correctly
- [ ] $200 Auto-Trading payment → Activates + referral reward
- [ ] Referral link works → Tracks referrer
- [ ] Webhook receives payments → Logs show confirmation
- [ ] Multiple currencies work → Test BTC, ETH, USDT
- [ ] Payment expiry → Expired payments don't activate
- [ ] Duplicate prevention → Same payment doesn't activate twice

---

## 🚨 Common Issues & Fixes

### Issue 1: Payment link doesn't generate
**Cause:** API key invalid or not set
**Fix:** 
1. Verify `NOWPAYMENTS_API_KEY` in Railway variables
2. Check API key is active in NOWPayments dashboard
3. Test API: `curl -H "x-api-key: YOUR_KEY" https://api.nowpayments.io/v1/status`

### Issue 2: Payment made but subscription doesn't activate
**Cause:** Webhook not reaching bot
**Fix:**
1. Check `WEBHOOK_BASE_URL` is set correctly
2. Verify IPN callback URL in NOWPayments matches Railway URL
3. Check Railway logs for webhook delivery
4. Ensure bot is deployed and running

### Issue 3: "Invalid signature" error in logs
**Cause:** IPN secret mismatch
**Fix:**
1. Copy IPN secret from NOWPayments dashboard
2. Update `NOWPAYMENTS_IPN_SECRET` in Railway
3. Redeploy bot
4. Test again

### Issue 4: Referral reward not triggered
**Cause:** Not an Auto-Trading subscription
**Fix:**
- Referral rewards ONLY trigger for Auto-Trading ($200/mo)
- Scan Mode and Manual Signals don't give referral rewards
- Check user subscribed to correct tier

---

## 🔧 Maintenance

### Weekly Tasks
- Check pending referral payouts (`referral_earnings`)
- Process manual crypto payouts to referrers
- Monitor successful vs failed payments

### Monthly Tasks
- Review payment logs for errors
- Update accepted currencies if needed
- Rotate API keys for security

---

## 📞 Support

**NOWPayments Support:**
- Email: support@nowpayments.io
- Documentation: https://documenter.getpostman.com/view/7907941/S1a32n38

**Telegram Bot Support:**
- Check Railway logs for detailed errors
- Verify all environment variables are set
- Test with small amounts first ($25 tier)

---

## ✅ Final Pre-Launch Checklist

- [ ] NOWPayments account created and verified
- [ ] API Key added to Railway (`NOWPAYMENTS_API_KEY`)
- [ ] IPN Secret added to Railway (`NOWPAYMENTS_IPN_SECRET`)
- [ ] Webhook URL added to Railway (`WEBHOOK_BASE_URL`)
- [ ] IPN callback configured in NOWPayments dashboard
- [ ] Accepted currencies configured (BTC, ETH, USDT, etc.)
- [ ] Test payment completed successfully ($25 tier)
- [ ] Subscription activated automatically
- [ ] Referral system tested with $200 Auto-Trading
- [ ] $50 reward notification received
- [ ] Admin payout notification received
- [ ] All logs show successful webhook delivery

**Status:** Ready to launch! 🚀

---

**Last Updated:** November 2025
**Bot:** @AISIGNALPERPBOT (Tradehub AI)
