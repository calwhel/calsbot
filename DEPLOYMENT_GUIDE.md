# 🚀 Deploy Your Crypto Trading Bot to Railway (EU Region)

This guide will help you deploy your bot to Railway.app with an EU/UK server to avoid US IP geo-blocking issues with MEXC and KuCoin.

---

## 📋 **What You'll Need:**

1. **GitHub Account** (free) - [Sign up here](https://github.com/signup)
2. **Railway Account** (free tier available) - [Sign up here](https://railway.app)
3. **Your Environment Variables** (API keys and secrets from Replit)

---

## 🔧 **Step 1: Push Your Code to GitHub**

### Option A: Using Replit's Git Integration (Easiest)

1. In Replit, click the **Tools** button in the sidebar
2. Select **Git** 
3. Click **Create a Git Repo**
4. Follow the prompts to connect to GitHub
5. Push your code to GitHub

### Option B: Manual Upload

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it: `crypto-trading-bot`
3. Make it **Private** (to protect your code)
4. Download your Replit project as ZIP
5. Upload files to your GitHub repository

---

## 🌍 **Step 2: Deploy to Railway with EU Region**

### 2.1 Create Railway Project

1. Go to [Railway.app](https://railway.app)
2. Click **Start a New Project**
3. Select **Deploy from GitHub repo**
4. Choose your `crypto-trading-bot` repository
5. Railway will auto-detect it as a Python app ✅

### 2.2 Set Your Region to Europe

**IMPORTANT:** Railway auto-deploys to US by default. You must change the region!

1. In your Railway project, click **Settings** (⚙️ icon)
2. Scroll down to **Region**
3. Select: **`eu-west-1` (Ireland)** or **`eu-central-1` (Frankfurt)**
4. Click **Save**
5. **Redeploy** your service for the region change to take effect

---

## 🔐 **Step 3: Add Your Environment Variables**

1. In Railway, click on your service
2. Go to **Variables** tab
3. Add these environment variables one by one:

```
TELEGRAM_BOT_TOKEN=<your_token>
BROADCAST_CHAT_ID=<your_chat_id>
CRYPTOPANIC_API_KEY=<your_api_key>
ENCRYPTION_KEY=<your_encryption_key>
EXCHANGE=<exchange_name>
SYMBOLS=<your_symbols>
SESSION_SECRET=<your_session_secret>
```

**Important:** Copy these from your Replit **Secrets** tab!

### Add Database Variables:

Railway auto-creates a PostgreSQL database. After deployment:

1. Click **New** → **Database** → **Add PostgreSQL**
2. Railway will auto-create these variables:
   - `DATABASE_URL`
   - `PGHOST`
   - `PGPORT`
   - `PGUSER`
   - `PGPASSWORD`
   - `PGDATABASE`

---

## ✅ **Step 4: Verify Deployment**

### 4.1 Check Deployment Status

1. Go to **Deployments** tab
2. Wait for status: **SUCCESS** ✅
3. Check the logs for any errors

### 4.2 Test Your Bot

1. Open Telegram
2. Send `/start` to your bot
3. Try `/test_mexc` or `/test_kucoin` to verify API connection
4. Send `/test_autotrader` to test a trade

**If it works, congratulations! Your bot is now running from an EU server!** 🎉

---

## 🔍 **Troubleshooting**

### Bot Not Starting?

1. Check **Logs** in Railway
2. Look for missing environment variables
3. Make sure all secrets are copied correctly

### Still Getting Geo-Blocked?

1. Verify your Railway region is set to **EU** (not US)
2. Redeploy after changing region
3. Check logs for IP address - should be EU-based

### Database Connection Issues?

1. Make sure PostgreSQL is added in Railway
2. Check that `DATABASE_URL` is automatically set
3. Review connection logs

---

## 💰 **Railway Pricing**

- **Free Tier:** $5/month in free credits (enough for small bots)
- **Pro Plan:** $20/month (for higher usage)
- Your bot will likely fit in the free tier!

---

## 📊 **After Deployment**

### Monitor Your Bot:

1. **Railway Dashboard** - Check logs, metrics, deployments
2. **Telegram Commands:**
   - `/dashboard` - View trading status
   - `/autotrading_status` - Check exchange connection
   - `/security_status` - Monitor safety limits

### Enable Auto-Trading:

1. Make sure your exchange API keys are set: `/set_mexc_api` or `/set_kucoin_api`
2. Enable auto-trading: `/toggle_autotrading`
3. Test it: `/test_autotrader`

---

## 🆘 **Need Help?**

If you run into issues:

1. Check Railway **logs** first
2. Review Telegram bot error messages
3. Verify all environment variables are set correctly
4. Make sure you're using an **EU region** in Railway

---

## 🎯 **Quick Checklist**

- [ ] Code pushed to GitHub
- [ ] Railway project created
- [ ] **Region set to EU** (Ireland or Frankfurt)
- [ ] All environment variables added
- [ ] PostgreSQL database added
- [ ] Deployment successful
- [ ] Bot responds in Telegram
- [ ] Exchange API connection works
- [ ] Test trade successful

**Once all checked, your bot is live and trading from an EU server!** ✅

---

Good luck with your deployment! 🚀
