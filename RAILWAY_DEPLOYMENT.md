# Deploy Your Crypto Trading Bot to Railway.app

## ✅ Your bot is ready for Railway! All files are configured.

---

## 🚀 Deployment Steps (5 Minutes)

### **Step 1: Push Code to GitHub** (if not already done)

```bash
# Initialize git (skip if already done)
git init
git add .
git commit -m "Ready for Railway deployment"

# Create new GitHub repo, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

---

### **Step 2: Deploy to Railway**

1. Go to **https://railway.app**
2. Click **"Login"** → Login with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your bot repository
6. Railway will automatically deploy!

---

### **Step 3: Add PostgreSQL Database**

1. In your Railway project dashboard
2. Click **"+ New"** button
3. Select **"Database"** → **"Add PostgreSQL"**
4. Done! Railway automatically connects it

---

### **Step 4: Add Environment Variables**

Click on your service → **"Variables"** tab → Add these:

```
TELEGRAM_BOT_TOKEN=your_bot_token_from_@BotFather
CRYPTOPANIC_API_KEY=your_cryptopanic_key
ENCRYPTION_KEY=your_32_character_encryption_key
EXCHANGE=kucoin
SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT
BROADCAST_CHAT_ID=your_telegram_id
SESSION_SECRET=random_secret_string_here
```

**Important:** Railway auto-generates `DATABASE_URL` when you add PostgreSQL - you don't need to add it manually!

---

### **Step 5: Deploy!**

1. Click **"Deploy"** after adding variables
2. Check **"Deployments"** tab to monitor progress
3. View **"Logs"** to see your bot starting

---

## 🎯 That's It!

Your bot will now run 24/7 on Railway! It will:
- ✅ Auto-restart if it crashes
- ✅ Keep running even when your laptop is off
- ✅ Auto-deploy when you push to GitHub
- ✅ Scale automatically

---

## 📊 Monitor Your Bot

- **View Logs**: Railway Dashboard → Your Service → Logs
- **Restart Bot**: Settings → Restart
- **Usage**: Check dashboard for resource usage

---

## 💰 Pricing

Railway offers:
- **$5 free credits** per month (enough for a bot like this)
- **Pay-as-you-go** after free credits
- **Typical cost:** $5-10/month for 24/7 bot + database

---

## 🔧 Troubleshooting

**Bot not responding?**
- Check Variables tab → Verify `TELEGRAM_BOT_TOKEN` is set
- Check Logs → Look for errors

**Database errors?**
- Make sure PostgreSQL service is added
- Railway auto-injects `DATABASE_URL` - don't override it

**Need to update code?**
- Just push to GitHub → Railway auto-deploys

---

## 📱 Quick Links

- **Railway Dashboard**: https://railway.app/dashboard
- **Railway Docs**: https://docs.railway.app
- **Template for reference**: https://railway.app/template/a0ln90

---

Your bot is now production-ready! 🚀
