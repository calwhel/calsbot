#!/bin/bash

echo "🚂 Railway Auto-Deploy Script"
echo "=============================="
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    curl -fsSL https://railway.app/install.sh | sh
    export PATH="$HOME/.railway/bin:$PATH"
fi

echo "🔐 Logging into Railway..."
railway login

echo ""
echo "📋 Which project do you want to deploy to?"
echo "1) Create NEW Railway project"
echo "2) Deploy to EXISTING project (f93ee534-1240-4059-a84b-37b9168f9de6)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo "🆕 Creating new Railway project..."
    railway init
elif [ "$choice" == "2" ]; then
    echo "🔗 Linking to existing project..."
    railway link f93ee534-1240-4059-a84b-37b9168f9de6
else
    echo "❌ Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 View logs: railway logs"
echo "🌐 Open dashboard: railway open"
