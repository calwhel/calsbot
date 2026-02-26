from typing import Literal, Dict, Any
from dataclasses import dataclass

SubscriptionTier = Literal["scan", "auto"]

@dataclass
class TierCapabilities:
    scan_mode: bool
    manual_signals: bool
    auto_trading: bool
    price_usd: float
    display_name: str
    description: str
    features: list[str]

TIER_CONFIG: Dict[SubscriptionTier, TierCapabilities] = {
    "scan": TierCapabilities(
        scan_mode=True,
        manual_signals=True,
        auto_trading=False,
        price_usd=65.00,
        display_name="🤖 AI Assistant",
        description="AI-powered analysis + market scanning",
        features=[
            "🤖 Tradehub AI Assistant (unlimited)",
            "🔍 Market Scanner (find opportunities)",
            "⚠️ Risk Assessment (trade evaluation)",
            "📊 Real-time coin analysis",
            "💬 Natural language trading Q&A",
            "📈 Technical indicators & trends",
            "🎯 Entry/exit recommendations",
            "📱 24/7 chat support"
        ]
    ),
    "auto": TierCapabilities(
        scan_mode=True,
        manual_signals=True,
        auto_trading=True,
        price_usd=80.00,
        display_name="🚀 Auto-Trading",
        description="Full automation + all AI features",
        features=[
            "✅ Everything in AI Assistant PLUS:",
            "🤖 Automated 24/7 trade execution",
            "🏦 Bitunix exchange integration",
            "🔍 Top Gainers scanner (real-time)",
            "📊 Volume surge detection",
            "🆕 New coin alerts",
            "⚙️ Advanced risk management",
            "📈 Smart exit system (auto TP/SL)",
            "🎛️ Position sizing & limits",
            "🔒 Safety controls & emergency stop",
            "🟢 LONGS + 🔴 SHORTS strategies",
            "🔥 Parabolic dump detection",
            "📊 PnL tracking & analytics"
        ]
    )
}

def get_tier_config(tier: SubscriptionTier) -> TierCapabilities:
    return TIER_CONFIG.get(tier, TIER_CONFIG["auto"])

def get_tier_from_user(user) -> SubscriptionTier:
    """Get subscription tier from user object"""
    if user.is_admin or user.grandfathered:
        return "auto"  # Admins and grandfathered users get full access
    
    sub_type = getattr(user, 'subscription_type', 'auto')
    if sub_type == "scan":
        return "scan"
    return "auto"  # Default to auto for "auto" and legacy "manual"

def has_scan_access(user) -> bool:
    """Check if user has access to AI Assistant / Scan features"""
    if user.is_admin or user.grandfathered:
        return True
    if not user.is_subscribed:
        return False
    # Both "scan" and "auto" tiers have scan access
    return user.subscription_type in ["scan", "auto", "manual"]

def has_manual_access(user) -> bool:
    """Check if user has access to manual signals"""
    if user.is_admin or user.grandfathered:
        return True
    if not user.is_subscribed:
        return False
    # Both tiers have manual signal access
    return user.subscription_type in ["scan", "auto", "manual"]

def has_auto_access(user) -> bool:
    """Check if user has access to auto-trading execution"""
    if user.is_admin or user.grandfathered:
        return True
    if not user.is_subscribed:
        return False
    # Only "auto" tier has auto-trading access
    return user.subscription_type == "auto"

def can_upgrade_to(current_tier: SubscriptionTier, target_tier: SubscriptionTier) -> bool:
    """Check if user can upgrade from current to target tier"""
    tier_order = {"scan": 1, "auto": 2}
    return tier_order.get(target_tier, 0) > tier_order.get(current_tier, 0)
