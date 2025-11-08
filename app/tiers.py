from typing import Literal, Dict, Any
from dataclasses import dataclass

SubscriptionTier = Literal["scan", "manual", "auto"]

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
        manual_signals=False,
        auto_trading=False,
        price_usd=25.00,
        display_name="📊 Scan Mode",
        description="Top Gainers scanner only",
        features=[
            "🔍 Top Gainers scanner (real-time)",
            "📊 Volume surge detection",
            "🆕 New coin alerts",
            "📈 Pump analysis"
        ]
    ),
    "manual": TierCapabilities(
        scan_mode=True,
        manual_signals=True,
        auto_trading=False,
        price_usd=100.00,
        display_name="💎 Manual Signals",
        description="Manual signals + scan mode",
        features=[
            "✅ All Scan Mode features",
            "🔔 Manual signal notifications",
            "🎯 Entry, TP, SL levels",
            "🟢 LONGS + 🔴 SHORTS strategies",
            "🔥 3-tier early pump detection",
            "📊 PnL tracking & analytics"
        ]
    ),
    "auto": TierCapabilities(
        scan_mode=True,
        manual_signals=True,
        auto_trading=True,
        price_usd=200.00,
        display_name="🤖 Auto-Trading",
        description="Full automation + all features",
        features=[
            "✅ All Manual Signals features",
            "🤖 Automated 24/7 trade execution",
            "🏦 Bitunix integration",
            "⚙️ Advanced risk management",
            "📈 Smart exit system",
            "🎛️ Position sizing & limits",
            "🔒 Safety controls & emergency stop"
        ]
    )
}

def get_tier_config(tier: SubscriptionTier) -> TierCapabilities:
    return TIER_CONFIG[tier]

def get_tier_from_user(user) -> SubscriptionTier:
    if user.grandfathered or user.is_admin:
        return "auto"
    
    tier = user.subscription_type
    if tier in ["scan", "manual", "auto"]:
        return tier
    
    return "scan"

def has_scan_access(user) -> bool:
    tier = get_tier_from_user(user)
    return TIER_CONFIG[tier].scan_mode

def has_manual_access(user) -> bool:
    tier = get_tier_from_user(user)
    return TIER_CONFIG[tier].manual_signals

def has_auto_access(user) -> bool:
    tier = get_tier_from_user(user)
    return TIER_CONFIG[tier].auto_trading

def can_upgrade_to(current_tier: SubscriptionTier, target_tier: SubscriptionTier) -> bool:
    tier_order = ["scan", "manual", "auto"]
    current_idx = tier_order.index(current_tier)
    target_idx = tier_order.index(target_tier)
    return target_idx > current_idx
