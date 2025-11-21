from typing import Literal, Dict, Any
from dataclasses import dataclass

SubscriptionTier = Literal["auto"]

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
    "auto": TierCapabilities(
        scan_mode=True,
        manual_signals=True,
        auto_trading=True,
        price_usd=130.00,
        display_name="🤖 Auto-Trading",
        description="Full automation + all features",
        features=[
            "🔍 Top Gainers scanner (real-time)",
            "📊 Volume surge detection",
            "🆕 New coin alerts",
            "🤖 Automated 24/7 trade execution",
            "🏦 Bitunix integration",
            "⚙️ Advanced risk management",
            "📈 Smart exit system",
            "🎛️ Position sizing & limits",
            "🔒 Safety controls & emergency stop",
            "🟢 LONGS + 🔴 SHORTS strategies",
            "🔥 Parabolic dump detection",
            "📊 PnL tracking & analytics"
        ]
    )
}

def get_tier_config(tier: SubscriptionTier) -> TierCapabilities:
    return TIER_CONFIG[tier]

def get_tier_from_user(user) -> SubscriptionTier:
    # Auto-Trading is the only tier now
    return "auto"

def has_scan_access(user) -> bool:
    return True  # All users have scan access

def has_manual_access(user) -> bool:
    return True  # All users have signal access

def has_auto_access(user) -> bool:
    # Auto-trading requires subscription
    return user.is_subscribed or user.grandfathered or user.is_admin

def can_upgrade_to(current_tier: SubscriptionTier, target_tier: SubscriptionTier) -> bool:
    # Single tier - no upgrades
    return False
