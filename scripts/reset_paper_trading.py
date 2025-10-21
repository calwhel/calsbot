"""
Reset all paper trading data after strategy change.
This script will:
1. Delete all paper trades
2. Reset everyone's paper trading mode to False
3. Reset auto-trading to disabled for safety
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models import PaperTrade, UserPreference

def reset_paper_trading():
    """Reset all paper trading data and settings"""
    db = SessionLocal()
    
    try:
        # Count existing data
        paper_trades_count = db.query(PaperTrade).count()
        users_with_paper_mode = db.query(UserPreference).filter(
            UserPreference.paper_trading_mode == True
        ).count()
        
        print(f"\n📊 Current State:")
        print(f"   - Paper trades: {paper_trades_count}")
        print(f"   - Users in paper mode: {users_with_paper_mode}")
        
        # Delete all paper trades
        deleted_trades = db.query(PaperTrade).delete()
        print(f"\n🗑️  Deleted {deleted_trades} paper trades")
        
        # Reset all users' paper trading mode to False
        updated_users = db.query(UserPreference).update({
            UserPreference.paper_trading_mode: False
        })
        print(f"✅ Reset paper mode for {updated_users} users")
        
        # Commit changes
        db.commit()
        print(f"\n✅ Paper trading reset complete!")
        print(f"   All users are now in LIVE mode (auto-trading still requires API keys)")
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error resetting paper trading: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("⚠️  WARNING: This will delete ALL paper trading data!")
    print("   All users will be switched to LIVE mode.")
    
    confirm = input("\nType 'RESET' to confirm: ")
    
    if confirm == "RESET":
        reset_paper_trading()
    else:
        print("\n❌ Reset cancelled")
