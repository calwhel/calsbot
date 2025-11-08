"""
One-time migration script to add crypto_wallet column to users table.
Run this once on Railway to fix the database.
"""
import os
from sqlalchemy import create_engine, text

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL not found in environment variables")
    exit(1)

print("🔄 Connecting to database...")
engine = create_engine(DATABASE_URL)

print("🔧 Adding crypto_wallet column to users table...")

try:
    with engine.connect() as conn:
        # Add the column if it doesn't exist
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS crypto_wallet VARCHAR"))
        conn.commit()
        print("✅ SUCCESS! crypto_wallet column added to users table")
        print("✅ Your bot should work now. Restart it on Railway.")
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nIf the column already exists, you can ignore this error.")
finally:
    engine.dispose()
