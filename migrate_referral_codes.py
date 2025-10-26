"""One-time migration to generate referral codes for existing users"""
import random
import string
from app.database import SessionLocal
from app.models import User

def generate_referral_code(db):
    """Generate a unique referral code"""
    while True:
        code = 'TH-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        existing = db.query(User).filter(User.referral_code == code).first()
        if not existing:
            return code

def main():
    db = SessionLocal()
    try:
        # Find all users without referral codes
        users_without_codes = db.query(User).filter(User.referral_code == None).all()
        
        print(f"Found {len(users_without_codes)} users without referral codes")
        
        for user in users_without_codes:
            user.referral_code = generate_referral_code(db)
            print(f"Generated code {user.referral_code} for user {user.telegram_id} (@{user.username or user.first_name})")
        
        db.commit()
        print(f"\n✅ Successfully generated referral codes for {len(users_without_codes)} users!")
    
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
