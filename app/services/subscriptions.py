from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import hmac
import json

from app.database import get_db, init_db
from app.models import User, Subscription
from app.config import settings

api = FastAPI()


@api.get("/")
async def root():
    return {"status": "ok", "service": "crypto-signals-bot"}


@api.get("/health")
async def health():
    return {"ok": True}


@api.post("/webhook/whop")
async def whop_webhook(
    request: Request,
    db: Session = Depends(get_db),
    x_whop_signature: Optional[str] = Header(None)
):
    body = await request.body()
    
    if settings.WHOP_WEBHOOK_SECRET:
        if not x_whop_signature:
            raise HTTPException(status_code=401, detail="Missing signature")
        
        expected_signature = hmac.new(
            settings.WHOP_WEBHOOK_SECRET.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(x_whop_signature, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        data = json.loads(body)
        
        telegram_id = data.get("telegram_id") or data.get("metadata", {}).get("telegram_id")
        if not telegram_id:
            raise HTTPException(status_code=400, detail="telegram_id required")
        
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        if not user:
            user = User(telegram_id=str(telegram_id))
            db.add(user)
            db.flush()
        
        now = datetime.utcnow()
        start_from = max(now, user.subscription_end) if user.subscription_end else now
        subscription_end = start_from + timedelta(days=30)
        user.subscription_end = subscription_end
        
        subscription = Subscription(
            user_id=user.id,
            payment_method="whop",
            transaction_id=data.get("payment_id") or data.get("id"),
            amount=data.get("amount", settings.SUB_PRICE_USDC),
            duration_days=30
        )
        db.add(subscription)
        db.commit()
        
        return {"ok": True, "subscription_end": subscription_end.isoformat()}
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/webhook/solana")
async def solana_webhook(
    request: Request,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)
):
    if settings.HELIUS_WEBHOOK_SECRET:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization")
        
        if authorization != settings.HELIUS_WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid authorization")
    
    try:
        data = await request.json()
        
        for tx in data:
            transaction_type = tx.get("type")
            
            if transaction_type != "TRANSFER":
                continue
            
            token_transfers = tx.get("tokenTransfers", [])
            for transfer in token_transfers:
                if transfer.get("mint") != settings.SPL_USDC_MINT:
                    continue
                
                to_address = transfer.get("toUserAccount")
                if to_address != settings.SOL_MERCHANT:
                    continue
                
                amount_raw = transfer.get("tokenAmount", 0)
                amount = amount_raw / 1_000_000
                
                if amount < settings.SUB_PRICE_USDC:
                    continue
                
                telegram_id = tx.get("description") or ""
                if not telegram_id.isdigit():
                    continue
                
                user = db.query(User).filter(User.telegram_id == telegram_id).first()
                if not user:
                    user = User(telegram_id=telegram_id)
                    db.add(user)
                    db.flush()
                
                now = datetime.utcnow()
                start_from = max(now, user.subscription_end) if user.subscription_end else now
                subscription_end = start_from + timedelta(days=30)
                user.subscription_end = subscription_end
                
                subscription = Subscription(
                    user_id=user.id,
                    payment_method="solana",
                    transaction_id=tx.get("signature"),
                    amount=amount,
                    duration_days=30
                )
                db.add(subscription)
                db.commit()
        
        return {"ok": True}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/webhooks/nowpayments")
async def nowpayments_webhook(
    request: Request,
    db: Session = Depends(get_db),
    x_nowpayments_sig: Optional[str] = Header(None, alias="x-nowpayments-sig")
):
    """
    Handle NOWPayments IPN callbacks for subscription payments
    Documentation: https://documenter.getpostman.com/view/7907941/S1a32n38#8f386065-2c5a-4c88-852b-5a470ea59d2e
    """
    body = await request.body()
    
    # Verify signature if IPN secret is configured
    if settings.NOWPAYMENTS_IPN_SECRET and x_nowpayments_sig:
        expected_signature = hmac.new(
            settings.NOWPAYMENTS_IPN_SECRET.encode(),
            body,
            hashlib.sha512
        ).hexdigest()
        
        if not hmac.compare_digest(x_nowpayments_sig, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        data = json.loads(body)
        
        payment_status = data.get("payment_status")
        order_id = data.get("order_id", "")
        
        # Only process finished/confirmed payments
        if payment_status not in ["finished", "confirmed"]:
            return {"ok": True, "message": f"Payment status {payment_status} - waiting for confirmation"}
        
        # Extract telegram_id and plan type from order_id (format: sub_{plan_type}_{telegram_id}_{timestamp})
        if not order_id.startswith("sub_"):
            raise HTTPException(status_code=400, detail="Invalid order_id format")
        
        parts = order_id.split("_")
        if len(parts) < 4:
            raise HTTPException(status_code=400, detail="Cannot extract details from order_id")
        
        plan_type = parts[1]  # "manual" or "auto"
        telegram_id = parts[2]
        
        # Find user
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User {telegram_id} not found")
        
        # Grant 30 days subscription and set plan type
        now = datetime.utcnow()
        start_from = max(now, user.subscription_end) if user.subscription_end else now
        subscription_end = start_from + timedelta(days=30)
        user.subscription_end = subscription_end
        user.subscription_type = plan_type  # Set "manual" or "auto"
        
        # Record the subscription payment
        subscription = Subscription(
            user_id=user.id,
            payment_method="nowpayments",
            transaction_id=data.get("payment_id", order_id),
            amount=data.get("price_amount", settings.SUBSCRIPTION_PRICE_USD),
            duration_days=30
        )
        db.add(subscription)
        db.commit()
        
        # Notify user via Telegram
        try:
            from app.services.bot import bot
            await bot.send_message(
                telegram_id,
                f"✅ <b>Payment Confirmed!</b>\n\n"
                f"Your premium subscription is now <b>active</b> until:\n"
                f"📅 <b>{subscription_end.strftime('%Y-%m-%d')}</b>\n\n"
                f"You now have full access to:\n"
                f"✅ 1:1 Day Trading Signals\n"
                f"✅ Top Gainers Scanner (24/7)\n"
                f"✅ Auto-Trading on Bitunix\n"
                f"✅ Advanced Analytics\n\n"
                f"Use /dashboard to get started!",
                parse_mode="HTML"
            )
        except Exception as e:
            # Log but don't fail the webhook if notification fails
            import logging
            logging.error(f"Failed to send subscription confirmation to {telegram_id}: {e}")
        
        return {"ok": True, "subscription_end": subscription_end.isoformat()}
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
