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


@api.post("/webhooks/oxapay")
async def oxapay_webhook(
    request: Request,
    db: Session = Depends(get_db),
    hmac: Optional[str] = Header(None, alias="HMAC")
):
    """
    Handle OxaPay webhooks for subscription payments
    Documentation: https://oxapay.com/docs
    """
    import logging
    logger = logging.getLogger(__name__)
    
    body = await request.body()
    body_str = body.decode() if isinstance(body, bytes) else body
    logger.info(f"🔔 OxaPay webhook received: {body_str[:200]}")
    
    # Verify webhook signature using MERCHANT_API_KEY
    if settings.OXAPAY_MERCHANT_API_KEY and hmac:
        from app.services.oxapay import OxaPayService
        oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
        
        if not oxapay.verify_webhook_signature(hmac, body_str, settings.OXAPAY_MERCHANT_API_KEY):
            logger.error("❌ Invalid OxaPay webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        data = json.loads(body_str)
        event_type = data.get("type", "")
        invoice_data = data.get("data", {})
        
        logger.info(f"✅ Event type: {event_type}")
        
        # Only process confirmed payments
        if event_type not in ["payment.confirmed", "invoice.confirmed"]:
            logger.info(f"⏳ Skipping event - type {event_type}")
            return {"ok": True, "message": f"Event {event_type} - not a confirmed payment"}
        
        # Extract invoice ID and metadata
        invoice_id = invoice_data.get("invoiceID") or invoice_data.get("id") or ""
        status = invoice_data.get("status", "").lower()
        
        # Look for telegram_id in metadata (OxaPay stores this in custom fields)
        metadata_str = invoice_data.get("metadata", "{}")
        try:
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        except:
            metadata = {}
        
        telegram_id = metadata.get("telegram_id") or invoice_data.get("telegramId") or ""
        plan_type = metadata.get("plan_type") or "auto"
        
        logger.info(f"✅ Invoice confirmed: {invoice_id}, Status: {status}, User: {telegram_id}")
        
        # Only process if status is "Paid"
        if status != "paid":
            logger.info(f"⏳ Invoice status is {status}, not yet paid")
            return {"ok": True, "message": f"Invoice status: {status}"}
        
        # Validate
        if not telegram_id:
            logger.error(f"❌ Missing telegram_id in metadata")
            raise HTTPException(status_code=400, detail="Cannot extract telegram_id")
        
        if plan_type not in ["scan", "manual", "auto"]:
            logger.warning(f"⚠️ Unknown plan type: {plan_type}, defaulting to auto")
            plan_type = "auto"
        
        # Find user
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User {telegram_id} not found")
        
        # Grant 30 days subscription and set plan type
        now = datetime.utcnow()
        start_from = max(now, user.subscription_end) if user.subscription_end else now
        subscription_end = start_from + timedelta(days=30)
        user.subscription_end = subscription_end
        user.subscription_type = plan_type  # Set "scan", "manual", or "auto"
        
        # Record the subscription payment
        charge_amount = settings.SUBSCRIPTION_PRICE_USD
        if invoice_data.get("amount"):
            try:
                charge_amount = float(invoice_data["amount"])
            except:
                pass
        
        subscription = Subscription(
            user_id=user.id,
            payment_method="oxapay",
            transaction_id=invoice_id,
            amount=charge_amount,
            duration_days=30
        )
        db.add(subscription)
        db.commit()  # COMMIT BEFORE SENDING NOTIFICATIONS
        
        # ✅ Send admin notification via direct Telegram API calls
        try:
            import httpx
            admins = db.query(User).filter(User.is_admin == True).all()
            logger.info(f"🔔 Sending subscription notification to {len(admins)} admin(s)")
            
            user_info = f"@{user.username}" if user.username else f"{user.first_name} (ID: {user.telegram_id})"
            if plan_type == "auto":
                plan_name = "🤖 Auto-Trading"
                plan_price = "$130/mo"
            elif plan_type == "manual":
                plan_name = "💎 Signals Only"
                plan_price = "$65/mo"
            else:
                plan_name = "📊 AI Assistant"
                plan_price = "$65/mo"
            
            referred_info = ""
            if user.referred_by:
                referrer = db.query(User).filter(User.referral_code == user.referred_by).first()
                if referrer:
                    referrer_name = f"@{referrer.username}" if referrer.username else referrer.first_name
                    referred_info = f"\n👥 <b>Referred by:</b> {referrer_name} (+$30 reward)"
            
            notification_text = (
                f"✅ <b>NEW SUBSCRIPTION!</b>\n\n"
                f"<b>User:</b> {user_info}\n"
                f"<b>Plan:</b> {plan_name} ({plan_price})\n"
                f"<b>Expires:</b> {subscription_end.strftime('%Y-%m-%d')}"
                f"{referred_info}"
            )
            
            tg_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            for admin in admins:
                try:
                    payload = {
                        "chat_id": int(admin.telegram_id),
                        "text": notification_text,
                        "parse_mode": "HTML"
                    }
                    with httpx.Client() as client:
                        response = client.post(tg_url, json=payload, timeout=10)
                        if response.status_code == 200:
                            logger.info(f"✅ Sent subscription notification to admin {admin.telegram_id}")
                        else:
                            logger.error(f"❌ Telegram API error for admin {admin.telegram_id}: {response.status_code}")
                except Exception as e:
                    logger.error(f"❌ Failed to send to admin {admin.telegram_id}: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to send admin notifications: {e}")
        
        # Process referral rewards - $30 cash for Auto-Trading subscriptions only
        if user.referred_by and plan_type == "auto":
            referrer = db.query(User).filter(User.referral_code == user.referred_by).first()
            if referrer:
                # Check if this referral has already been paid (prevent duplicates)
                import json as json_lib
                paid_list = json_lib.loads(referrer.paid_referrals) if referrer.paid_referrals else []
                
                if user.id not in paid_list:
                    # Add $30 to pending earnings
                    referrer.referral_earnings = (referrer.referral_earnings or 0.0) + 30.0
                    
                    # Notify referrer about pending $30 reward via direct HTTP
                    try:
                        ref_name = user.username if user.username else user.first_name or "Someone"
                        
                        # Check if wallet is set
                        wallet_reminder = ""
                        if not referrer.crypto_wallet:
                            wallet_reminder = "\n\n⚠️ <b>Action Required:</b> Set your wallet address to receive payment!\nUse: /setwallet [your_address]"
                        
                        payload = {
                            "chat_id": int(referrer.telegram_id),
                            "text": (
                                f"💰 <b>$30 Referral Reward Pending!</b>\n\n"
                                f"@{ref_name} just subscribed to <b>Auto-Trading ($130/mo)</b> using your referral link!\n\n"
                                f"🎁 <b>+$30 USD</b> will be sent to you via crypto!\n"
                                f"💵 <b>Total Pending:</b> ${referrer.referral_earnings:.2f}"
                                f"{wallet_reminder}\n\n"
                                f"<i>Keep sharing to earn more!</i> 🚀"
                            ),
                            "parse_mode": "HTML"
                        }
                        with httpx.Client() as client:
                            response = client.post(tg_url, json=payload, timeout=10)
                            if response.status_code == 200:
                                logger.info(f"✅ Sent referral notification to {referrer.telegram_id}")
                            else:
                                logger.error(f"❌ Telegram API error for referrer {referrer.telegram_id}: {response.status_code}")
                    except Exception as e:
                        logger.error(f"Failed to notify referrer {referrer.telegram_id}: {e}")
                    
                    # Send admin notification about pending payout via direct HTTP
                    try:
                        ref_username = f"@{referrer.username}" if referrer.username else f"{referrer.first_name} (ID: {referrer.telegram_id})"
                        new_sub_username = f"@{user.username}" if user.username else f"{user.first_name} (ID: {user.telegram_id})"
                        wallet_info = f"<b>Wallet:</b> <code>{referrer.crypto_wallet}</code>" if referrer.crypto_wallet else "⚠️ <b>Wallet:</b> <i>Not set yet</i>"
                        
                        for admin in admins:
                            try:
                                payload = {
                                    "chat_id": int(admin.telegram_id),
                                    "text": (
                                        f"🎁 <b>NEW REFERRAL PAYOUT PENDING!</b>\n\n"
                                        f"<b>Referrer:</b> {ref_username}\n"
                                        f"<b>Referrer ID:</b> <code>{referrer.telegram_id}</code>\n"
                                        f"{wallet_info}\n"
                                        f"<b>New Subscriber:</b> {new_sub_username}\n"
                                        f"<b>Subscription Tier:</b> 🤖 Auto-Trading ($130/mo)\n"
                                        f"<b>Reward:</b> $30 USD\n\n"
                                        f"💰 <b>Referrer's Total Pending:</b> ${referrer.referral_earnings:.2f}\n\n"
                                        f"<i>Use /pending_payouts to view all pending payouts</i>"
                                    ),
                                    "parse_mode": "HTML"
                                }
                                with httpx.Client() as client:
                                    response = client.post(tg_url, json=payload, timeout=10)
                                    if response.status_code == 200:
                                        logger.info(f"✅ Sent referral payout notification to admin {admin.telegram_id}")
                                    else:
                                        logger.error(f"❌ Telegram API error for admin {admin.telegram_id}: {response.status_code}")
                            except Exception as e:
                                logger.error(f"Failed to notify admin {admin.telegram_id}: {e}")
                    except Exception as e:
                        logger.error(f"Failed to send admin notification: {e}")
        
        db.commit()
        
        # Notify user via direct HTTP with tier-specific features
        try:
            # Tier-specific feature lists
            if plan_type == "scan":
                features = "✅ Top Gainers scanner\n✅ Volume surge detection\n✅ New coin alerts"
                plan_name = "📊 Scan Mode"
            elif plan_type == "manual":
                features = "✅ Manual signal notifications\n✅ Top Gainers scanner\n✅ LONGS + SHORTS strategies\n✅ PnL tracking"
                plan_name = "💎 Manual Signals"
            else:  # auto
                features = "✅ Automated 24/7 execution\n✅ All Manual Signals features\n✅ Auto-Trading on Bitunix\n✅ Advanced risk management"
                plan_name = "🤖 Auto-Trading"
            
            payload = {
                "chat_id": int(telegram_id),
                "text": (
                    f"✅ <b>Payment Confirmed!</b>\n\n"
                    f"Your <b>{plan_name}</b> subscription is now active until:\n"
                    f"📅 <b>{subscription_end.strftime('%Y-%m-%d')}</b>\n\n"
                    f"You now have access to:\n"
                    f"{features}\n\n"
                    f"Use /dashboard to get started!"
                ),
                "parse_mode": "HTML"
            }
            with httpx.Client() as client:
                response = client.post(tg_url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Sent subscription confirmation to user {telegram_id}")
                else:
                    logger.error(f"❌ Telegram API error for user {telegram_id}: {response.status_code}")
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
