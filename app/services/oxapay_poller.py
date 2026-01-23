import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import PendingInvoice, User, Subscription
from app.services.oxapay import OxaPayService
from app.config import settings
import httpx

logger = logging.getLogger(__name__)


async def activate_subscription_from_invoice(
    db: Session,
    invoice: PendingInvoice,
    tg_url: str
) -> bool:
    """
    Activate subscription when invoice is paid
    Returns True if successful
    """
    try:
        user = db.query(User).filter(User.id == invoice.user_id).first()
        if not user:
            logger.error(f"User {invoice.user_id} not found for invoice {invoice.track_id}")
            return False
        
        # Grant 30 days subscription
        now = datetime.utcnow()
        start_from = max(now, user.subscription_end) if user.subscription_end else now
        subscription_end = start_from + timedelta(days=30)
        user.subscription_end = subscription_end
        user.subscription_type = invoice.plan_type
        
        # Record subscription payment
        subscription = Subscription(
            user_id=user.id,
            payment_method="oxapay",
            transaction_id=invoice.track_id,
            amount=invoice.amount,
            duration_days=30
        )
        db.add(subscription)
        
        # Mark invoice as activated
        invoice.status = "paid"
        invoice.activated_at = now
        
        db.commit()
        
        logger.info(f"✅ Subscription activated for user {user.telegram_id} via invoice {invoice.track_id}")
        
        # Send admin notification
        try:
            admins = db.query(User).filter(User.is_admin == True).all()
            user_info = f"@{user.username}" if user.username else f"{user.first_name} (ID: {user.telegram_id})"
            if invoice.plan_type == "auto":
                plan_name = "🚀 Auto-Trading"
                plan_price = "$130/mo"
            else:
                plan_name = "🤖 AI Assistant"
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
            
            for admin in admins:
                try:
                    payload = {
                        "chat_id": int(admin.telegram_id),
                        "text": notification_text,
                        "parse_mode": "HTML"
                    }
                    async with httpx.AsyncClient() as client:
                        response = await client.post(tg_url, json=payload, timeout=10)
                        if response.status_code == 200:
                            logger.info(f"✅ Sent subscription notification to admin {admin.telegram_id}")
                except Exception as e:
                    logger.error(f"Failed to notify admin {admin.telegram_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to send admin notifications: {e}")
        
        # Process referral rewards - $30 cash for Auto-Trading subscriptions only
        if user.referred_by and invoice.plan_type == "auto":
            referrer = db.query(User).filter(User.referral_code == user.referred_by).first()
            if referrer:
                import json as json_lib
                paid_list = json_lib.loads(referrer.paid_referrals) if referrer.paid_referrals else []
                
                if user.id not in paid_list:
                    # Add $30 to pending earnings
                    referrer.referral_earnings = (referrer.referral_earnings or 0.0) + 30.0
                    
                    # Mark this user as paid so they don't get rewarded again on renewal
                    paid_list.append(user.id)
                    referrer.paid_referrals = json_lib.dumps(paid_list)
                    
                    # Notify referrer
                    try:
                        ref_name = user.username if user.username else user.first_name or "Someone"
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
                        async with httpx.AsyncClient() as client:
                            response = await client.post(tg_url, json=payload, timeout=10)
                            if response.status_code == 200:
                                logger.info(f"✅ Sent referral notification to {referrer.telegram_id}")
                    except Exception as e:
                        logger.error(f"Failed to notify referrer: {e}")
                    
                    # Notify admins about pending payout
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
                                async with httpx.AsyncClient() as client:
                                    response = await client.post(tg_url, json=payload, timeout=10)
                            except Exception as e:
                                logger.error(f"Failed to notify admin: {e}")
                    except Exception as e:
                        logger.error(f"Failed to send referral payout notifications: {e}")
        
        db.commit()
        
        # Notify user
        try:
            if invoice.plan_type == "scan":
                features = "✅ Top Gainers scanner\n✅ Volume surge detection\n✅ New coin alerts"
                plan_name = "📊 Scan Mode"
            elif invoice.plan_type == "manual":
                features = "✅ Manual signal notifications\n✅ Top Gainers scanner\n✅ LONGS + SHORTS strategies\n✅ PnL tracking"
                plan_name = "💎 Manual Signals"
            else:  # auto
                features = "✅ Automated 24/7 execution\n✅ All Manual Signals features\n✅ Auto-Trading on Bitunix\n✅ Advanced risk management"
                plan_name = "🤖 Auto-Trading"
            
            payload = {
                "chat_id": int(user.telegram_id),
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
            async with httpx.AsyncClient() as client:
                response = await client.post(tg_url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Sent subscription confirmation to user {user.telegram_id}")
        except Exception as e:
            logger.error(f"Failed to send user confirmation: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error activating subscription from invoice {invoice.track_id}: {e}")
        db.rollback()
        return False


async def poll_oxapay_payments():
    """
    Background task that polls OxaPay API every 60 seconds
    to check pending invoice statuses
    """
    logger.info("🔄 Starting OxaPay payment poller...")
    
    if not settings.OXAPAY_MERCHANT_API_KEY:
        logger.warning("⚠️ OXAPAY_MERCHANT_API_KEY not set - poller disabled")
        return
    
    oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
    tg_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    
    while True:
        try:
            db = SessionLocal()
            
            try:
                # Get all pending invoices (not expired)
                pending_invoices = db.query(PendingInvoice).filter(
                    PendingInvoice.status == "pending"
                ).all()
                
                logger.info(f"📋 Checking {len(pending_invoices)} pending OxaPay invoices...")
                
                for invoice in pending_invoices:
                    try:
                        # Check if expired (>2 hours old)
                        if invoice.is_expired:
                            logger.info(f"⏰ Invoice {invoice.track_id} expired - marking as expired")
                            invoice.status = "expired"
                            db.commit()
                            continue
                        
                        # Check payment status via OxaPay API
                        result = oxapay.check_payment_status(invoice.track_id)
                        
                        if result:
                            status = result.get("status", "").lower()
                            logger.info(f"📊 Invoice {invoice.track_id}: status={status}")
                            
                            if status == "paid":
                                logger.info(f"💰 Payment confirmed for invoice {invoice.track_id}!")
                                # Activate subscription
                                await activate_subscription_from_invoice(db, invoice, tg_url)
                            elif status == "expired":
                                invoice.status = "expired"
                                db.commit()
                            elif status == "canceled":
                                invoice.status = "failed"
                                db.commit()
                        else:
                            logger.warning(f"⚠️ Could not check status for invoice {invoice.track_id}")
                    
                    except Exception as e:
                        logger.error(f"Error checking invoice {invoice.track_id}: {e}")
                        continue
                
            finally:
                db.close()
            
            # Wait 60 seconds before next check
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in OxaPay poller loop: {e}")
            await asyncio.sleep(60)  # Wait before retrying
