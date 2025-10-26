import requests
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NOWPaymentsService:
    """
    NOWPayments API integration for crypto subscriptions
    Supports BTC, ETH, USDT, and 200+ cryptocurrencies
    """
    
    def __init__(self, api_key: str, sandbox: bool = False):
        self.api_key = api_key
        base = "api-sandbox" if sandbox else "api"
        self.base_url = f"https://{base}.nowpayments.io/v1"
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
    
    def check_status(self) -> bool:
        """Check if API is working"""
        try:
            url = f"{self.base_url}/status"
            response = requests.get(url, headers=self.headers)
            return response.status_code == 200 and response.json().get("message") == "OK"
        except Exception as e:
            logger.error(f"NOWPayments status check failed: {e}")
            return False
    
    def get_available_currencies(self) -> list:
        """Get list of available cryptocurrencies"""
        try:
            url = f"{self.base_url}/currencies"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.json().get("currencies", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get currencies: {e}")
            return []
    
    def create_subscription_plan(
        self,
        title: str,
        amount: float,
        currency: str = "usd",
        interval_days: int = 30,
        ipn_callback_url: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create a subscription plan
        
        Args:
            title: Plan name (e.g., "Premium Monthly")
            amount: Price amount (e.g., 29.99)
            currency: Currency code (usd, eur, btc, etc.)
            interval_days: Billing interval (30 = monthly, 7 = weekly)
            ipn_callback_url: Webhook URL for payment notifications
        
        Returns:
            Plan data with plan_id, or None if failed
        """
        try:
            url = f"{self.base_url}/subscriptions/plans"
            data = {
                "title": title,
                "interval_day": interval_days,
                "amount": amount,
                "currency": currency.lower()
            }
            
            if ipn_callback_url:
                data["ipn_callback_url"] = ipn_callback_url
            
            response = requests.post(url, json=data, headers=self.headers)
            
            if response.status_code == 200:
                plan = response.json()
                logger.info(f"Created subscription plan: {plan.get('id')} - {title}")
                return plan
            else:
                logger.error(f"Failed to create plan: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating subscription plan: {e}")
            return None
    
    def subscribe_user(self, plan_id: str, email: str) -> Optional[Dict]:
        """
        Subscribe a user to a plan
        
        Args:
            plan_id: Subscription plan ID
            email: User's email address
        
        Returns:
            Subscription data with subscription_id and payment_link
        """
        try:
            url = f"{self.base_url}/subscriptions"
            data = {
                "plan_id": plan_id,
                "email": email
            }
            
            response = requests.post(url, json=data, headers=self.headers)
            
            if response.status_code == 200:
                subscription = response.json()
                logger.info(f"Created subscription for {email}: {subscription.get('id')}")
                return subscription
            else:
                logger.error(f"Failed to subscribe user: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error subscribing user: {e}")
            return None
    
    def get_subscription(self, subscription_id: str) -> Optional[Dict]:
        """Get subscription details"""
        try:
            url = f"{self.base_url}/subscriptions/{subscription_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Error getting subscription: {e}")
            return None
    
    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription"""
        try:
            url = f"{self.base_url}/subscriptions/{subscription_id}"
            response = requests.delete(url, headers=self.headers)
            
            success = response.status_code == 200
            if success:
                logger.info(f"Cancelled subscription: {subscription_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling subscription: {e}")
            return False
    
    def create_one_time_payment(
        self,
        price_amount: float,
        price_currency: str,
        order_id: str,
        ipn_callback_url: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create a one-time payment invoice (for initial subscription payment)
        
        Args:
            price_amount: Amount to charge
            price_currency: Currency (usd, eur, btc, etc.)
            order_id: Unique order identifier
            ipn_callback_url: Webhook URL for payment status
        
        Returns:
            Invoice data with payment_id and invoice_url
        """
        try:
            url = f"{self.base_url}/invoice"
            data = {
                "price_amount": price_amount,
                "price_currency": price_currency.lower(),
                "order_id": order_id,
                "order_description": "Trading Bot Subscription"
            }
            
            if ipn_callback_url:
                data["ipn_callback_url"] = ipn_callback_url
            
            response = requests.post(url, json=data, headers=self.headers)
            
            if response.status_code == 200:
                invoice = response.json()
                logger.info(f"Created invoice: {invoice.get('id')} for order {order_id}")
                return invoice
            else:
                logger.error(f"Failed to create invoice: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating invoice: {e}")
            return None
    
    def get_payment_status(self, payment_id: str) -> Optional[Dict]:
        """Check status of a payment"""
        try:
            url = f"{self.base_url}/payment/{payment_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Error getting payment status: {e}")
            return None
