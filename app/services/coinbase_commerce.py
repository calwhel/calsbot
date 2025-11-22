import requests
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CoinbaseCommerceService:
    """
    Coinbase Commerce API integration for crypto subscriptions
    Supports 200+ cryptocurrencies with built-in webhook support
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.commerce.coinbase.com"
        self.headers = {
            "X-CC-Api-Key": api_key,
            "X-CC-Version": "2018-03-22",
            "Content-Type": "application/json"
        }
    
    def check_status(self) -> bool:
        """Check if API is working"""
        try:
            url = f"{self.base_url}/charges"
            response = requests.get(url, headers=self.headers)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Coinbase Commerce status check failed: {e}")
            return False
    
    def create_charge(
        self,
        amount: float,
        currency: str = "USD",
        description: str = "Trading Bot Subscription",
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Create a one-time charge (payment)
        
        Args:
            amount: Amount to charge (e.g., 130.00 for $130)
            currency: Currency code (USD, EUR, etc.)
            description: Payment description
            metadata: Custom metadata (user_id, plan, etc.)
        
        Returns:
            Charge data with charge_id and hosted_url for payment
        """
        try:
            url = f"{self.base_url}/charges"
            data = {
                "name": description,
                "description": description,
                "pricing_type": "fixed_price",
                "local_price": {
                    "amount": str(amount),
                    "currency": currency.upper()
                },
                "metadata": metadata or {}
            }
            
            response = requests.post(url, json=data, headers=self.headers)
            
            if response.status_code == 201:
                charge = response.json().get("data", {})
                logger.info(f"Created charge: {charge.get('id')} for ${amount} {currency}")
                return charge
            else:
                logger.error(f"Failed to create charge: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating charge: {e}")
            return None
    
    def get_charge(self, charge_id: str) -> Optional[Dict]:
        """Get charge details"""
        try:
            url = f"{self.base_url}/charges/{charge_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json().get("data", {})
            return None
            
        except Exception as e:
            logger.error(f"Error getting charge: {e}")
            return None
    
    def cancel_charge(self, charge_id: str) -> bool:
        """Cancel a charge"""
        try:
            url = f"{self.base_url}/charges/{charge_id}/cancel"
            response = requests.post(url, headers=self.headers)
            
            success = response.status_code == 200
            if success:
                logger.info(f"Cancelled charge: {charge_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling charge: {e}")
            return False
    
    def verify_webhook_signature(self, signature: str, body: bytes, webhook_secret: str) -> bool:
        """
        Verify Coinbase Commerce webhook signature
        
        Args:
            signature: X-CC-Webhook-Signature header value
            body: Raw request body
            webhook_secret: Webhook secret from Coinbase
        
        Returns:
            True if signature is valid
        """
        try:
            import hmac
            import hashlib
            
            expected_sig = hmac.new(
                webhook_secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_sig)
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False
