import requests
import logging
from typing import Dict, Optional
import json

logger = logging.getLogger(__name__)


class OxaPayService:
    """
    OxaPay API integration for crypto subscriptions
    Supports BTC, ETH, USDT, USDC with webhook support
    """
    
    def __init__(self, merchant_api_key: str):
        self.api_key = merchant_api_key
        self.base_url = "https://api.oxapay.com/merchants"
        self.headers = {
            "Authorization": f"Bearer {merchant_api_key}",
            "Content-Type": "application/json"
        }
    
    def check_status(self) -> bool:
        """Check if API is working"""
        try:
            url = f"{self.base_url}/info"
            response = requests.get(url, headers=self.headers)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"OxaPay status check failed: {e}")
            return False
    
    def create_invoice(
        self,
        amount: float,
        currency: str = "USD",
        description: str = "Trading Bot Subscription",
        metadata: Optional[Dict] = None,
        order_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create an invoice for payment
        
        Args:
            amount: Amount to charge (e.g., 130.00 for $130)
            currency: Currency code (USD)
            description: Payment description
            metadata: Custom metadata (user_id, plan, etc.)
            order_id: Unique order ID
        
        Returns:
            Invoice data with invoice_id and payment_url
        """
        try:
            url = f"{self.base_url}/invoices/create"
            
            data = {
                "amount": str(amount),
                "currency": currency.upper(),
                "description": description,
                "orderID": order_id or "",
                "callbackUrl": "",  # Webhook will be set separately
                "returnUrl": ""
            }
            
            # Add metadata if provided
            if metadata:
                data["metadata"] = json.dumps(metadata)
            
            response = requests.post(url, json=data, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    invoice = result.get("data", {})
                    logger.info(f"Created OxaPay invoice: {invoice.get('invoiceID')} for ${amount} {currency}")
                    return invoice
                else:
                    logger.error(f"OxaPay error: {result.get('message')}")
                    return None
            else:
                logger.error(f"Failed to create invoice: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating OxaPay invoice: {e}")
            return None
    
    def get_invoice(self, invoice_id: str) -> Optional[Dict]:
        """Get invoice details"""
        try:
            url = f"{self.base_url}/invoices/{invoice_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    return result.get("data", {})
            return None
            
        except Exception as e:
            logger.error(f"Error getting invoice: {e}")
            return None
    
    def verify_webhook_signature(self, signature: str, body: str, webhook_secret: str) -> bool:
        """
        Verify OxaPay webhook signature
        
        Args:
            signature: X-OXA-SIGNATURE header value (hex string)
            body: Raw request body (JSON string)
            webhook_secret: Webhook secret from OxaPay dashboard
        
        Returns:
            True if signature is valid
        """
        try:
            import hmac
            import hashlib
            
            # OxaPay uses SHA-256 HMAC
            expected_sig = hmac.new(
                webhook_secret.encode(),
                body.encode() if isinstance(body, str) else body,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_sig)
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False
