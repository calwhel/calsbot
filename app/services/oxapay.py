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
        self.base_url = "https://api.oxapay.com"
        self.headers = {
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
        order_id: Optional[str] = None,
        callback_url: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create an invoice for payment
        
        Args:
            amount: Amount to charge (e.g., 130.00 for $130)
            currency: Currency code (USD)
            description: Payment description
            metadata: Custom metadata (user_id, plan, etc.)
            order_id: Unique order ID
            callback_url: Webhook URL for payment notifications
        
        Returns:
            Invoice data with payLink and trackId
        """
        try:
            url = f"{self.base_url}/merchants/request"
            
            data = {
                "merchant": self.api_key,
                "amount": str(amount),
                "currency": currency.upper(),
                "description": description,
                "orderId": order_id or "",
                "callbackUrl": callback_url or "",
                "lifeTime": 60  # 60 minutes expiration
            }
            
            # Add metadata if provided - OxaPay stores this in custom fields
            if metadata:
                data["metadata"] = json.dumps(metadata)
            
            logger.info(f"Creating OxaPay invoice: amount={amount}, order_id={order_id}")
            response = requests.post(url, json=data, headers=self.headers, timeout=10)
            
            result = response.json()
            logger.info(f"OxaPay response: {result}")
            
            if response.status_code == 200 and result.get("result") == 100:
                # Success response includes payLink and trackId directly
                invoice = {
                    "payLink": result.get("payLink"),
                    "trackId": result.get("trackId"),
                    "amount": amount,
                    "orderId": order_id
                }
                logger.info(f"Created OxaPay invoice: {result.get('trackId')} for ${amount} {currency}")
                return invoice
            else:
                error_msg = result.get("message", f"HTTP {response.status_code}")
                logger.error(f"OxaPay error: {error_msg}")
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
    
    def verify_webhook_signature(self, signature: str, body: str, api_key: str) -> bool:
        """
        Verify OxaPay webhook signature
        
        Args:
            signature: HMAC header value (hex string)
            body: Raw request body (JSON string)
            api_key: MERCHANT_API_KEY used to sign
        
        Returns:
            True if signature is valid
        """
        try:
            import hmac
            import hashlib
            
            # OxaPay uses SHA-512 HMAC with MERCHANT_API_KEY
            expected_sig = hmac.new(
                api_key.encode(),
                body.encode() if isinstance(body, str) else body,
                hashlib.sha512
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_sig)
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False
