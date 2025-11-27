from cryptography.fernet import Fernet
from app.config import settings
import logging
import hashlib

logger = logging.getLogger(__name__)


def get_encryption_key() -> bytes:
    """Get or generate encryption key for API credentials"""
    if hasattr(settings, 'ENCRYPTION_KEY') and settings.ENCRYPTION_KEY:
        key = settings.ENCRYPTION_KEY.encode()
        # Log a fingerprint of the key (NOT the key itself) for debugging
        fingerprint = hashlib.sha256(key).hexdigest()[:8]
        logger.info(f"🔑 ENCRYPTION_KEY loaded (fingerprint: {fingerprint})")
        return key
    
    # CRITICAL: In production, we MUST have the encryption key set
    logger.error("❌ ENCRYPTION_KEY not set! API key encryption/decryption will FAIL!")
    raise RuntimeError("ENCRYPTION_KEY environment variable is required but not set!")


cipher = Fernet(get_encryption_key())


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for secure storage"""
    if not api_key:
        return ""
    
    try:
        encrypted = cipher.encrypt(api_key.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key for use"""
    if not encrypted_key:
        logger.warning("⚠️ decrypt_api_key called with empty key!")
        return ""
    
    try:
        decrypted = cipher.decrypt(encrypted_key.encode())
        result = decrypted.decode()
        # Log success with preview (first/last 4 chars only)
        if len(result) > 8:
            logger.debug(f"✅ Decrypted key: {result[:4]}...{result[-4:]} (len={len(result)})")
        return result
    except Exception as e:
        logger.error(f"❌ Decryption FAILED! Error: {e}")
        logger.error(f"   → Encrypted key preview: {encrypted_key[:20]}... (len={len(encrypted_key)})")
        logger.error(f"   → This usually means ENCRYPTION_KEY changed or is missing!")
        raise
