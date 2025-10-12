from cryptography.fernet import Fernet
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def get_encryption_key() -> bytes:
    """Get or generate encryption key for API credentials"""
    if hasattr(settings, 'ENCRYPTION_KEY') and settings.ENCRYPTION_KEY:
        return settings.ENCRYPTION_KEY.encode()
    
    # Generate a key if not set (for development)
    # IMPORTANT: In production, this should be set as an environment variable
    key = Fernet.generate_key()
    logger.warning("Using generated encryption key - set ENCRYPTION_KEY environment variable in production!")
    return key


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
        return ""
    
    try:
        decrypted = cipher.decrypt(encrypted_key.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        raise
