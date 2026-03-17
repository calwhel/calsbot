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
    """
    Decrypt an API key for use.
    Handles two storage formats gracefully:
      1. Fernet-encrypted (starts with 'gAAAAA', len > 80) — decrypt normally.
      2. Raw/plaintext (stored before encryption was introduced, len < 80) — return as-is
         and re-encrypt it in the DB on next save so future reads work correctly.
    """
    if not encrypted_key:
        logger.warning("⚠️ decrypt_api_key called with empty key!")
        return ""

    # Fernet tokens always start with 'gAAAAA' and are base64url so only contain
    # A-Z a-z 0-9 + - _ =. If the stored value doesn't look like a Fernet token,
    # treat it as a plaintext key that was stored before encryption was enabled.
    looks_like_fernet = encrypted_key.startswith("gAAAAA") and len(encrypted_key) > 80
    if not looks_like_fernet:
        logger.warning(
            f"⚠️ API key appears to be stored as plaintext (len={len(encrypted_key)}) — "
            f"using as-is. Re-save API keys in the bot to encrypt them."
        )
        return encrypted_key

    try:
        decrypted = cipher.decrypt(encrypted_key.encode())
        result = decrypted.decode()
        if len(result) > 8:
            logger.debug(f"✅ Decrypted key: {result[:4]}...{result[-4:]} (len={len(result)})")
        return result
    except Exception as e:
        logger.error(f"❌ Decryption FAILED! Error: {e}")
        logger.error(f"   → Encrypted key preview: {encrypted_key[:20]}... (len={len(encrypted_key)})")
        logger.error(f"   → This usually means ENCRYPTION_KEY changed or is missing!")
        # Last-resort fallback: if the key is short enough to be a raw API key, return it
        if len(encrypted_key) < 100:
            logger.warning("⚠️ Returning raw value as last-resort fallback (key may be unencrypted)")
            return encrypted_key
        raise
