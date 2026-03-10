import os
import base64
import logging
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Static salt for simplicity in this context (used for KDF)
SALT = b"thumber-trader-v2-salt-long-string-for-security"

def _get_key_from_env() -> Optional[bytes]:
    """Derive a 32-byte key from APP_ENV_ENC_KEY or auto-generate one in .enc_key."""
    key_str = os.getenv("APP_ENV_ENC_KEY")
    
    if not key_str:
        # Fallback to local .enc_key file
        enc_file = Path(".enc_key")
        if enc_file.exists():
            key_str = enc_file.read_text().strip()
        else:
            # Generate a new random 32-byte hex string (64 chars) and save it
            key_str = os.urandom(32).hex()
            enc_file.write_text(key_str + "\n")
            # Set restrictive permissions (rw-------)
            try:
                os.chmod(enc_file, 0o600)
            except Exception as exc:
                logging.debug(f"Unable to set 0600 permissions on .enc_key: {exc}")
            logging.info("Generated new local encryption key in .enc_key")
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    return kdf.derive(key_str.encode())

def _get_fernet():
    """Get a Fernet instance using derived key (legacy)."""
    key_bytes = _get_key_from_env()
    if not key_bytes:
        return None
    # Fernet requires url-safe base64 32-byte key
    key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(key)

def _should_encrypt_key(key: str) -> bool:
    """Check if a key is sensitive and should be encrypted."""
    sensitive_keys = {
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "DISCORD_WEBHOOK_URL",
        "PAGERDUTY_ROUTING_KEY",
        "SLACK_WEBHOOK_URL"
    }
    return key.upper() in sensitive_keys

def encrypt_aes(value: str) -> str:
    """Encrypt a value using AES-256-GCM."""
    key_bytes = _get_key_from_env()
    if not key_bytes or not value:
        return value
    
    aesgcm = AESGCM(key_bytes)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, value.encode(), None)
    
    # Store as base64(nonce + ciphertext)
    result = base64.b64encode(nonce + ciphertext).decode()
    return f"AES:{result}"

def decrypt_aes(value: str) -> str:
    """Decrypt a value using AES-256-GCM."""
    if not value.startswith("AES:"):
        return value
        
    key_bytes = _get_key_from_env()
    if not key_bytes:
        logging.warning("Encryption key not found, cannot decrypt AES value.")
        return value
        
    try:
        data = base64.b64decode(value[4:])
        nonce = data[0:12]
        ciphertext = data[12:]
        aesgcm = AESGCM(key_bytes)
        return aesgcm.decrypt(nonce, ciphertext, None).decode()
    except Exception as e:
        logging.error(f"Failed to decrypt AES value: {e}")
        return value

def encrypt_value(value: str) -> str:
    """Encrypt a value using the latest AES-256-GCM method."""
    if not value or value.startswith("AES:") or value.startswith("ENC:"):
        return value
    return encrypt_aes(value)

def decrypt_value(value: str) -> str:
    """Decrypt a value, supporting both legacy Fernet (ENC:) and AES-GCM (AES:)."""
    if not isinstance(value, str):
        return value
        
    if value.startswith("AES:"):
        return decrypt_aes(value)
        
    if value.startswith("ENC:"):
        fernet = _get_fernet()
        if not fernet:
            return value
        try:
            return fernet.decrypt(value[4:].encode()).decode()
        except Exception:
            return value
            
    return value

def update_env_file(updates: dict):
    """
    Update multiple keys in the .env file.
    If a key exists, replace it. If not, append it.
    """
    env_path = Path(".env")
    if not env_path.exists():
        env_path.touch()

    lines = env_path.read_text().splitlines()
    new_lines = []
    processed_keys = set()

    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            new_lines.append(line)
            continue
        
        # Simple parsing key=value
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            
            if key in updates:
                new_value = updates[key]
                
                # Encrypt if sensitive
                if _should_encrypt_key(key):
                    new_value = encrypt_value(new_value)
                
                # Escape newlines to keep .env file strictly 1 line per key
                if isinstance(new_value, str):
                    new_value = new_value.replace("\r\n", "\\n").replace("\n", "\\n")
                
                # Standardize on key="value" for updated fields
                new_lines.append(f'{key}="{new_value}"')
                processed_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append new keys
    for key, value in updates.items():
        if key not in processed_keys:
            if _should_encrypt_key(key):
                value = encrypt_value(value)
                
            if isinstance(value, str):
                value = value.replace("\r\n", "\\n").replace("\n", "\\n")
            new_lines.append(f'{key}="{value}"')

    env_path.write_text("\n".join(new_lines) + "\n")
    
    # Set restrictive permissions (rw-------)
    try:
        os.chmod(env_path, 0o600)
    except Exception as exc:
        # Best effort for permissions
        logging.debug(f"Unable to enforce 0600 permissions on .env: {exc}")
