
import os
import base64
import logging
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def _get_fernet():
    """
    Get a Fernet instance for encryption/decryption.
    Uses APP_ENV_ENC_KEY to derive a key.
    If not set, returns None.
    """
    key_str = os.getenv("APP_ENV_ENC_KEY")
    if not key_str:
        return None
    
    # Derive a 32-byte key from the provided string
    salt = b"thumber-trader-salt" # Static salt for simplicity in this context
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(key_str.encode()))
    return Fernet(key)

def _should_encrypt_key(key: str) -> bool:
    """Check if a key is sensitive and should be encrypted."""
    sensitive_keys = {
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "DISCORD_WEBHOOK_URL"
    }
    return key.upper() in sensitive_keys

def encrypt_value(value: str) -> str:
    """Encrypt a value and prefix it with ENC: if a key is available."""
    fernet = _get_fernet()
    if not fernet or not value:
        return value
    
    if value.startswith("ENC:"):
        return value
        
    encrypted = fernet.encrypt(value.encode()).decode()
    return f"ENC:{encrypted}"

def decrypt_value(value: str) -> str:
    """Decrypt a value if it starts with ENC: and a key is available."""
    if not isinstance(value, str) or not value.startswith("ENC:"):
        return value
        
    fernet = _get_fernet()
    if not fernet:
        logging.warning("Encryption key not found, cannot decrypt sensitive value.")
        return value
        
    try:
        decrypted = fernet.decrypt(value[4:].encode()).decode()
        return decrypted
    except Exception as e:
        logging.error(f"Failed to decrypt value: {e}")
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
    except Exception:
        # Best effort for permissions
        pass
