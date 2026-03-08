import os
import tempfile
import pytest
from pathlib import Path
from app.utils.helpers import _get_key_from_env, encrypt_value, decrypt_value, update_env_file

def test_encryption_lifecycle():
    # Ensure no ENV var
    if "APP_ENV_ENC_KEY" in os.environ:
        del os.environ["APP_ENV_ENC_KEY"]
        
    # Test auto-generation of .enc_key
    if Path(".enc_key").exists():
        Path(".enc_key").unlink()
    
    key1 = _get_key_from_env()
    assert key1 is not None
    assert Path(".enc_key").exists()
    
    key2 = _get_key_from_env()
    assert key1 == key2 # Should be deterministic from file
    
    # Test encryption
    plaintext = "super_secret_webhook_url"
    ciphertext = encrypt_value(plaintext)
    assert ciphertext != plaintext
    assert ciphertext.startswith("AES:")
    
    # Test decryption
    decrypted = decrypt_value(ciphertext)
    assert decrypted == plaintext
    
def test_update_env_file_encryption():
    if "APP_ENV_ENC_KEY" in os.environ:
        del os.environ["APP_ENV_ENC_KEY"]
        
    test_env = Path(".env")
    if test_env.exists():
        test_env_bak = test_env.read_text()
    else:
        test_env_bak = ""
        
    try:
        # Initial write
        update_env_file({"COINBASE_API_KEY": "my_api_key", "NORMAL_VAR": "plain"})
        
        content = test_env.read_text()
        assert "NORMAL_VAR=\"plain\"" in content
        assert "COINBASE_API_KEY=\"AES:" in content
        assert "my_api_key" not in content
        
    finally:
        test_env.write_text(test_env_bak)
