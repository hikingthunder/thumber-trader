from app.web.router import _build_change_summary
from app.utils.helpers import encrypt_value, decrypt_value


def test_build_change_summary_orders_and_counts_keys():
    summary = _build_change_summary(["Z_KEY", "A_KEY", "M_KEY"])
    assert summary.startswith("Updated 3 keys:")
    assert "A_KEY, M_KEY, Z_KEY" in summary


def test_build_change_summary_mentions_rollback_source():
    summary = _build_change_summary(["PAPER_TRADING_MODE"], rollback_from_id=7)
    assert summary == "Rollback to config version 7: PAPER_TRADING_MODE"


def test_env_snapshot_cipher_roundtrip_hides_plaintext():
    snapshot = "COINBASE_API_KEY=\"super-secret\"\nPAPER_TRADING_MODE=\"true\"\n"
    encrypted = encrypt_value(snapshot)
    assert encrypted != snapshot
    assert encrypted.startswith("AES:")
    restored = decrypt_value(encrypted)
    assert restored == snapshot
