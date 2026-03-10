import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.core.exchange import CoinbaseExchange

def test_balance_parsing():
    asyncio.run(_run_balance_parsing_test())


async def _run_balance_parsing_test():
    # Mock API key/secret
    api_key = "test_key"
    api_secret = "test_secret"
    
    # Mock RESTClient
    with patch("app.core.exchange.RESTClient") as MockClient:
        # 1. Test with list of dicts (Old/Standard API)
        exchange = CoinbaseExchange(api_key, api_secret)
        
        mock_response_dict = {
            "accounts": [
                {
                    "currency": "BTC",
                    "available_balance": {"value": "1.234", "currency": "BTC"}
                },
                {
                    "currency": "USD",
                    "available_balance": {"value": "5000.00", "currency": "USD"}
                }
            ]
        }
        
        exchange.client.get_accounts = MagicMock(return_value=mock_response_dict)
        balances = await exchange.get_account_balances()
        print(f"Test 1 (Dict structure): {balances}")
        assert balances.get("BTC") == Decimal("1.234")
        assert balances.get("USD") == Decimal("5000.00")

        # 2. Test with objects (Some SDK versions)
        class MockAccount:
            def __init__(self, currency, value):
                self.currency = currency
                self.available_balance = {"value": value}

        class MockResponse:
            def __init__(self, accounts):
                self.accounts = accounts

        mock_response_obj = MockResponse([
            MockAccount("ETH", "10.5"),
            MockAccount("USDT", "250.75")
        ])

        exchange.client.get_accounts = MagicMock(return_value=mock_response_obj)
        balances = await exchange.get_account_balances()
        print(f"Test 2 (Object structure): {balances}")
        assert balances.get("ETH") == Decimal("10.5")
        assert balances.get("USDT") == Decimal("250.75")

        # 3. Test with flat string balances (Advanced API variations)
        mock_response_flat = {
            "accounts": [
                {
                    "currency": "SOL",
                    "available_balance": "42.0"
                }
            ]
        }
        exchange.client.get_accounts = MagicMock(return_value=mock_response_flat)
        balances = await exchange.get_account_balances()
        print(f"Test 3 (Flat structure): {balances}")
        assert balances.get("SOL") == Decimal("42.0")

        # 4. Test with list response + nested currency object + only total balance
        class MockCurrency:
            def __init__(self, code):
                self.code = code

        class MockBalanceObj:
            def __init__(self, value):
                self.value = value

        class MockAccountV2:
            def __init__(self, code, value):
                self.currency = MockCurrency(code)
                self.balance = MockBalanceObj(value)
                self.available_balance = None

        exchange.client.get_accounts = MagicMock(return_value=[MockAccountV2("BTC", "0.50")])
        balances = await exchange.get_account_balances()
        print(f"Test 4 (List/object V2 structure): {balances}")
        assert balances.get("BTC") == Decimal("0.50")

    print("\nAll balance parsing tests passed!")

if __name__ == "__main__":
    test_balance_parsing()
