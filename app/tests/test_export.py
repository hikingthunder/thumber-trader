
import unittest
from app.utils.export import export_data, models_to_dicts, map_to_accounting, get_accounting_headers
from datetime import datetime

class TestExportUtility(unittest.TestCase):
    def test_models_to_dicts_basic(self):
        class MockModel:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
        # Mock inspection
        from unittest.mock import MagicMock
        import app.utils.export
        old_inspect = app.utils.export.inspect
        app.utils.export.inspect = MagicMock()
        
        mock_col1 = MagicMock()
        mock_col1.key = "id"
        mock_col2 = MagicMock()
        mock_col2.key = "ts"
        
        app.utils.export.inspect.return_value.mapper.column_attrs = [mock_col1, mock_col2]
        
        ts_val = 1700000000.0
        mock_model = MockModel(id=1, ts=ts_val)
        
        result = models_to_dicts([mock_model])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], 1)
        # ts should be formatted
        expected_ts = datetime.fromtimestamp(ts_val).strftime('%Y-%m-%d %H:%M:%S')
        self.assertEqual(result[0]["ts"], expected_ts)
        
        app.utils.export.inspect = old_inspect

    def test_map_to_accounting_fills(self):
        data = [{
            "ts": "2024-01-01 12:00:00",
            "product_id": "BTC-USD",
            "side": "BUY",
            "base_size": "0.1",
            "price": "50000",
            "fee_paid": "10"
        }]
        mapped = map_to_accounting(data, "fills")
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0]["Symbol"], "BTC-USD")
        self.assertEqual(mapped[0]["Action"], "BUY")
        self.assertEqual(mapped[0]["Total"], 5000.0)

    def test_export_data_csv(self):
        data = [{"a": 1, "b": 2}]
        headers = ["a", "b"]
        content, filename, mimetype = export_data(data, headers, "test", "csv")
        self.assertEqual(mimetype, "text/csv")
        self.assertTrue(filename.startswith("test_"))
        self.assertTrue(filename.endswith(".csv"))
        self.assertIn(b"a,b", content)
        self.assertIn(b"1,2", content)

if __name__ == '__main__':
    unittest.main()
