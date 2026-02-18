
import tablib
from datetime import datetime
from typing import List, Any, Dict
from sqlalchemy.inspection import inspect

def export_data(data: List[Dict[str, Any]], headers: List[str], filename_prefix: str, format: str = 'csv'):
    """
    Export a list of dictionaries to the specified format.
    """
    ds = tablib.Dataset()
    ds.headers = headers
    
    for row in data:
        ds.append([row.get(h, "") for h in headers])
    
    if format == 'xlsx':
        content = ds.export('xlsx')
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif format == 'ods':
        content = ds.export('ods')
        mimetype = 'application/vnd.oasis.opendocument.spreadsheet'
    else:
        content = ds.export('csv')
        mimetype = 'text/csv'
    
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
    
    return content, filename, mimetype

def models_to_dicts(models: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert a list of SQLAlchemy models to a list of dictionaries.
    """
    result = []
    for model in models:
        d = {}
        for column in inspect(model).mapper.column_attrs:
            val = getattr(model, column.key)
            # Format certain types for better export readability
            if column.key.endswith('_ts') or column.key == 'ts':
                try:
                    val = datetime.fromtimestamp(float(val)).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
            d[column.key] = val
        result.append(d)
    return result

def get_accounting_headers(type: str) -> List[str]:
    """
    Returns headers suitable for Quickbooks/Quicken imports.
    """
    if type == "fills":
        return ["Date", "Description", "Action", "Symbol", "Quantity", "Price", "Fees", "Total"]
    elif type == "tax_matches":
        return ["Date Acquired", "Date Sold", "Description", "Quantity", "Cost Basis", "Proceeds", "Realized PnL"]
    return []

def map_to_accounting(data: List[Dict[str, Any]], type: str) -> List[Dict[str, Any]]:
    """
    Map internal model data to accounting-friendly naming.
    """
    mapped = []
    if type == "fills":
        for row in data:
            mapped.append({
                "Date": row.get("ts"),
                "Description": f"Trade fill for {row.get('product_id')}",
                "Action": row.get("side"),
                "Symbol": row.get("product_id"),
                "Quantity": row.get("base_size"),
                "Price": row.get("price"),
                "Fees": row.get("fee_paid"),
                "Total": float(row.get("price", 0)) * float(row.get("base_size", 0))
            })
    elif type == "tax_matches":
        for row in data:
            mapped.append({
                "Date Acquired": row.get("acquired_ts"),
                "Date Sold": row.get("created_ts"), # created_ts is sell time in TaxLotMatch
                "Description": f"Realized PnL match for lot {row.get('lot_id')}",
                "Quantity": row.get("matched_base_size"),
                "Cost Basis": row.get("cost_basis_usd"),
                "Proceeds": row.get("proceeds_usd"),
                "Realized PnL": row.get("realized_pnl_usd")
            })
    return mapped
