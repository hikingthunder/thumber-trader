
import tablib
import csv
import io
from datetime import datetime
from typing import List, Any, Dict
from sqlalchemy.inspection import inspect

def export_data(data, headers_or_format=None, filename_prefix: str = "export", format: str = 'csv'):
    """Export trade/accounting data in legacy tuple mode or workbook mode."""
    # Workbook mode used by /export/generate route.
    if isinstance(data, dict):
        file_format = str(headers_or_format or format or "csv").lower()
        book = tablib.Databook()
        for sheet_name, rows in data.items():
            if isinstance(rows, dict):
                rows = [rows]
            rows = rows or []
            dataset = tablib.Dataset(title=str(sheet_name)[:31])
            headers = list(rows[0].keys()) if rows else []
            if headers:
                dataset.headers = headers
                for row in rows:
                    dataset.append([row.get(h, "") for h in headers])
            book.add_sheet(dataset)

        if file_format == "xlsx":
            return book.export("xlsx")
        if file_format == "ods":
            return book.export("ods")

        # tablib Databook has no direct CSV export; flatten each sheet into one text payload.
        output = io.StringIO()
        for idx, dataset in enumerate(book._datasets):
            if idx > 0:
                output.write("\n")
            output.write(f"[{dataset.title}]\n")
            output.write(dataset.export("csv"))
        return output.getvalue().encode("utf-8")

    # Legacy mode used by tests and helper utilities.
    headers = list(headers_or_format or [])
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
    elif format == 'qfx':
        content = export_qfx(data)
        mimetype = 'application/x-ofx'
        format = 'qfx'
    elif format == 'qbo':
        content = export_qbo(data)
        mimetype = 'text/plain'
        format = 'qbo'
    elif format == 'turbotax':
        content = export_turbotax(data)
        mimetype = 'text/csv'
        format = 'csv'
        filename_prefix = f"turbotax_{filename_prefix}"
    else:
        content = ds.export('csv').encode('utf-8')
        mimetype = 'text/csv'

    filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

    return content, filename, mimetype


def export_qfx(data: List[Dict[str, Any]]) -> str:
    """Generate OFX 2.2 XML (Quicken-compatible QFX) from transaction data."""
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    
    transactions = ""
    for row in data:
        ts = row.get("ts") or row.get("Date") or now
        # Try to parse the timestamp to OFX format
        if isinstance(ts, str) and "-" in ts:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                ofx_date = dt.strftime("%Y%m%d%H%M%S")
            except ValueError:
                ofx_date = now
        else:
            ofx_date = now
        
        side = str(row.get("side", row.get("Action", "BUY"))).upper()
        trntype = "CREDIT" if side == "SELL" else "DEBIT"
        
        price = float(row.get("price", row.get("Price", 0)))
        size = float(row.get("base_size", row.get("Quantity", 0)))
        amount = price * size
        if trntype == "DEBIT":
            amount = -amount
        
        memo = f"{side} {size} @ {price}"
        fitid = str(hash(f"{ofx_date}{side}{amount}"))[-12:]
        
        transactions += f"""<STMTTRN>
<TRNTYPE>{trntype}
<DTPOSTED>{ofx_date}
<TRNAMT>{amount:.2f}
<FITID>{fitid}
<MEMO>{memo}
</STMTTRN>
"""
    
    ofx = f"""OFXHEADER:100
DATA:OFXSGML
VERSION:102
SECURITY:NONE
ENCODING:USASCII
CHARSET:1252
COMPRESSION:NONE
OLDFILEUID:NONE
NEWFILEUID:NONE

<OFX>
<SIGNONMSGSRSV1>
<SONRS>
<STATUS>
<CODE>0
<SEVERITY>INFO
</STATUS>
<DTSERVER>{now}
<LANGUAGE>ENG
</SONRS>
</SIGNONMSGSRSV1>
<BANKMSGSRSV1>
<STMTTRNRS>
<TRNUID>{now}
<STATUS>
<CODE>0
<SEVERITY>INFO
</STATUS>
<STMTRS>
<CURDEF>USD
<BANKACCTFROM>
<BANKID>ThumberTrader
<ACCTID>TRADING
<ACCTTYPE>CHECKING
</BANKACCTFROM>
<BANKTRANLIST>
<DTSTART>{now}
<DTEND>{now}
{transactions}</BANKTRANLIST>
</STMTRS>
</STMTTRNRS>
</BANKMSGSRSV1>
</OFX>"""
    return ofx


def export_qbo(data: List[Dict[str, Any]]) -> str:
    """Generate IIF format (QuickBooks-compatible) from transaction data."""
    lines = []
    # IIF header
    lines.append("!TRNS\tTRNSTYPE\tDATE\tACCNT\tAMOUNT\tMEMO")
    lines.append("!SPL\tTRNSTYPE\tDATE\tACCNT\tAMOUNT\tMEMO")
    lines.append("!ENDTRNS")
    
    for row in data:
        ts = row.get("ts") or row.get("Date") or ""
        if isinstance(ts, str) and "-" in ts:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                date_str = dt.strftime("%m/%d/%Y")
            except ValueError:
                date_str = ts
        else:
            date_str = str(ts)
        
        side = str(row.get("side", row.get("Action", "BUY"))).upper()
        price = float(row.get("price", row.get("Price", 0)))
        size = float(row.get("base_size", row.get("Quantity", 0)))
        amount = price * size
        if side == "BUY":
            amount = -amount
        
        memo = f"{side} {size} @ {price}"
        product = row.get("product_id", row.get("Symbol", "BTC-USD"))
        
        # Transaction line
        lines.append(f"TRNS\tGENERAL JOURNAL\t{date_str}\tTrading Account\t{amount:.2f}\t{memo}")
        # Split line (balancing entry)
        lines.append(f"SPL\tGENERAL JOURNAL\t{date_str}\t{product} Inventory\t{-amount:.2f}\t{memo}")
        lines.append("ENDTRNS")
    
    return "\n".join(lines)


def export_turbotax(data: List[Dict[str, Any]]) -> str:
    """Generate TurboTax-compatible CSV with Form 8949 columns."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Form 8949 headers
    writer.writerow([
        "Description of Property",
        "Date Acquired",
        "Date Sold or Disposed Of",
        "Proceeds (Sales Price)",
        "Cost or Other Basis",
        "Gain or (Loss)",
        "Holding Period (Short/Long)"
    ])
    
    for row in data:
        description = row.get("Description", f"Crypto trade - {row.get('product_id', row.get('Symbol', 'BTC'))}")
        date_acquired = row.get("acquired_ts", row.get("Date Acquired", ""))
        date_sold = row.get("created_ts", row.get("Date Sold", row.get("ts", row.get("Date", ""))))
        proceeds = float(row.get("proceeds_usd", row.get("Proceeds", 0)))
        cost_basis = float(row.get("cost_basis_usd", row.get("Cost Basis", 0)))
        gain_loss = proceeds - cost_basis if proceeds and cost_basis else float(row.get("realized_pnl_usd", row.get("Realized PnL", 0)))
        
        # Determine holding period
        holding = "Short-Term"
        try:
            if date_acquired and date_sold:
                acq = datetime.strptime(str(date_acquired), "%Y-%m-%d %H:%M:%S")
                sold = datetime.strptime(str(date_sold), "%Y-%m-%d %H:%M:%S")
                if (sold - acq).days > 365:
                    holding = "Long-Term"
        except (ValueError, TypeError):
            pass
        
        writer.writerow([
            description,
            date_acquired,
            date_sold,
            f"{proceeds:.2f}",
            f"{cost_basis:.2f}",
            f"{gain_loss:.2f}",
            holding
        ])
    
    return output.getvalue()


def calculate_fee_summary(fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate fee statistics from fill data."""
    total_fees = 0.0
    total_buy_volume = 0.0
    total_sell_volume = 0.0
    trade_count = len(fills)
    
    for fill in fills:
        fee = float(fill.get("fee_paid", 0))
        total_fees += fee
        
        price = float(fill.get("price", 0))
        size = float(fill.get("base_size", 0))
        volume = price * size
        
        side = str(fill.get("side", "")).upper()
        if side == "BUY":
            total_buy_volume += volume
        elif side == "SELL":
            total_sell_volume += volume
    
    total_volume = total_buy_volume + total_sell_volume
    avg_fee_rate = (total_fees / total_volume * 100) if total_volume > 0 else 0
    
    return {
        "total_fees_usd": round(total_fees, 4),
        "total_volume_usd": round(total_volume, 2),
        "buy_volume_usd": round(total_buy_volume, 2),
        "sell_volume_usd": round(total_sell_volume, 2),
        "avg_fee_rate_pct": round(avg_fee_rate, 4),
        "trade_count": trade_count
    }


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
                except (TypeError, ValueError, OSError):
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

