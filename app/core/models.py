
from dataclasses import dataclass
from decimal import Decimal
from typing import List

@dataclass
class HMMModelSnapshot:
    init_prob: List[float]
    trans: List[List[float]]
    means: List[float]
    variances: List[float]
    training_returns: List[float]

@dataclass
class TaxLotMatch:
    lot_id: int
    matched_base_size: Decimal
    buy_price: Decimal
    sell_price: Decimal
    proceeds_usd: Decimal
    cost_basis_usd: Decimal
    realized_pnl_usd: Decimal
    acquired_ts: float

@dataclass
class TaxLotSellResult:
    matched_base_size: Decimal
    unmatched_base_size: Decimal
    total_proceeds_usd: Decimal
    total_cost_basis_usd: Decimal
    total_realized_pnl_usd: Decimal
    matches: List[TaxLotMatch]
