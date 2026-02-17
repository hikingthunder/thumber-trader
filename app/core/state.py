
import threading
from decimal import Decimal
from typing import Dict, Tuple, Optional

class SharedPaperPortfolio:
    def __init__(self, usd: Decimal):
        self.balances: Dict[str, Decimal] = {"USD": usd}
        self.lock = threading.RLock()

    def get_balance(self, currency: str) -> Decimal:
        with self.lock:
            return self.balances.get(currency, Decimal("0"))

    def apply_delta(self, currency: str, delta: Decimal) -> None:
        with self.lock:
            self.balances[currency] = self.balances.get(currency, Decimal("0")) + delta


class SharedRiskState:
    def __init__(self):
        self.lock = threading.RLock()
        self.cross_asset_inventory_caps: Dict[str, Decimal] = {}
        self.layer_inventory_ratios: Dict[Tuple[str, str], Decimal] = {}
        self.pairwise_correlations: Dict[Tuple[str, str], Decimal] = {}
        self.portfolio_beta = Decimal("0")
        self.cointegration_targets: Dict[str, Decimal] = {}
        self.cointegration_signals: Dict[Tuple[str, str], Dict[str, str]] = {}
        self.black_litterman_weights: Dict[str, Decimal] = {}
        self.black_litterman_views: Dict[str, Decimal] = {}

    def set_inventory_cap(self, product_id: str, cap: Decimal) -> None:
        with self.lock:
            self.cross_asset_inventory_caps[product_id] = cap

    def get_inventory_cap(self, product_id: str) -> Optional[Decimal]:
        with self.lock:
            return self.cross_asset_inventory_caps.get(product_id)

    def set_layer_inventory_ratio(self, product_id: str, layer_name: str, ratio: Decimal) -> None:
        key = (product_id.upper(), layer_name.lower())
        with self.lock:
            self.layer_inventory_ratios[key] = ratio

    def get_layer_inventory_ratio(self, product_id: str, layer_name: str) -> Decimal:
        key = (product_id.upper(), layer_name.lower())
        with self.lock:
            return self.layer_inventory_ratios.get(key, Decimal("0"))

    def get_total_inventory_ratio(self, product_id: str) -> Decimal:
        wanted = product_id.upper()
        with self.lock:
            return sum((ratio for (pid, _layer), ratio in self.layer_inventory_ratios.items() if pid == wanted), Decimal("0"))

    def set_correlation(self, left: str, right: str, value: Decimal) -> None:
        key = tuple(sorted((left, right)))
        with self.lock:
            self.pairwise_correlations[key] = value

    def get_correlation(self, left: str, right: str) -> Optional[Decimal]:
        key = tuple(sorted((left, right)))
        with self.lock:
            return self.pairwise_correlations.get(key)

    def set_portfolio_beta(self, beta: Decimal) -> None:
        with self.lock:
            self.portfolio_beta = beta

    def get_portfolio_beta(self) -> Decimal:
        with self.lock:
            return self.portfolio_beta


    def set_cointegration_target(self, product_id: str, target: Decimal) -> None:
        with self.lock:
            self.cointegration_targets[product_id] = target

    def get_cointegration_target(self, product_id: str) -> Optional[Decimal]:
        with self.lock:
            return self.cointegration_targets.get(product_id)

    def set_cointegration_signal(self, left: str, right: str, signal: Dict[str, str]) -> None:
        key = tuple(sorted((left, right)))
        with self.lock:
            self.cointegration_signals[key] = dict(signal)

    def get_cointegration_signals(self) -> Dict[Tuple[str, str], Dict[str, str]]:
        with self.lock:
            return {key: dict(value) for key, value in self.cointegration_signals.items()}

    def set_black_litterman_weight(self, product_id: str, weight: Decimal) -> None:
        with self.lock:
            self.black_litterman_weights[product_id] = weight

    def get_black_litterman_weight(self, product_id: str) -> Optional[Decimal]:
        with self.lock:
            return self.black_litterman_weights.get(product_id)

    def set_black_litterman_view(self, product_id: str, value: Decimal) -> None:
        with self.lock:
            self.black_litterman_views[product_id] = value

    def get_black_litterman_view(self, product_id: str) -> Optional[Decimal]:
        with self.lock:
            return self.black_litterman_views.get(product_id)
