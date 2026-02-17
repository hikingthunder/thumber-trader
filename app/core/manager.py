
import asyncio
import logging
from typing import Dict, List, Optional
from decimal import Decimal

from app.config import settings
from app.core.exchange import CoinbaseExchange
from app.core.strategy import GridStrategy
from app.core.state import SharedRiskState

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manages the lifecycle of trading strategies.
    In the V2 architecture, this is initialized by the FastAPI lifespan.
    """
    def __init__(self):
        self.exchange: Optional[CoinbaseExchange] = None
        self.strategies: Dict[str, GridStrategy] = {}
        self.tasks: List[asyncio.Task] = []
        self.shared_risk_state = SharedRiskState()
        self.running = False

    async def initialize(self):
        """Initialize connections and strategies."""
        logger.info("Initializing StrategyManager...")
        
        # Initialize Exchange
        try:
            # Helper to safely extract secret
            def get_secret(v):
                return v.get_secret_value() if hasattr(v, "get_secret_value") else v

            self.exchange = CoinbaseExchange(
                api_key=get_secret(settings.coinbase_api_key),
                api_secret=get_secret(settings.coinbase_api_secret)
            )
            # Basic connectivity check
            accounts = await self.exchange.get_account_balances()
            logger.info(f"Connected to Coinbase. Found {len(accounts)} accounts.")
        except Exception as e:
            logger.error(f"Failed to connect to Exchange: {e}")
            raise

        # Initialize Strategies
        # For now, we only support the single configured product_id
        # In future, iterate over a list of products
        product_id = settings.product_id
        strategy = GridStrategy(
            product_id=product_id,
            exchange=self.exchange,
            shared_risk_state=self.shared_risk_state
        )
        self.strategies[product_id] = strategy
        logger.info(f"Initialized GridStrategy for {product_id}")

    async def start(self):
        """Start all registered strategies."""
        if self.running:
            logger.warning("StrategyManager is already running.")
            return

        self.running = True
        logger.info("Starting strategies...")
        
        for pid, strategy in self.strategies.items():
            task = asyncio.create_task(strategy.run(), name=f"strategy-{pid}")
            self.tasks.append(task)
            logger.info(f"Started task for {pid}")

    async def stop(self):
        """Stop all strategies."""
        if not self.running:
            return

        logger.info("Stopping strategies...")
        self.running = False
        
        # Signal strategies to stop
        for strategy in self.strategies.values():
            strategy.stop()
        
        # Wait for tasks to finish
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks = []
        
        logger.info("All strategies stopped.")

    async def get_global_stats(self) -> Dict:
        """Return a summary of status for API including balances and PnL."""
        
        # 1. Get Exchange Balances
        balances = {}
        if self.exchange:
            try:
                balances = await self.exchange.get_account_balances()
            except Exception as e:
                logger.error(f"Failed to get balances: {e}")
        
        # 2. Get Strategy Stats
        strategy_stats = {}
        total_realized_pnl = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        
        for pid, s in self.strategies.items():
            stats = await s.get_stats()
            strategy_stats[pid] = stats
            total_realized_pnl += stats.get("realized_pnl", Decimal("0"))
            total_unrealized_pnl += stats.get("unrealized_pnl", Decimal("0"))

        return {
            "running": self.running,
            "strategies": strategy_stats,
            "balances": balances,
            "total_realized_pnl": total_realized_pnl,
            "total_unrealized_pnl": total_unrealized_pnl
        }

    def get_status(self) -> Dict:
        """Return a summary of status for API (sync version, deprecated for stats)."""
        return {
            "running": self.running,
            "strategies": {
                pid: {
                    "running": s.running,
                    "orders": len(s.orders),
                    "last_price": s.last_price
                }
                for pid, s in self.strategies.items()
            }
        }

# Global singleton instance to be used by FastAPI
manager = StrategyManager()
