
from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import String, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.database.db import Base

class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    side: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[str] = mapped_column(String, nullable=False) # Stored as string for Decimal precision
    base_size: Mapped[str] = mapped_column(String, nullable=False)
    grid_index: Mapped[int] = mapped_column(Integer, nullable=False)
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    created_ts: Mapped[float] = mapped_column(Float, nullable=False)
    eligible_fill_ts: Mapped[float] = mapped_column(Float, nullable=False)

class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, nullable=False)
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[str] = mapped_column(String, nullable=False)
    base_size: Mapped[str] = mapped_column(String, nullable=False)
    fee_paid: Mapped[str] = mapped_column(String, nullable=False)
    grid_index: Mapped[int] = mapped_column(Integer, nullable=False)
    order_id: Mapped[str] = mapped_column(String, nullable=False)
    tax_lot_method: Mapped[str] = mapped_column(String, default="FIFO")
    realized_pnl_usd: Mapped[str] = mapped_column(String, default="0")

    # Relationships
    tax_lots: Mapped[list["TaxLot"]] = relationship("TaxLot", back_populates="buy_fill", cascade="all, delete-orphan")
    tax_lot_matches: Mapped[list["TaxLotMatch"]] = relationship("TaxLotMatch", back_populates="sell_fill", cascade="all, delete-orphan")

class TaxLot(Base):
    __tablename__ = "tax_lots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    buy_fill_id: Mapped[int] = mapped_column(ForeignKey("fills.id"), nullable=True)
    acquired_ts: Mapped[float] = mapped_column(Float, nullable=False)
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    buy_price: Mapped[str] = mapped_column(String, nullable=False)
    original_base_size: Mapped[str] = mapped_column(String, nullable=False)
    remaining_base_size: Mapped[str] = mapped_column(String, nullable=False)
    fee_paid_usd: Mapped[str] = mapped_column(String, nullable=False)
    closed_ts: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_ts: Mapped[float] = mapped_column(Float, nullable=False)
    updated_ts: Mapped[float] = mapped_column(Float, nullable=False)

    buy_fill: Mapped["Fill"] = relationship("Fill", back_populates="tax_lots")
    matches: Mapped[list["TaxLotMatch"]] = relationship("TaxLotMatch", back_populates="lot")

class TaxLotMatch(Base):
    __tablename__ = "tax_lot_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sell_fill_id: Mapped[int] = mapped_column(ForeignKey("fills.id"), nullable=False)
    lot_id: Mapped[int] = mapped_column(ForeignKey("tax_lots.id"), nullable=False)
    matched_base_size: Mapped[str] = mapped_column(String, nullable=False)
    buy_price: Mapped[str] = mapped_column(String, nullable=False)
    sell_price: Mapped[str] = mapped_column(String, nullable=False)
    proceeds_usd: Mapped[str] = mapped_column(String, nullable=False)
    cost_basis_usd: Mapped[str] = mapped_column(String, nullable=False)
    realized_pnl_usd: Mapped[str] = mapped_column(String, nullable=False)
    acquired_ts: Mapped[float] = mapped_column(Float, nullable=False)
    created_ts: Mapped[float] = mapped_column(Float, nullable=False)

    sell_fill: Mapped["Fill"] = relationship("Fill", back_populates="tax_lot_matches")
    lot: Mapped["TaxLot"] = relationship("TaxLot", back_populates="matches")

class DailyStats(Base):
    __tablename__ = "daily_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, nullable=False)
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    pnl_per_1k: Mapped[str] = mapped_column(String, nullable=False)
    var_95_24h_pct: Mapped[str] = mapped_column(String, nullable=False)
    turnover_ratio: Mapped[str] = mapped_column(String, nullable=False)
    calmar_ratio: Mapped[str] = mapped_column(String, default="0")
    information_ratio: Mapped[str] = mapped_column(String, default="0")
    attribution_rsi: Mapped[str] = mapped_column(String, default="0")
    attribution_macd: Mapped[str] = mapped_column(String, default="0")
    attribution_book_imbalance: Mapped[str] = mapped_column(String, default="0")
    shap_rsi: Mapped[str] = mapped_column(String, default="0")
    shap_macd: Mapped[str] = mapped_column(String, default="0")
    shap_book_imbalance: Mapped[str] = mapped_column(String, default="0")

class StateMeta(Base):
    __tablename__ = "state_meta"
    
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)

class HALock(Base):
    __tablename__ = "ha_lock"

    lock_name: Mapped[str] = mapped_column(String, primary_key=True)
    holder_id: Mapped[str] = mapped_column(String, nullable=False)
    lease_expires_ts: Mapped[float] = mapped_column(Float, nullable=False)
    holder_last_ws_sequence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    holder_ws_sequence_ts: Mapped[float] = mapped_column(Float, nullable=False)
    updated_ts: Mapped[float] = mapped_column(Float, nullable=False)
