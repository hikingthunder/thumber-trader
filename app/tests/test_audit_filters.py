from sqlalchemy import select
from sqlalchemy.dialects import sqlite

from app.auth.auth_router import _apply_audit_filter
from app.auth.models import AuditLog


def _sql(stmt):
    return str(stmt.compile(dialect=sqlite.dialect(), compile_kwargs={"literal_binds": True}))


def test_apply_audit_filter_config_uses_prefix_match():
    stmt = _apply_audit_filter(select(AuditLog), "config")
    sql = _sql(stmt)
    assert "audit_logs.action LIKE" in sql
    assert "config%" in sql


def test_apply_audit_filter_rollback_targets_exact_action():
    stmt = _apply_audit_filter(select(AuditLog), "rollback")
    sql = _sql(stmt)
    assert "audit_logs.action =" in sql
    assert "config_rollback" in sql


def test_apply_audit_filter_unknown_keeps_query_unchanged():
    base = select(AuditLog)
    filtered = _apply_audit_filter(base, "all")
    assert _sql(base) == _sql(filtered)
