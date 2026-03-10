import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.main import app
import app.web.router as web_router
from app.auth.models import AuditLog
from app.database.db import AsyncSessionLocal, init_db
from app.database.models import ConfigVersion


CSRF_TOKEN = "integration-csrf-token"
TEST_ACTOR = "integration-admin"


async def _cleanup_versions_and_audit():
    async with AsyncSessionLocal() as session:
        await session.execute(delete(ConfigVersion).where(ConfigVersion.actor_username == TEST_ACTOR))
        await session.execute(delete(AuditLog).where(AuditLog.action.in_(["config_change", "config_rollback"])))
        await session.commit()


async def _fetch_versions():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ConfigVersion)
            .where(ConfigVersion.actor_username == TEST_ACTOR)
            .order_by(ConfigVersion.id.asc())
        )
        return list(result.scalars().all())


def test_config_save_then_rollback_creates_version_history(monkeypatch):
    asyncio.run(init_db())
    asyncio.run(_cleanup_versions_and_audit())

    fake_env = {"text": "BASE=\"1\"\n"}

    def _fake_update_env_file(updates):
        fake_env["text"] = "\n".join(f'{k}="{v}"' for k, v in sorted(updates.items())) + "\n"
        return True

    def _fake_get_env_text():
        return fake_env["text"]

    def _fake_save_env_text(snapshot):
        fake_env["text"] = snapshot

    async def _fake_log_audit(*_args, **_kwargs):
        return None

    monkeypatch.setattr(web_router, "update_env_file", _fake_update_env_file)
    monkeypatch.setattr(web_router, "_get_env_text", _fake_get_env_text)
    monkeypatch.setattr(web_router, "_save_env_text", _fake_save_env_text)
    monkeypatch.setattr(web_router, "log_audit", _fake_log_audit)

    app.dependency_overrides[web_router.get_current_user] = lambda: SimpleNamespace(
        id=777,
        role="admin",
        username=TEST_ACTOR,
    )

    with TestClient(app) as client:
        save_response = client.post(
            "/config/save",
            data={"paper_trading_mode": "on", "product_ids": "BTC-USD,ETH-USD"},
            headers={"X-CSRF-Token": CSRF_TOKEN},
            cookies={"thumber_csrf_token": CSRF_TOKEN},
        )
        assert save_response.status_code == 200
        assert "versioned" in save_response.text

        versions = asyncio.run(_fetch_versions())
        assert len(versions) == 1

        rollback_response = client.post(
            "/config/rollback",
            data={"version_id": str(versions[0].id)},
            headers={"X-CSRF-Token": CSRF_TOKEN},
            cookies={"thumber_csrf_token": CSRF_TOKEN},
        )
        assert rollback_response.status_code == 200
        assert "rollback applied" in rollback_response.text.lower()

    versions = asyncio.run(_fetch_versions())
    assert len(versions) == 2
    assert versions[-1].rollback_from_id == versions[0].id

    app.dependency_overrides.clear()
    asyncio.run(_cleanup_versions_and_audit())
