from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.web.router import _require_admin


def test_require_admin_allows_admin_role():
    admin_user = SimpleNamespace(role="admin")
    _require_admin(admin_user)


def test_require_admin_rejects_viewer_role():
    viewer_user = SimpleNamespace(role="viewer")
    with pytest.raises(HTTPException) as exc:
        _require_admin(viewer_user)
    assert exc.value.status_code == 403
    assert exc.value.detail == "Admin access required"
