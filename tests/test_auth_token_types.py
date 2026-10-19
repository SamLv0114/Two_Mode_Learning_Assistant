"""Regression tests for the access/refresh token boundary."""

import unittest
from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.deps import get_current_user, get_db_session
from src.api.routers.auth import REFRESH_JTI_PREFIX, get_redis, router as auth_router
from src.api.security import create_access_token, create_refresh_token, decode_token


class _UserQuery:
    def __init__(self, user):
        self.user = user

    def filter(self, *_args):
        return self

    def first(self):
        return self.user


class _DB:
    def __init__(self):
        self.user = SimpleNamespace(id=7, email="test@example.com", is_active=True)

    def query(self, *_args):
        return _UserQuery(self.user)


class _Redis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def delete(self, key):
        self.values.pop(key, None)


class TokenTypeTests(unittest.TestCase):
    def test_refresh_token_never_authenticates_as_bearer(self):
        app = FastAPI()
        app.dependency_overrides[get_db_session] = _DB
        redis = _Redis()
        app.dependency_overrides[get_redis] = lambda: redis
        app.include_router(auth_router)

        @app.get("/required")
        async def required(user=Depends(get_current_user)):
            return {"user_id": user.id}

        access = create_access_token(7, "test@example.com")
        refresh, jti = create_refresh_token(7)
        redis.setex(f"{REFRESH_JTI_PREFIX}{jti}", 3600, "7")
        self.assertEqual(decode_token(refresh)["type"], "refresh")

        with TestClient(app) as client:
            self.assertEqual(client.get("/required", headers={"Authorization": f"Bearer {access}"}).status_code, 200)
            self.assertEqual(client.get("/required", headers={"Authorization": f"Bearer {refresh}"}).status_code, 401)

            rotated = client.post("/auth/refresh", json={"refresh_token": refresh})
            self.assertEqual(rotated.status_code, 200)
            self.assertEqual(client.post("/auth/refresh", json={"refresh_token": refresh}).status_code, 401)
            self.assertEqual(client.get("/required", headers={"Authorization": f"Bearer {refresh}"}).status_code, 401)
            new_access = rotated.json()["access_token"]
            self.assertEqual(client.get("/required", headers={"Authorization": f"Bearer {new_access}"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
