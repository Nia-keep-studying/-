"""Login and customer read permissions using an isolated temporary database."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, select

from backend import sqlmodel_server as server
from agent import order_client
from agent.business_entry import get_return_entry
from urllib.parse import urlsplit, parse_qs
from settings import RETURN_APPLICATION_URL
import httpx


class AuthSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        database = Path(self.directory.name) / "auth.db"
        self.engine = create_engine(
            f"sqlite:///{database.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        self.enterContext(patch.object(server, "database_file", database))
        self.enterContext(patch.object(server, "engine", self.engine))
        self.client = self.enterContext(TestClient(server.app))
        with Session(self.engine) as session:
            session.add_all([
                server.User(user_id="U1001", username="alice", password_hash=server.password_hasher.hash("auth-test-password")),
                server.User(user_id="U1002", username="bob", password_hash=server.password_hasher.hash("auth-test-password")),
            ])
            session.commit()
            session.add_all([
                server.Order(order_id="A1001", user_id="U1001", product="电脑", status="已发货"),
                server.Order(order_id="A1002", user_id="U1001", product="手机", status="待发货"),
                server.Order(order_id="A2001", user_id="U1002", product="显示器", status="已发货"),
            ])
            session.commit()

    def login(self, username="alice"):
        response = self.client.post("/login", json={
            "username": username, "password": "auth-test-password",
        })
        self.assertEqual(response.status_code, 200)
        return response.json()["access_token"]

    @staticmethod
    def headers(token):
        return {"Authorization": f"Bearer {token}"}

    def test_login_checks_password_and_stores_only_token_hash(self):
        for username, password in [("alice", "incorrect"), ("missing", "auth-test-password")]:
            response = self.client.post("/login", json={"username": username, "password": password})
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["detail"], "用户名或密码错误")
        with Session(self.engine) as session:
            self.assertEqual(session.exec(select(server.LoginSession)).all(), [])

        token = self.login()
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with Session(self.engine) as session:
            record = session.get(server.LoginSession, token_hash)
            self.assertIsNotNone(record)
            self.assertEqual(record.user_id, "U1001")
            self.assertNotEqual(record.token_hash, token)
            self.assertIsNone(session.get(server.LoginSession, token))
        response = self.client.get("/me", headers=self.headers(token))
        self.assertEqual(response.json(), {"user_id": "U1001", "username": "alice"})

    def test_reads_require_login_and_filter_by_owner(self):
        for path in ["/me", "/orders", "/orders/A1001"]:
            for headers in [{}, self.headers("invalid-token"), {"Authorization": "Basic invalid"}]:
                self.assertEqual(self.client.get(path, headers=headers).status_code, 401)
        alice = self.headers(self.login("alice"))
        bob = self.headers(self.login("bob"))
        self.assertEqual({o["order_id"] for o in self.client.get("/orders", headers=alice).json()}, {"A1001", "A1002"})
        self.assertEqual([o["order_id"] for o in self.client.get("/orders", headers=bob).json()], ["A2001"])
        filtered = self.client.get("/orders", headers=alice, params={"status": "已发货", "product_keyword": "电脑"})
        self.assertEqual([o["order_id"] for o in filtered.json()], ["A1001"])
        for path, headers in [("/orders/A1001", alice), ("/orders/A2001", bob)]:
            self.assertEqual(self.client.get(path, headers=headers).status_code, 200)
        for path, headers in [("/orders/A2001", alice), ("/orders/A1001", bob), ("/orders/missing", alice)]:
            self.assertEqual(self.client.get(path, headers=headers).status_code, 404)

    def test_logout_revokes_current_token_and_keeps_other_sessions(self):
        old = self.login()
        other_device = self.login()
        bob = self.login("bob")
        self.assertEqual(self.client.post("/logout").status_code, 401)
        response = self.client.post("/logout", headers=self.headers(old))
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        self.assertEqual(self.client.get("/me", headers=self.headers(old)).status_code, 401)
        with Session(self.engine) as session:
            token_hash = hashlib.sha256(old.encode("utf-8")).hexdigest()
            self.assertIsNone(session.get(server.LoginSession, token_hash))
        for token in [other_device, bob, self.login()]:
            self.assertEqual(self.client.get("/me", headers=self.headers(token)).status_code, 200)

    def test_expiration_is_enforced_even_if_record_remains(self):
        issued_at = 2_000_000_000
        with patch.object(server, "LOGIN_SESSION_TTL_SECONDS", 60), patch.object(server.time, "time", return_value=issued_at):
            token = self.login()
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with Session(self.engine) as session:
            self.assertEqual(session.get(server.LoginSession, token_hash).expires_at, issued_at + 60)
        with patch.object(server.time, "time", return_value=issued_at + 59):
            self.assertEqual(self.client.get("/orders", headers=self.headers(token)).status_code, 200)
        with patch.object(server.time, "time", return_value=issued_at + 60):
            self.assertEqual(self.client.get("/orders", headers=self.headers(token)).status_code, 401)
        with Session(self.engine) as session:
            self.assertIsNotNone(session.get(server.LoginSession, token_hash))

    def test_customer_api_exposes_only_auth_and_order_reads(self):
        operations = {
            (method.upper(), path)
            for path, item in self.client.get("/openapi.json").json()["paths"].items()
            for method in item if method in {"get", "post", "put", "patch", "delete"}
        }
        self.assertEqual(operations, {
            ("POST", "/login"), ("GET", "/me"), ("POST", "/logout"),
            ("GET", "/orders"), ("GET", "/orders/{order_id}"),
        })

    def test_http_client_credentials_filters_and_empty_results(self):
        def local_request(method):
            def send(url, **kwargs):
                kwargs.pop("timeout", None)
                return getattr(self.client, method)(url, **kwargs)
            return send

        with patch.object(order_client.httpx, "post", side_effect=local_request("post")), patch.object(order_client.httpx, "get", side_effect=local_request("get")):
            token = order_client.login("alice", "auth-test-password")["access_token"]
            self.assertTrue(order_client.is_logged_in(token))
            self.assertFalse(order_client.is_logged_in("invalid-test-token"))
            self.assertEqual(order_client.get_order("A1001", token)["user_id"], "U1001")
            self.assertIn("error", order_client.get_order("A2001", token))
            self.assertEqual([o["order_id"] for o in order_client.get_order_list(token, product_keyword="手机")], ["A1002"])
            self.assertEqual([o["order_id"] for o in order_client.get_order_list(token, status="已发货")], ["A1001"])
            self.assertEqual(len(order_client.get_order_list(token, offset=1, limit=1)), 1)
            self.assertEqual(order_client.get_order_list(token, product_keyword="不存在的商品"), [])
            self.assertEqual(order_client.get_order_list("invalid-test-token")["code"], "unauthenticated")
            self.client.post("/logout", headers=self.headers(token))
            self.assertFalse(order_client.is_logged_in(token))

    def test_login_check_keeps_service_failure_distinct_from_expiration(self):
        request = httpx.Request("GET", "http://example.test/me")
        with patch.object(order_client.httpx, "get", side_effect=httpx.ConnectError("simulated")):
            with self.assertRaises(httpx.HTTPError):
                order_client.is_logged_in("test-token")
        with patch.object(order_client.httpx, "get", return_value=httpx.Response(503, request=request)):
            with self.assertRaises(httpx.HTTPStatusError):
                order_client.is_logged_in("test-token")

    def test_return_entry_preparation_checks_owner_status_and_preserves_auth_error(self):
        def local_get(url, **kwargs):
            kwargs.pop("timeout", None)
            return self.client.get(url, **kwargs)

        token = self.login()
        with patch.object(order_client.httpx, "get", side_effect=local_get):
            entry = get_return_entry("A1001", token)
            self.assertEqual(entry["order_id"], "A1001")
            self.assertTrue(entry["is_demo"])
            parsed = urlsplit(entry["entry_url"])
            base = urlsplit(RETURN_APPLICATION_URL)
            self.assertEqual((parsed.scheme, parsed.netloc, parsed.path), (base.scheme, base.netloc, base.path))
            self.assertEqual(parse_qs(parsed.query), {"order_id": ["A1001"]})
            self.assertNotIn(token, entry["entry_url"])
            self.assertIn("不表示申请已提交或退款已批准", entry["message"])
            denied = get_return_entry("A2001", token)
            self.assertIn("error", denied)
            self.assertNotIn("entry_url", denied)
            pending = get_return_entry("A1002", token)
            self.assertEqual(pending["code"], "unsupported_status")
            self.assertNotIn("entry_url", pending)
            invalid = get_return_entry("A1001", "invalid-test-token")
            self.assertEqual(invalid["code"], "unauthenticated")
            self.assertNotIn("entry_url", invalid)
