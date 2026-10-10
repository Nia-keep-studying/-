"""Login and customer read permissions using an isolated temporary database."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, select

from backend import sqlmodel_server as server


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
