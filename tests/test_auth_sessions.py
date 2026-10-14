"""Login and customer read permissions using an isolated temporary database."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

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

    def test_customer_api_has_no_order_writes(self):
        operations = {
            (method.upper(), path)
            for path, item in self.client.get("/openapi.json").json()["paths"].items()
            for method in item if method in {"get", "post", "put", "patch", "delete"}
        }
        self.assertEqual(operations, {
            ("POST", "/login"), ("GET", "/me"), ("POST", "/logout"),
            ("GET", "/orders"), ("GET", "/orders/{order_id}"),
            ("POST", "/human-tickets"),
        })

    def ticket_payload(self, **changes):
        payload = {"summary": "订单售后条件无法确认", "reason": "用户要求人工",
                   "conversation_id": "offline-thread"}
        payload.update(changes)
        return payload

    def test_ticket_without_order_is_saved(self):
        response = self.client.post("/human-tickets", headers=self.headers(self.login()),
                                    json=self.ticket_payload())
        self.assertEqual(response.status_code, 201, response.text)
        with Session(self.engine) as session:
            ticket = session.get(server.HumanTicket, response.json()["ticket_id"])
            self.assertIsNone(ticket.order_id)
            self.assertEqual(ticket.user_id, "U1001")
            self.assertEqual(ticket.status, "待处理")

    def test_ticket_uses_authenticated_owner_and_server_status(self):
        response = self.client.post("/human-tickets", headers=self.headers(self.login()),
            json=self.ticket_payload(order_id="A1001", user_id="U1002", status="已完成",
                                     policy_reference="退款与取消规则 / 发货后退款"))
        self.assertEqual(response.status_code, 201, response.text)
        with Session(self.engine) as session:
            ticket = session.get(server.HumanTicket, response.json()["ticket_id"])
            self.assertEqual((ticket.user_id, ticket.order_id, ticket.status),
                             ("U1001", "A1001", "待处理"))
            self.assertEqual(ticket.conversation_id, "offline-thread")
            self.assertEqual(ticket.summary, "订单售后条件无法确认")
            self.assertGreater(ticket.created_at, 0)
        self.assertIn("尚无真实客服接入", response.json()["message"])

    def test_ticket_rejection_never_inserts_a_record(self):
        headers = self.headers(self.login())
        cases = [({}, self.ticket_payload(), 401),
                 (headers, self.ticket_payload(order_id="A2001"), 404),
                 (headers, self.ticket_payload(order_id="missing"), 404),
                 (headers, self.ticket_payload(summary=""), 422)]
        for auth, body, code in cases:
            with self.subTest(code=code, order=body.get("order_id")):
                self.assertEqual(self.client.post("/human-tickets", headers=auth,
                                                  json=body).status_code, code)
        with Session(self.engine) as session:
            self.assertEqual(session.exec(select(server.HumanTicket)).all(), [])

    def test_expired_session_cannot_create_ticket(self):
        token = self.login()
        with Session(self.engine) as session:
            record = session.get(server.LoginSession, hashlib.sha256(token.encode()).hexdigest())
            record.expires_at = 0
            session.add(record)
            session.commit()
        self.assertEqual(self.client.post("/human-tickets", headers=self.headers(token),
                                         json=self.ticket_payload()).status_code, 401)
        with Session(self.engine) as session:
            self.assertEqual(session.exec(select(server.HumanTicket)).all(), [])

    def test_agent_ticket_tool_saves_current_thread_and_hides_runtime(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from agent.langchain_agent_demo import build_agent
        from tests.test_project_smoke import ToolCallingModel, FakeEmbedding, FakeReranker
        from rag.rag_loader import load_documents, split_documents

        registered = []

        class RecordingModel(ToolCallingModel):
            def bind_tools(self, tools, **kwargs):
                registered.extend(tools)
                return self

        token = self.login()
        inputs = [("first-thread", None), ("new-thread", "A1001"), ("denied-thread", "A2001")]
        responses = []
        for number, (_, order_id) in enumerate(inputs):
            args = {"summary": "需要人工确认售后条件", "reason": "用户主动要求人工"}
            if order_id is not None:
                args["order_id"] = order_id
            responses.extend([
                AIMessage(content="", tool_calls=[{
                    "name": "create_support_ticket", "args": args,
                    "id": f"ticket-call-{number}", "type": "tool_call",
                }]),
                AIMessage(content="模拟最终回答"),
            ])
        chunks = split_documents(load_documents())
        embedding = FakeEmbedding()
        agent = build_agent(RecordingModel(responses=responses), chunks, embedding,
                            embedding.encode(chunks), FakeReranker(), token)

        def local_post(url, **kwargs):
            kwargs.pop("timeout", None)
            return self.client.post(url, **kwargs)

        config = {"configurable": {"thread_id": "first-thread"}}
        saved_ids = []
        with patch.object(order_client.httpx, "post", side_effect=local_post) as post:
            for number, (thread_id, order_id) in enumerate(inputs):
                config["configurable"]["thread_id"] = thread_id
                result = agent.invoke({"messages": [{"role": "user", "content": "请转人工"}]}, config=config)
                messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual(messages[0].tool_call_id, f"ticket-call-{number}")
                self.assertNotIn(token, messages[0].content)
                body = json.loads(messages[0].content)
                if order_id == "A2001":
                    self.assertIn("error", body)
                    self.assertNotIn("ticket_id", body)
                else:
                    self.assertEqual(body["status"], "待处理")
                    saved_ids.append(body["ticket_id"])
                self.assertEqual(post.call_args.kwargs["headers"], self.headers(token))
                self.assertEqual(post.call_args.kwargs["json"]["conversation_id"], thread_id)

        tools = {tool.name: tool for tool in registered}
        schema = tools["create_support_ticket"].tool_call_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"summary", "reason", "order_id", "policy_reference"})
        with Session(self.engine) as session:
            tickets = session.exec(select(server.HumanTicket)).all()
            self.assertEqual(len(tickets), 2)
            self.assertEqual({t.ticket_id for t in tickets}, set(saved_ids))
            self.assertEqual({t.conversation_id for t in tickets}, {"first-thread", "new-thread"})
            self.assertEqual({t.order_id for t in tickets}, {None, "A1001"})
            self.assertTrue(all(t.user_id == "U1001" and t.status == "待处理" for t in tickets))

    def test_return_page_is_public_shell_with_protected_order_data(self):
        response = self.client.get("/demo/return-application?order_id=A1001")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("A1001", response.text)
        self.assertEqual(self.client.get("/orders/A1001").status_code, 401)

    def test_saved_ticket_is_not_repeated_after_model_failure(self):
        from openai import APIConnectionError
        from langchain_core.messages import AIMessage
        from tests.test_project_smoke import ToolCallingModel, FakeEmbedding, FakeReranker
        from agent import langchain_agent_demo as app

        token = self.login()
        model_calls = []

        class FailingAfterToolModel(ToolCallingModel):
            def _generate(self, *args, **kwargs):
                model_calls.append(len(model_calls) + 1)
                if len(model_calls) == 2:
                    raise APIConnectionError(request=httpx.Request("POST", "http://example.test/model"))
                return super()._generate(*args, **kwargs)

        model = FailingAfterToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "create_support_ticket",
                "args": {"summary": "需要人工处理退货", "reason": "用户明确要求人工"},
                "id": "saved-ticket-call", "type": "tool_call",
            }]),
            AIMessage(content="新会话恢复后的回答"),
        ])
        agent = app.build_agent(model, [], FakeEmbedding(), [], FakeReranker(), token)

        def local_post(url, **kwargs):
            kwargs.pop("timeout", None)
            return self.client.post(url, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-test-key"}))
            stack.enter_context(patch("builtins.input", side_effect=["alice", "帮我转人工", "你好", "q"]))
            output = stack.enter_context(patch("builtins.print"))
            stack.enter_context(patch.object(app, "getpass", return_value="test-password"))
            stack.enter_context(patch.object(app, "login", return_value={"access_token": token, "username": "alice"}))
            stack.enter_context(patch.object(app, "is_logged_in", return_value=True))
            stack.enter_context(patch.object(app, "uuid4", side_effect=["ticket-saved-thread", "recovery-thread"]))
            for name in ["load_documents", "split_documents", "load_embedding_model", "build_chunk_vectors", "load_rerank_model", "ChatDeepSeek"]:
                stack.enter_context(patch.object(app, name))
            stack.enter_context(patch.object(app, "build_agent", return_value=agent))
            post = stack.enter_context(patch.object(order_client.httpx, "post", side_effect=local_post))
            app.main()
            self.assertEqual(post.call_count, 1)
            self.assertEqual(len(model_calls), 3)
            output.assert_any_call("本轮请求未能正常完成，异常类型：", "APIConnectionError")
            output.assert_any_call("最终回答：", "新会话恢复后的回答")
            self.assertNotIn(token, str(output.call_args_list))
        with Session(self.engine) as session:
            tickets = session.exec(select(server.HumanTicket)).all()
            self.assertEqual(len(tickets), 1)
            self.assertEqual(tickets[0].conversation_id, "ticket-saved-thread")
        old_messages = agent.get_state({"configurable": {"thread_id": "ticket-saved-thread"}}).values["messages"]
        self.assertEqual([m.type for m in old_messages], ["human", "ai", "tool"])
        new_messages = agent.get_state({"configurable": {"thread_id": "recovery-thread"}}).values["messages"]
        self.assertEqual([m.type for m in new_messages], ["human", "ai"])

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
