"""Login and customer read permissions using an isolated temporary database."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from contextlib import ExitStack
from types import SimpleNamespace

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
        self.enterContext(patch.object(server, "engine", self.engine))
        # Offline checks must not open the real PostgreSQL checkpoint connection.
        checkpoint_factory = self.enterContext(patch.object(server.PostgresSaver, "from_conn_string"))
        self.checkpointer = checkpoint_factory.return_value.__enter__.return_value
        # Lifespan now prepares the Agent; offline checks must never load real models.
        with patch("agent.langchain_agent_demo.initialize_agent", return_value=object()) as initialize:
            self.client = self.enterContext(TestClient(server.app))
        self.initialize_agent = initialize
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

    def test_homepage_serves_public_shell_without_running_agent(self):
        with patch("agent.langchain_agent_demo.run_chat_turn") as run:
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIn('id="login-form"', response.text)
        self.assertIn('id="chat-form"', response.text)
        run.assert_not_called()
        with Session(self.engine) as session:
            self.assertEqual(session.exec(select(server.ChatConversation)).all(), [])
            self.assertEqual(session.exec(select(server.LoginSession)).all(), [])

    def test_chat_runs_tools_and_keeps_history_only_in_the_selected_thread(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from agent.langchain_agent_demo import build_agent
        from tests.test_project_smoke import ToolCallingModel, FakeEmbedding, FakeReranker

        model_inputs = []

        class RecordingModel(ToolCallingModel):
            def _generate(self, messages, *args, **kwargs):
                model_inputs.append(list(messages))
                return super()._generate(messages, *args, **kwargs)

        agent = build_agent(RecordingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "query_order", "args": {"order_id": "A1001"},
                "id": "chat-order-lookup", "type": "tool_call",
            }]),
            AIMessage(content="A1001 的电脑已发货。"),
            AIMessage(content="刚才查询的是 A1001。"),
            AIMessage(content="这是新会话。"),
        ]), [], FakeEmbedding(), [], FakeReranker())
        token = self.login()
        question = "查询 A1001 的订单"

        def local_get(url, **kwargs):
            kwargs.pop("timeout", None)
            return self.client.get(url, **kwargs)

        with patch.object(server.app.state, "agent", agent), \
             patch.object(order_client.httpx, "get", side_effect=local_get) as lookup, \
             patch("builtins.print"):
            first = self.client.post("/chat", headers=self.headers(token), json={"question": question})
            self.assertEqual(first.status_code, 200)
            conversation_id = first.json()["conversation_id"]
            self.assertEqual(first.json()["answer"], "A1001 的电脑已发货。")
            self.assertEqual(lookup.call_args.kwargs["headers"], self.headers(token))

            second = self.client.post("/chat", headers=self.headers(token), json={
                "question": "刚才查的是哪一单？", "conversation_id": conversation_id,
            })
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["conversation_id"], conversation_id)
            self.assertEqual(second.json()["answer"], "刚才查询的是 A1001。")
            self.assertTrue(any(isinstance(m, HumanMessage) and m.content == question for m in model_inputs[2]))
            self.assertTrue(any(isinstance(m, ToolMessage) and "A1001" in m.content for m in model_inputs[2]))

            fresh = self.client.post("/chat", headers=self.headers(token), json={"question": "你好"})
            self.assertEqual(fresh.status_code, 200)
            self.assertNotEqual(fresh.json()["conversation_id"], conversation_id)
            self.assertFalse(any(isinstance(m, HumanMessage) and m.content == question for m in model_inputs[3]))

        self.initialize_agent.assert_called_once_with(checkpointer=self.checkpointer)
        with Session(self.engine) as session:
            conversations = session.exec(select(server.ChatConversation)).all()
            self.assertEqual(len(conversations), 2)
            self.assertTrue(all(c.user_id == "U1001" and c.created_at > 0 for c in conversations))
        self.assertNotIn(token, repr(model_inputs))

    def test_chat_rejects_other_owners_invalid_sessions_and_blank_questions_before_agent(self):
        alice, bob = self.login(), self.login("bob")
        with patch("agent.langchain_agent_demo.run_chat_turn", return_value="模拟回答") as run:
            created = self.client.post("/chat", headers=self.headers(alice), json={"question": "你好"})
            self.assertEqual(created.status_code, 200)
            conversation_id = created.json()["conversation_id"]
            run.reset_mock()

            cases = [
                (self.headers(bob), {"question": "继续", "conversation_id": conversation_id}, 404),
                (self.headers(alice), {"question": "继续", "conversation_id": "missing-thread"}, 404),
                ({}, {"question": "你好"}, 401),
                (self.headers("invalid-token"), {"question": "你好"}, 401),
                (self.headers(alice), {"question": "   "}, 422),
                (self.headers(alice), {"question": ""}, 422),
            ]
            for headers, body, expected in cases:
                with self.subTest(status=expected, body=body):
                    response = self.client.post("/chat", headers=headers, json=body)
                    self.assertEqual(response.status_code, expected)
            self.assertEqual(self.client.post("/logout", headers=self.headers(alice)).status_code, 204)
            expired = self.client.post("/chat", headers=self.headers(alice), json={
                "question": "继续", "conversation_id": conversation_id,
            })
            self.assertEqual(expired.status_code, 401)
            run.assert_not_called()

        with Session(self.engine) as session:
            conversations = session.exec(select(server.ChatConversation)).all()
            self.assertEqual([(c.conversation_id, c.user_id) for c in conversations], [(conversation_id, "U1001")])

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
            ("POST", "/human-tickets"), ("POST", "/chat"),
            ("GET", "/conversations"), ("POST", "/conversations"),
            ("GET", "/conversations/{conversation_id}/messages"),
        })

    def test_conversation_list_is_owned_ordered_and_paginated(self):
        with Session(self.engine) as session:
            session.add_all([
                server.ChatConversation(conversation_id="alice-old", user_id="U1001", created_at=10),
                server.ChatConversation(conversation_id="alice-a", user_id="U1001", created_at=20),
                server.ChatConversation(conversation_id="alice-z", user_id="U1001", created_at=20),
                server.ChatConversation(conversation_id="bob-new", user_id="U1002", created_at=30),
            ])
            session.commit()
        alice = self.headers(self.login())
        with patch.object(server.app.state, "agent", Mock()) as agent:
            response = self.client.get("/conversations", headers=alice)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {
                "conversations": [
                    {"conversation_id": "alice-z", "created_at": 20},
                    {"conversation_id": "alice-a", "created_at": 20},
                    {"conversation_id": "alice-old", "created_at": 10},
                ], "offset": 0, "limit": 20,
            })
            page = self.client.get("/conversations?offset=1&limit=1", headers=alice)
            self.assertEqual(page.json(), {
                "conversations": [{"conversation_id": "alice-a", "created_at": 20}],
                "offset": 1, "limit": 1,
            })
            self.assertEqual(self.client.get("/conversations?offset=3", headers=alice).json()["conversations"], [])
            bob = self.client.get("/conversations", headers=self.headers(self.login("bob")))
            self.assertEqual(bob.json()["conversations"], [{"conversation_id": "bob-new", "created_at": 30}])
            for query in ["offset=-1", "limit=0", "limit=101"]:
                self.assertEqual(self.client.get("/conversations?" + query, headers=alice).status_code, 422)
            for headers in [{}, self.headers("invalid-token")]:
                self.assertEqual(self.client.get("/conversations", headers=headers).status_code, 401)
            agent.get_state.assert_not_called()
            agent.invoke.assert_not_called()

    def test_conversation_history_returns_visible_text_without_running_agent(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        with Session(self.engine) as session:
            session.add(server.ChatConversation(conversation_id="alice-history", user_id="U1001", created_at=10))
            session.commit()
        token = self.login()
        agent = Mock()
        agent.get_state.return_value = SimpleNamespace(values={"messages": [
            SystemMessage(content="private system instructions"),
            HumanMessage(content="查询订单 A1001"),
            AIMessage(content="internal tool preface", tool_calls=[{
                "name": "query_order", "args": {"order_id": "A1001"},
                "id": "history-call", "type": "tool_call",
            }]),
            ToolMessage(content="private tool payload", tool_call_id="history-call"),
            AIMessage(content=[
                {"type": "reasoning", "reasoning": "private reasoning"},
                {"type": "text", "text": "订单 A1001 已发货。"},
            ]),
            AIMessage(content=""),
        ]})
        with patch.object(server.app.state, "agent", agent):
            response = self.client.get("/conversations/alice-history/messages", headers=self.headers(token))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {
                "conversation_id": "alice-history", "messages": [
                    {"role": "user", "content": "查询订单 A1001"},
                    {"role": "assistant", "content": "订单 A1001 已发货。"},
                ],
            })
            agent.get_state.assert_called_once_with({"configurable": {"thread_id": "alice-history"}})
            agent.get_state.return_value = SimpleNamespace(values={})
            empty = self.client.get("/conversations/alice-history/messages", headers=self.headers(token))
            self.assertEqual(empty.json(), {"conversation_id": "alice-history", "messages": []})
            agent.invoke.assert_not_called()

    def test_conversation_history_rejects_access_before_reading_checkpoints(self):
        with Session(self.engine) as session:
            session.add(server.ChatConversation(conversation_id="alice-private", user_id="U1001", created_at=10))
            session.commit()
        alice, bob = self.login(), self.login("bob")
        with patch.object(server.app.state, "agent", Mock()) as agent:
            for headers, conversation_id, status in [
                (self.headers(bob), "alice-private", 404),
                (self.headers(alice), "missing-conversation", 404),
                ({}, "alice-private", 401),
                (self.headers("invalid-token"), "alice-private", 401),
            ]:
                with self.subTest(status=status, conversation_id=conversation_id):
                    response = self.client.get(f"/conversations/{conversation_id}/messages", headers=headers)
                    self.assertEqual(response.status_code, status)
            self.client.post("/logout", headers=self.headers(alice))
            self.assertEqual(self.client.get("/conversations/alice-private/messages", headers=self.headers(alice)).status_code, 401)
            agent.get_state.assert_not_called()
            agent.invoke.assert_not_called()

    def ticket_payload(self, **changes):
        payload = {"summary": "订单售后条件无法确认", "reason": "用户要求人工",
                   "conversation_id": "offline-thread",
                   "idempotency_key": "offline-ticket-operation"}
        payload.update(changes)
        return payload

    def register_test_conversation(self, conversation_id, user_id="U1001"):
        with Session(self.engine) as session:
            session.add(server.ChatConversation(
                conversation_id=conversation_id, user_id=user_id, created_at=10,
            ))
            session.commit()

    def test_create_conversation_registers_authenticated_owner(self):
        token = self.login()
        self.assertEqual(self.client.post("/conversations").status_code, 401)
        self.assertEqual(self.client.post("/conversations", headers=self.headers("invalid")).status_code, 401)
        with patch.object(server.app.state, "agent", Mock()) as agent:
            response = self.client.post("/conversations", headers=self.headers(token))
            self.assertEqual(response.status_code, 200, response.text)
            conversation_id = response.json()["conversation_id"]
            agent.invoke.assert_not_called()
        with Session(self.engine) as session:
            conversations = session.exec(select(server.ChatConversation)).all()
            self.assertEqual(len(conversations), 1)
            self.assertEqual(conversations[0].conversation_id, conversation_id)
            self.assertEqual(conversations[0].user_id, "U1001")

    def test_ticket_rejects_missing_and_other_users_conversations(self):
        self.register_test_conversation("bob-private", "U1002")
        headers = self.headers(self.login())
        for conversation_id in ("missing", "bob-private"):
            with self.subTest(conversation_id=conversation_id):
                response = self.client.post("/human-tickets", headers=headers,
                    json=self.ticket_payload(conversation_id=conversation_id))
                self.assertEqual(response.status_code, 404)
        with Session(self.engine) as session:
            self.assertEqual(session.exec(select(server.HumanTicket)).all(), [])

    def test_http_conversation_client_parses_success_and_raises_on_error(self):
        token = self.login()

        def local_post(url, **kwargs):
            kwargs.pop("timeout", None)
            result = self.client.post(url, **kwargs)
            # TestClient may use a separate HTTPX distribution; the adapter uses httpx.
            return httpx.Response(result.status_code, content=result.content,
                headers=result.headers, request=httpx.Request("POST", str(result.request.url)))

        with patch.object(order_client.httpx, "post", side_effect=local_post) as post:
            conversation_id = order_client.create_conversation(token)
            self.assertTrue(post.call_args.args[0].endswith("/conversations"))
            self.assertEqual(post.call_args.kwargs["headers"], self.headers(token))
            with self.assertRaises(httpx.HTTPStatusError):
                order_client.create_conversation("invalid")
        with Session(self.engine) as session:
            self.assertEqual(session.get(server.ChatConversation, conversation_id).user_id, "U1001")

    def test_ticket_client_returns_conflict_without_retrying(self):
        conflict = httpx.Response(409,
            json={"detail": "该幂等编号已用于其他请求，请核对内容"},
            request=httpx.Request("POST", "http://offline/human-tickets"))
        with patch.object(order_client.httpx, "post", return_value=conflict) as post:
            result = order_client.create_human_ticket(
                access_token="offline-token", summary="需要人工确认", reason="用户要求人工",
                conversation_id="offline-thread", idempotency_key="same-operation",
            )
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("error"))
        self.assertNotIn("ticket_id", result)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["json"]["idempotency_key"], "same-operation")

    def test_ticket_without_order_is_saved(self):
        self.register_test_conversation("offline-thread")
        response = self.client.post("/human-tickets", headers=self.headers(self.login()),
                                    json=self.ticket_payload())
        self.assertEqual(response.status_code, 201, response.text)
        with Session(self.engine) as session:
            ticket = session.get(server.HumanTicket, response.json()["ticket_id"])
            self.assertIsNone(ticket.order_id)
            self.assertEqual(ticket.user_id, "U1001")
            self.assertEqual(ticket.status, "待处理")

    def test_ticket_uses_authenticated_owner_and_server_status(self):
        self.register_test_conversation("offline-thread")
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
        self.assertIn("待处理", response.json()["message"])
        self.assertNotIn("已接听", response.json()["message"])

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
        from agent.langchain_agent_demo import AgentContext, build_agent
        from tests.test_project_smoke import ToolCallingModel, FakeEmbedding, FakeReranker
        from rag.rag_loader import load_documents, split_documents

        registered = []

        class RecordingModel(ToolCallingModel):
            def bind_tools(self, tools, **kwargs):
                registered.extend(tools)
                return self

        token = self.login()
        inputs = [("first-thread", None), ("new-thread", "A1001"), ("denied-thread", "A2001")]
        for thread_id, _ in inputs:
            self.register_test_conversation(thread_id)
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
                            embedding.encode(chunks), FakeReranker())

        def local_post(url, **kwargs):
            kwargs.pop("timeout", None)
            return self.client.post(url, **kwargs)

        config = {"configurable": {"thread_id": "first-thread"}}
        saved_ids = []
        with patch.object(order_client.httpx, "post", side_effect=local_post) as post:
            for number, (thread_id, order_id) in enumerate(inputs):
                config["configurable"]["thread_id"] = thread_id
                result = agent.invoke({"messages": [{"role": "user", "content": "请转人工"}]},
                                      config=config, context=AgentContext(access_token=token))
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
        agent = app.build_agent(model, [], FakeEmbedding(), [], FakeReranker())

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
            stack.enter_context(patch.object(server, "uuid4", side_effect=[
                "ticket-saved-thread", "offline-ticket-id", "recovery-thread",
            ]))
            for name in ["load_documents", "split_documents", "load_embedding_model", "build_chunk_vectors", "load_rerank_model", "ChatDeepSeek"]:
                stack.enter_context(patch.object(app, name))
            stack.enter_context(patch.object(app, "build_agent", return_value=agent))
            post = stack.enter_context(patch.object(order_client.httpx, "post", side_effect=local_post))
            app.main()
            self.assertEqual(post.call_count, 3)
            self.assertEqual([urlsplit(call.args[0]).path for call in post.call_args_list],
                             ["/conversations", "/human-tickets", "/conversations"])
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

    def test_shared_agent_uses_each_run_identity_for_all_business_tools(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from agent.langchain_agent_demo import AgentContext, build_agent
        from tests.test_project_smoke import ToolCallingModel, FakeEmbedding, FakeReranker

        # The return-entry tool now requires a valid receipt within seven days.
        # Keep these fixtures local to this identity check; other tests use shipped orders.
        with Session(self.engine) as session:
            for order_id in ("A1001", "A2001"):
                order = session.get(server.Order, order_id)
                order.status = "已签收"
                order.received_at = int(server.time.time()) - 2 * 24 * 60 * 60
                session.add(order)
            session.commit()

        alice_old = self.login("alice")
        bob = self.login("bob")
        alice_new = self.login("alice")
        cases = [
            ("alice-first", alice_old, "U1001", "A1001", "A1001", False),
            ("bob-next", bob, "U1002", "A2001", "A1001", False),
            ("alice-new-token", alice_new, "U1001", "A1001", "A1001", False),
            ("revoked-token", alice_old, "U1001", "A1001", "A1001", True),
        ]
        for thread_id, _, user_id, *_ in cases:
            self.register_test_conversation(thread_id, user_id)
        registered, responses, model_messages = [], [], []
        for index, (_, _, _, order_id, detail_id, _) in enumerate(cases):
            calls = [
                ("query_order", {"order_id": detail_id}),
                ("query_orders", {}),
                ("query_return_entry", {"order_id": order_id}),
                ("create_support_ticket", {
                    "summary": "请人工确认退货条件", "reason": "用户主动要求人工",
                    "order_id": order_id,
                }),
            ]
            responses.extend([
                AIMessage(content="", tool_calls=[{
                    "name": name, "args": args, "id": f"identity-{index}-{name}",
                    "type": "tool_call",
                } for name, args in calls]),
                AIMessage(content="模拟本轮结束"),
            ])

        class RecordingModel(ToolCallingModel):
            def bind_tools(self, tools, **kwargs):
                registered.extend(tools)
                return self

            def _generate(self, messages, *args, **kwargs):
                model_messages.extend(messages)
                return super()._generate(messages, *args, **kwargs)

        agent = build_agent(RecordingModel(responses=responses), [], FakeEmbedding(), [], FakeReranker())

        def local_request(method):
            def send(url, **kwargs):
                kwargs.pop("timeout", None)
                return getattr(self.client, method)(url, **kwargs)
            return send

        with patch.object(order_client.httpx, "get", side_effect=local_request("get")) as get, \
             patch.object(order_client.httpx, "post", side_effect=local_request("post")) as post:
            for index, (thread_id, token, user_id, order_id, detail_id, revoked) in enumerate(cases):
                with self.subTest(thread=thread_id):
                    get.reset_mock()
                    post.reset_mock()
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": "查询订单和入口，并按我的要求转人工"}]},
                        config={"configurable": {"thread_id": thread_id}},
                        context=AgentContext(access_token=token),
                    )
                    tools = {m.name: json.loads(m.content) for m in result["messages"] if isinstance(m, ToolMessage)}
                    self.assertEqual(get.call_count, 3)
                    self.assertEqual(post.call_count, 1)
                    for call in [*get.call_args_list, *post.call_args_list]:
                        self.assertEqual(call.kwargs["headers"], self.headers(token))
                    self.assertEqual(post.call_args.kwargs["json"]["conversation_id"], thread_id)
                    if revoked:
                        self.assertTrue(all(body["code"] == "unauthenticated" for body in tools.values()))
                    else:
                        self.assertTrue(all(order["user_id"] == user_id for order in tools["query_orders"]))
                        self.assertTrue(tools["query_orders"])
                        self.assertEqual(tools["query_return_entry"]["order_id"], order_id)
                        self.assertIn("entry_url", tools["query_return_entry"])
                        self.assertIn("ticket_id", tools["create_support_ticket"])
                        self.assertEqual(tools["create_support_ticket"]["order_id"], order_id)
                        if thread_id == "bob-next":
                            self.assertIn("error", tools["query_order"])
                        else:
                            self.assertEqual(tools["query_order"]["user_id"], user_id)
                    if index == 0:
                        self.assertEqual(self.client.post("/logout", headers=self.headers(alice_old)).status_code, 204)

        for tool in registered:
            properties = tool.tool_call_schema.model_json_schema()["properties"]
            self.assertNotIn("runtime", properties)
            self.assertNotIn("access_token", properties)
        for token in (alice_old, bob, alice_new):
            self.assertNotIn(token, repr(model_messages))
        with Session(self.engine) as session:
            tickets = session.exec(select(server.HumanTicket)).all()
            self.assertEqual(len(tickets), 3)
            self.assertEqual({t.conversation_id: (t.user_id, t.order_id) for t in tickets}, {
                "alice-first": ("U1001", "A1001"),
                "bob-next": ("U1002", "A2001"),
                "alice-new-token": ("U1001", "A1001"),
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
        with Session(self.engine) as session:
            order = session.get(server.Order, "A1001")
            order.status = "已签收"
            order.received_at = int(server.time.time()) - 2 * 24 * 60 * 60
            session.add(order)
            session.commit()

        def local_get(url, **kwargs):
            kwargs.pop("timeout", None)
            return self.client.get(url, **kwargs)

        token = self.login()
        with patch.object(order_client.httpx, "get", side_effect=local_get):
            entry = get_return_entry("A1001", token)
            self.assertEqual(entry["order_id"], "A1001")
            self.assertTrue(entry["is_demo"])
            self.assertIn("北京时间", entry["received_at_display"])
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
