"""Offline integration checks for the migrated project."""
import importlib
from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from sqlmodel import Session, create_engine, select

from agent.langchain_agent_demo import build_agent
from agent import langchain_agent_demo as agent_app
from agent.order_client import get_order
from backend import sqlmodel_server as server
from rag.policy_search import retrieve_policy
from rag.rag_loader import load_documents, split_documents
from rag.rag_search import merge_results


class FakeEmbedding:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


class FakeReranker:
    def predict(self, pairs):
        return np.array([10.0 if "章节:地址修改" in text else -1.0 for _, text in pairs])


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ProjectSmokeTests(unittest.TestCase):
    def test_imports_have_no_startup_input_or_model_download(self):
        with patch("builtins.input", side_effect=AssertionError("Import requested input")):
            for module in (
                "settings", "backend.sqlmodel_server", "agent.order_client",
                "agent.langchain_agent_demo", "agent.business_entry", "rag.rag_loader", "rag.rag_search",
                "rag.embedding_demo", "rag.rerank_demo", "rag.policy_search",
                "rag.eval_retrieval",
            ):
                importlib.import_module(module)

    def test_seed_is_repeatable_and_records_order_ownership(self):
        from backend import seed_demo

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "orders.db"
            engine = create_engine(
                f"sqlite:///{database.as_posix()}",
                connect_args={"check_same_thread": False},
            )
            try:
                with patch.object(server, "database_file", database), patch.object(server, "engine", engine), patch.object(seed_demo, "engine", engine):
                    seed_demo.seed_demo()
                    seed_demo.seed_demo()
                    with Session(engine) as session:
                        self.assertEqual(len(session.exec(select(server.User)).all()), 2)
                        orders = session.exec(select(server.Order)).all()
                        self.assertEqual({o.order_id: o.user_id for o in orders}, {
                            "A1001": "U1001", "A1002": "U1001", "A2001": "U1002",
                        })
                        user = session.get(server.User, "U1001")
                        self.assertTrue(server.password_hasher.verify("demo-password", user.password_hash))
            finally:
                engine.dispose()

    def test_order_adapter_distinguishes_not_found_and_server_error(self):
        request = httpx.Request("GET", "http://example.test/orders/A1001")
        with patch("agent.order_client.httpx.get", return_value=httpx.Response(404, request=request)):
            self.assertEqual(get_order("A1001", "test-token")["error"], "订单不存在")
        with patch("agent.order_client.httpx.get", return_value=httpx.Response(503, request=request)):
            with self.assertRaises(httpx.HTTPStatusError):
                get_order("A1001", "test-token")

    def test_documents_and_rag_pipeline_preserve_chunk_identity(self):
        documents = load_documents()
        chunks = split_documents(documents)
        self.assertEqual(len(documents), 3)
        self.assertEqual(len(chunks), 12)
        address = next(chunk for chunk in chunks if chunk["section"] == "地址修改")
        self.assertEqual(merge_results([address], [{"chunk": address, "score": 0.9}]), [address])
        vectors = FakeEmbedding().encode(chunks)
        result = retrieve_policy(
            "发货后能改收货地址吗？", chunks, FakeEmbedding(), vectors, FakeReranker()
        )
        self.assertEqual(result["chunks"][0], address)
        self.assertEqual(len(result["chunks"]), 3)
        self.assertEqual(
            retrieve_policy("空知识库", [], FakeEmbedding(), np.empty((0, 2)), FakeReranker()),
            {"chunks": []},
        )

    def test_agent_tool_results_are_paired_and_threads_are_isolated(self):
        chunks = split_documents(load_documents())
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "search_policy", "args": {"query": "发货后改地址"},
                "id": "policy-call-1", "type": "tool_call",
            }]),
            AIMessage(content="模拟最终回答"),
            AIMessage(content="新会话模拟回答"),
        ])
        agent = build_agent(model, chunks, FakeEmbedding(), FakeEmbedding().encode(chunks), FakeReranker(), "test-token")
        first = agent.invoke(
            {"messages": [{"role": "user", "content": "发货后改地址"}]},
            config={"configurable": {"thread_id": "first"}},
        )
        tool_messages = [message for message in first["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(tool_messages[0].tool_call_id, "policy-call-1")
        self.assertIn("地址修改", tool_messages[0].content)
        self.assertEqual(first["messages"][-1].content, "模拟最终回答")
        second = agent.invoke(
            {"messages": [{"role": "user", "content": "你好"}]},
            config={"configurable": {"thread_id": "second"}},
        )
        self.assertEqual(len(second["messages"]), 2)
        self.assertEqual(len(first["messages"]), 4)

    def test_order_list_tool_is_registered_and_uses_program_token(self):
        registered = []

        class RecordingModel(ToolCallingModel):
            def bind_tools(self, tools, **kwargs):
                registered.extend(tools)
                return self

        model = RecordingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "query_orders",
                "args": {"status": "已发货", "product_keyword": "手机"},
                "id": "list-call-1", "type": "tool_call",
            }]),
            AIMessage(content="找到手机订单 A1002"),
        ])
        chunks = split_documents(load_documents())
        token = "program-test-token"
        agent = build_agent(model, chunks, FakeEmbedding(), FakeEmbedding().encode(chunks), FakeReranker(), token)
        with patch("agent.langchain_agent_demo.get_order_list", return_value=[{
            "order_id": "A1002", "product": "手机", "status": "已发货",
        }]) as lookup:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "查已发货的手机订单"}]},
                config={"configurable": {"thread_id": "order-list"}},
            )
        lookup.assert_called_once_with(
            access_token=token, status="已发货", product_keyword="手机", offset=0, limit=10,
        )
        tools = {tool.name: tool for tool in registered}
        self.assertEqual(set(tools), {"query_order", "query_orders", "search_policy"})
        self.assertEqual(set(tools["query_orders"].get_input_schema().model_json_schema()["properties"]), {
            "status", "product_keyword", "offset", "limit",
        })
        self.assertEqual(set(tools["query_order"].get_input_schema().model_json_schema()["properties"]), {"order_id"})
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(messages[0].tool_call_id, "list-call-1")
        self.assertIn("A1002", messages[0].content)
        self.assertNotIn(token, messages[0].content)

    def test_chat_does_not_invoke_agent_after_expiry_or_validation_failure(self):
        for checks, questions in [
            ([False], ["alice", "查订单"]),
            ([httpx.ConnectError("simulated"), False], ["alice", "查订单", "再查订单"]),
        ]:
            with self.subTest(checks=len(checks)), ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-test-key"}))
                stack.enter_context(patch("builtins.input", side_effect=questions))
                output = stack.enter_context(patch("builtins.print"))
                stack.enter_context(patch.object(agent_app, "getpass", return_value="test-password"))
                stack.enter_context(patch.object(agent_app, "login", return_value={"access_token": "test-token", "username": "alice"}))
                for name in ["load_documents", "split_documents", "load_embedding_model", "build_chunk_vectors", "load_rerank_model", "ChatDeepSeek"]:
                    stack.enter_context(patch.object(agent_app, name))
                fake_agent = Mock()
                stack.enter_context(patch.object(agent_app, "build_agent", return_value=fake_agent))
                validation = stack.enter_context(patch.object(agent_app, "is_logged_in", side_effect=checks))
                agent_app.main()
                fake_agent.invoke.assert_not_called()
                self.assertEqual(validation.call_count, len(checks))
                output.assert_any_call("登录已失效，请重新运行程序并登录")


if __name__ == "__main__":
    unittest.main()
