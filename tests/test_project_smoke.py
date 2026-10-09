"""Offline integration checks for the migrated project."""
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from sqlmodel import create_engine

from agent.langchain_agent_demo import build_agent
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
                "agent.langchain_agent_demo", "rag.rag_loader", "rag.rag_search",
                "rag.embedding_demo", "rag.rerank_demo", "rag.policy_search",
                "rag.eval_retrieval",
            ):
                importlib.import_module(module)

    def test_order_api_uses_isolated_database_and_preserves_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "orders.db"
            engine = create_engine(
                f"sqlite:///{database.as_posix()}",
                connect_args={"check_same_thread": False},
            )
            try:
                with patch.object(server, "database_file", database), patch.object(server, "engine", engine):
                    with TestClient(server.app) as client:
                        self.assertEqual(client.get("/orders").json(), [])
                        order = {"order_id": "A1001", "product": "演示商品", "status": "已发货"}
                        self.assertEqual(client.post("/orders", json=order).status_code, 201)
                        self.assertEqual(client.post("/orders", json=order).status_code, 409)
                        self.assertEqual(client.get("/orders/A1001").json(), order)
                        self.assertEqual(len(client.get("/orders", params={"status": "已发货"}).json()), 1)
                        self.assertEqual(client.get("/orders/missing").status_code, 404)
                        self.assertEqual(client.patch("/orders/A1001/status", json={"status": "已取消"}).status_code, 200)
                        self.assertEqual(client.patch("/orders/A1001/status", json={"status": "待发货"}).status_code, 403)
                        self.assertEqual(client.get("/orders/A1001").json()["status"], "已取消")
            finally:
                engine.dispose()

    def test_order_adapter_distinguishes_not_found_and_server_error(self):
        request = httpx.Request("GET", "http://example.test/orders/A1001")
        with patch("agent.order_client.httpx.get", return_value=httpx.Response(404, request=request)):
            self.assertEqual(get_order("A1001")["error"], "订单不存在")
        with patch("agent.order_client.httpx.get", return_value=httpx.Response(503, request=request)):
            with self.assertRaises(httpx.HTTPStatusError):
                get_order("A1001")

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
        agent = build_agent(model, chunks, FakeEmbedding(), FakeEmbedding().encode(chunks), FakeReranker())
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


if __name__ == "__main__":
    unittest.main()
