"""Offline integration checks for the migrated project."""
import importlib
from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import numpy as np
from openai import APITimeoutError, APIStatusError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
                with patch.object(server, "engine", engine), patch.object(seed_demo, "engine", engine):
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
        agent = build_agent(model, chunks, FakeEmbedding(), FakeEmbedding().encode(chunks), FakeReranker())
        first = agent.invoke(
            {"messages": [{"role": "user", "content": "发货后改地址"}]},
            config={"configurable": {"thread_id": "first"}},
            context=agent_app.AgentContext(access_token="test-token"),
        )
        tool_messages = [message for message in first["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(tool_messages[0].tool_call_id, "policy-call-1")
        self.assertIn("地址修改", tool_messages[0].content)
        self.assertEqual(first["messages"][-1].content, "模拟最终回答")
        second = agent.invoke(
            {"messages": [{"role": "user", "content": "你好"}]},
            config={"configurable": {"thread_id": "second"}},
            context=agent_app.AgentContext(access_token="test-token"),
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
        agent = build_agent(model, chunks, FakeEmbedding(), FakeEmbedding().encode(chunks), FakeReranker())
        with patch("agent.langchain_agent_demo.get_order_list", return_value=[{
            "order_id": "A1002", "product": "手机", "status": "已发货",
        }]) as lookup:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "查已发货的手机订单"}]},
                config={"configurable": {"thread_id": "order-list"}},
                context=agent_app.AgentContext(access_token=token),
            )
        lookup.assert_called_once_with(
            access_token=token, status="已发货", product_keyword="手机", offset=0, limit=10,
        )
        tools = {tool.name: tool for tool in registered}
        self.assertEqual(set(tools), {"query_order", "query_orders", "search_policy", "query_return_entry", "create_support_ticket"})
        self.assertEqual(set(tools["query_orders"].tool_call_schema.model_json_schema()["properties"]), {
            "status", "product_keyword", "offset", "limit",
        })
        self.assertEqual(set(tools["query_order"].tool_call_schema.model_json_schema()["properties"]), {"order_id"})
        self.assertEqual(set(tools["query_return_entry"].tool_call_schema.model_json_schema()["properties"]), {"order_id"})
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
                stack.enter_context(patch.object(agent_app, "create_conversation", return_value="registered-thread"))
                for name in ["load_documents", "split_documents", "load_embedding_model", "build_chunk_vectors", "load_rerank_model", "ChatDeepSeek"]:
                    stack.enter_context(patch.object(agent_app, name))
                fake_agent = Mock()
                stack.enter_context(patch.object(agent_app, "build_agent", return_value=fake_agent))
                validation = stack.enter_context(patch.object(agent_app, "is_logged_in", side_effect=checks))
                agent_app.main()
                fake_agent.invoke.assert_not_called()
                self.assertEqual(validation.call_count, len(checks))
                output.assert_any_call("登录已失效，请重新运行程序并登录")

    def test_chat_turn_passes_context_and_counts_only_reported_new_usage(self):
        history = [AIMessage(content="旧回答", usage_metadata={
            "input_tokens": 900, "output_tokens": 99, "total_tokens": 999,
        })]
        messages = history + [
            HumanMessage(content="查订单"),
            AIMessage(content="", usage_metadata={
                "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            }, tool_calls=[{
                "name": "query_order", "args": {"order_id": "A1001"},
                "id": "usage-call", "type": "tool_call",
            }]),
            ToolMessage(content="已发货", name="query_order", tool_call_id="usage-call"),
            AIMessage(content="无用量记录的消息"),
            AIMessage(content="订单已发货", usage_metadata={
                "input_tokens": 20, "output_tokens": 3, "total_tokens": 23,
            }),
        ]
        fake_agent = Mock()
        fake_agent.get_state.return_value = SimpleNamespace(values={"messages": history})
        fake_agent.invoke.return_value = {"messages": messages}
        config = {"configurable": {"thread_id": "usage-thread"}}
        context = agent_app.AgentContext(access_token="private-context-token")
        with patch("builtins.print") as output, \
             patch("builtins.input", side_effect=AssertionError("Unexpected terminal input")), \
             patch.object(agent_app, "perf_counter", side_effect=[1.0, 2.0]):
            answer = agent_app.run_chat_turn(fake_agent, "查订单", config, context=context)
        self.assertEqual(answer, "订单已发货")
        fake_agent.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "查订单"}]},
            config=config, context=context,
        )
        output.assert_any_call("本轮已报告token用量：输入30,输出5,合计35")
        output.assert_any_call("本轮Agent执行耗时：1.00秒")
        self.assertNotIn(context.access_token, str(output.call_args_list))
        self.assertFalse(any(c.args[0] == "最终回答：" for c in output.call_args_list))

    def test_chat_request_errors_switch_thread_and_do_not_retry_question(self):
        request = httpx.Request("POST", "http://example.test/api")
        errors = [
            httpx.ReadTimeout("simulated timeout", request=request),
            httpx.HTTPStatusError("simulated status", request=request,
                                  response=httpx.Response(503, request=request)),
            APITimeoutError(request=request),
            APIStatusError("simulated model status", response=httpx.Response(429, request=request), body=None),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__), ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-test-key"}))
                stack.enter_context(patch("builtins.input", side_effect=["alice", "创建工单", "你好", "q"]))
                output = stack.enter_context(patch("builtins.print"))
                stack.enter_context(patch.object(agent_app, "getpass", return_value="test-password"))
                stack.enter_context(patch.object(agent_app, "login", return_value={"access_token": "private-test-token", "username": "alice"}))
                stack.enter_context(patch.object(agent_app, "is_logged_in", return_value=True))
                stack.enter_context(patch.object(agent_app, "create_conversation", side_effect=["failed-thread", "fresh-thread"]))
                timer = stack.enter_context(patch.object(agent_app, "perf_counter", side_effect=[10.0, 12.5, 20.0, 23.0]))
                for name in ["load_documents", "split_documents", "load_embedding_model", "build_chunk_vectors", "load_rerank_model", "ChatDeepSeek"]:
                    stack.enter_context(patch.object(agent_app, name))
                calls = []

                def invoke(data, config, context):
                    self.assertEqual(context.access_token, "private-test-token")
                    calls.append((data["messages"][0]["content"], config["configurable"]["thread_id"]))
                    if len(calls) == 1:
                        raise error
                    return {"messages": [AIMessage(content="恢复后的回答")]}

                fake_agent = Mock()
                fake_agent.get_state.return_value = SimpleNamespace(values={"messages": []})
                fake_agent.invoke.side_effect = invoke
                stack.enter_context(patch.object(agent_app, "build_agent", return_value=fake_agent))
                agent_app.main()
                self.assertEqual(calls, [("创建工单", "failed-thread"), ("你好", "fresh-thread")])
                output.assert_any_call("本轮请求未能正常完成，异常类型：", type(error).__name__)
                output.assert_any_call("已切换到新会话，编号：", "fresh-thread")
                output.assert_any_call("如果涉及工单创建，请先核对保存结果，再决定是否重新提交。")
                output.assert_any_call("本轮Agent执行耗时：2.50秒")
                output.assert_any_call("本轮Agent执行耗时：3.00秒")
                self.assertEqual(timer.call_count, 4)
                answers = [c for c in output.call_args_list if c.args[0] == "最终回答："]
                self.assertEqual(len(answers), 1)
                self.assertEqual(answers[0].args[1], "恢复后的回答")
                self.assertNotIn("private-test-token", str(output.call_args_list))

    def test_login_loop_retries_before_initialization_and_returns_on_quit(self):
        request = httpx.Request("POST", "http://example.test/login")
        error = httpx.HTTPStatusError("wrong password", request=request,
                                     response=httpx.Response(401, request=request))
        cases = [
            (["alice", "alice", "q"], ["wrong", "correct"],
             [error, {"access_token": "private-test-token", "username": "alice"}], 2, 1),
            ([" Q "], [], [], 0, 0),
            (["alice"], ["q"], [], 0, 0),
        ]
        for questions, passwords, responses, attempts, initializations in cases:
            with self.subTest(attempts=attempts, questions=questions), ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-test-key"}))
                stack.enter_context(patch("builtins.input", side_effect=questions))
                output = stack.enter_context(patch("builtins.print"))
                password_input = stack.enter_context(patch.object(agent_app, "getpass", side_effect=passwords))
                login = stack.enter_context(patch.object(agent_app, "login", side_effect=responses))
                creation = stack.enter_context(patch.object(agent_app, "create_conversation", return_value="registered-thread"))
                resources = [stack.enter_context(patch.object(agent_app, name)) for name in
                             ["load_documents", "split_documents", "load_embedding_model", "build_chunk_vectors", "load_rerank_model", "ChatDeepSeek"]]
                fake = Mock()
                build = stack.enter_context(patch.object(agent_app, "build_agent", return_value=fake))
                agent_app.main()
                self.assertEqual(login.call_count, attempts)
                self.assertEqual(password_input.call_count, len(passwords))
                self.assertEqual(build.call_count, initializations)
                self.assertEqual(creation.call_count, initializations)
                self.assertTrue(all(resource.call_count == initializations for resource in resources))
                fake.invoke.assert_not_called()
                if initializations:
                    self.assertEqual(len(build.call_args.args), 5)
                    self.assertNotIn("access_token", build.call_args.kwargs)
                    output.assert_any_call("已登录：", "alice")
                self.assertNotIn("private-test-token", str(output.call_args_list))


if __name__ == "__main__":
    unittest.main()
