"""Checkpoint connections must live until shutdown and close on startup failure."""
from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import sqlmodel_server as server


class CheckpointLifecycleTests(TestCase):
    def check_lifecycle(self, fail_initialization=False):
        events = []
        saver = Mock()
        saver.setup.side_effect = lambda: events.append("setup")
        agent = object()

        @contextmanager
        def connection(_):
            events.append("open")
            try:
                yield saver
            finally:
                events.append("close")

        def initialize(*, checkpointer):
            self.assertIs(checkpointer, saver)
            self.assertEqual(events, ["open", "setup"])
            events.append("agent")
            if fail_initialization:
                raise RuntimeError("simulated initialization failure")
            return agent

        app = FastAPI(lifespan=server.lifespan)
        with patch.object(server, "create_db_and_tables") as tables, \
             patch.object(server.PostgresSaver, "from_conn_string", side_effect=connection) as factory, \
             patch("agent.langchain_agent_demo.initialize_agent", side_effect=initialize):
            if fail_initialization:
                with self.assertRaisesRegex(RuntimeError, "simulated initialization failure"):
                    with TestClient(app):
                        self.fail("Startup failure must prevent serving requests")
            else:
                with TestClient(app):
                    self.assertIs(app.state.agent, agent)
                    self.assertEqual(events, ["open", "setup", "agent"])
            tables.assert_called_once_with()
            factory.assert_called_once_with(server.CHECKPOINT_DATABASE_URL)
        self.assertEqual(events, ["open", "setup", "agent", "close"])

    def test_connection_is_kept_open_while_serving(self):
        self.check_lifecycle()

    def test_connection_closes_when_agent_initialization_fails(self):
        self.check_lifecycle(fail_initialization=True)
