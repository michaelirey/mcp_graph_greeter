import pytest
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

import mcp_graph_greeter as mg
import greeter_service as gs


class DummyGraph:
    async def ainvoke(self, initial_state):
        return {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}


@asynccontextmanager
async def dummy_greeter():
    yield DummyGraph()


@pytest.mark.asyncio
async def test_invoke_greeter_returns_messages(monkeypatch):
    monkeypatch.setattr(gs, "mcp_graph_greeter", dummy_greeter)
    messages = await gs.invoke_greeter("hi")
    assert isinstance(messages, list)
    assert all(isinstance(m, BaseMessage) for m in messages)


class DummyModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config):
        return AIMessage(content="ok")

    async def ainvoke(self, messages, config):
        return AIMessage(content="ok")


def test_build_greeter_graph_compiles(monkeypatch):
    monkeypatch.setattr(mg, "ChatOpenAI", lambda model, api_key=None: DummyModel())
    graph = mg.build_greeter_graph([])
    assert graph is not None


def test_load_server_configs():
    configs = mg.load_server_configs()
    assert any(c.server_name == "filesystem" for c in configs)


@pytest.mark.asyncio
async def test_graph_factory_namespaces(monkeypatch):
    fs = mg.ServerConfig(
        server_name="fs",
        endpoint={},
        allowed_tools=["write", "read"],
        sensitive_tools=["write"],
    )
    sh = mg.ServerConfig(
        server_name="sh",
        endpoint={},
        allowed_tools=["shell"],
        sensitive_tools=["shell"],
    )

    monkeypatch.setattr(mg, "load_server_configs", lambda: [fs, sh])

    class DummyTool:
        def __init__(self, name, server):
            self.name = name
            self.metadata = {"server": server}

    class DummyClient:
        def __init__(self, servers):
            pass

        async def get_tools(self):
            return [DummyTool("write", "fs"), DummyTool("shell", "sh")]

    monkeypatch.setattr(mg, "MultiServerMCPClient", DummyClient)
    monkeypatch.setattr(mg, "ChatOpenAI", lambda model, api_key=None: DummyModel())

    captured = {}

    def fake_build(tools, sensitive_tools=None):
        captured["names"] = [t.name for t in tools]
        captured["sens"] = sensitive_tools
        return DummyGraph()

    monkeypatch.setattr(mg, "build_greeter_graph", fake_build)

    async with mg.graph_factory() as _:
        pass

    assert captured["names"] == ["fs.write", "sh.shell"]
    assert captured["sens"] == ["fs.write", "sh.shell"]


@pytest.mark.asyncio
async def test_graph_factory_metadata_fallback(monkeypatch):
    fs = mg.ServerConfig(
        server_name="fs",
        endpoint={},
        allowed_tools=["write"],
        sensitive_tools=[],
    )

    monkeypatch.setattr(mg, "load_server_configs", lambda: [fs])

    class DummyTool:
        def __init__(self, name):
            self.name = name
            self.metadata = {}

    class DummyClient:
        def __init__(self, servers):
            pass

        async def get_tools(self):
            return [DummyTool("write")]

    monkeypatch.setattr(mg, "MultiServerMCPClient", DummyClient)
    monkeypatch.setattr(mg, "ChatOpenAI", lambda model, api_key=None: DummyModel())

    captured = {}

    def fake_build(tools, sensitive_tools=None):
        captured["names"] = [t.name for t in tools]
        return DummyGraph()

    monkeypatch.setattr(mg, "build_greeter_graph", fake_build)

    async with mg.graph_factory() as _:
        pass

    assert captured["names"] == ["fs.write"]

