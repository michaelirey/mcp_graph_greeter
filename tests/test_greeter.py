import pytest
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

import mcp_graph_greeter as mg

class DummyGraph:
    async def ainvoke(self, initial_state):
        return {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]} 

@asynccontextmanager
async def dummy_greeter():
    yield DummyGraph()

@pytest.mark.asyncio
async def test_invoke_greeter_returns_messages(monkeypatch):
    monkeypatch.setattr(mg, "mcp_graph_greeter", dummy_greeter)
    messages = await mg.invoke_greeter("hi")
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

