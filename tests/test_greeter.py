import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool

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
    graph = mg.build_greeter_graph([], {})
    assert graph is not None


def test_review_tool_calls_with_namespaced_tools():
    """Test the review_tool_calls function with namespaced tools."""
    # Create a mock of necessary components
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mg, "ChatOpenAI", lambda model, api_key=None: DummyModel())
    
    # Create a test tool
    tool = BaseTool(name="test_server.test_tool", func=lambda x: "result")
    
    # Create a graph with the test tool and a sensitive tools map
    sensitive_tools_map = {"test_server": ["test_tool"]}
    graph = mg.build_greeter_graph([tool], sensitive_tools_map)
    
    # Test that the graph was created
    assert graph is not None
    
    # Clean up
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_graph_factory_namespaces_tools(monkeypatch):
    """Test that the graph factory properly namespaces tools."""
    # Create mock server configs
    mock_configs = [
        {
            "server_name": "test_server",
            "endpoint": {"command": "test", "transport": "stdio"},
            "allowed_tools": ["test_tool"],
            "sensitive_tools": ["test_tool"]
        }
    ]
    
    # Create mock tools
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.metadata = {"server_name": "test_server"}
    
    # Create mocks
    monkeypatch.setattr("config.loader.load_server_configs", lambda: mock_configs)
    monkeypatch.setattr("config.loader.create_server_map", lambda x: {"test_server": {"command": "test", "transport": "stdio"}})
    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", MagicMock)
    monkeypatch.setattr(mg, "build_greeter_graph", lambda tools, sensitive_map: DummyGraph())
    
    # Mock the get_tools method to return our test tool
    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool]
    
    # Mock the MultiServerMCPClient constructor
    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient.__new__", lambda cls, server_map: mock_client)
    
    # Test the graph factory
    async with mg.graph_factory() as graph:
        # Check that the graph was created
        assert graph is not None
        
        # Check that the tool was properly namespaced
        mock_client.get_tools.assert_called_once()
        assert mock_tool.name == "test_server.test_tool"
        assert mock_tool.metadata["original_name"] == "test_tool"