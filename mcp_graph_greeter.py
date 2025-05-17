"""Entry points for the MCP Graph Greeter package."""

from langchain_openai import ChatOpenAI

from . import core
from .core import GreeterState, build_greeter_graph
from .api import mcp_graph_greeter, invoke_greeter
from .cli import graph_factory

# expose ChatOpenAI so tests can monkeypatch it
core.ChatOpenAI = ChatOpenAI

__all__ = [
    "GreeterState",
    "build_greeter_graph",
    "mcp_graph_greeter",
    "invoke_greeter",
    "graph_factory",
]
