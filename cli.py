"""Support for running the greeter via the LangGraph CLI."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config import FILESYSTEM_SERVER
from .core import build_greeter_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def graph_factory() -> AsyncIterator:
    """Graph factory function for LangGraph CLI."""

    logger.info("Initializing MCP Graph Greeter for LangGraph CLI")
    client = MultiServerMCPClient(FILESYSTEM_SERVER)

    try:
        async with client.session("filesystem") as session:
            filesystem_tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(filesystem_tools)} filesystem tools")
            graph = build_greeter_graph(filesystem_tools)
            logger.info("MCP Graph Greeter created successfully")
            yield graph
    except Exception as e:
        logger.error(f"Error creating MCP Graph Greeter: {str(e)}")
        raise
    finally:
        logger.info("MCP Graph Greeter factory closing")
