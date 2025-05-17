"""API helpers for interacting with the greeter programmatically."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config import FILESYSTEM_SERVER
from .core import build_greeter_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def mcp_graph_greeter() -> AsyncIterator:
    """Create the greeter graph and manage MCP resources."""

    client = MultiServerMCPClient(FILESYSTEM_SERVER)
    async with client.session("filesystem") as session:
        try:
            filesystem_tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(filesystem_tools)} filesystem tools")
            graph = build_greeter_graph(filesystem_tools)
            logger.info("MCP Graph Greeter created successfully")
            yield graph
        except Exception as e:
            logger.error(f"Error creating MCP Graph Greeter: {str(e)}")
            raise
        finally:
            logger.info("MCP Graph Greeter closed")


async def invoke_greeter(
    message: str, context_messages: Optional[List[BaseMessage]] = None
) -> List[BaseMessage]:
    """Invoke the greeter with a message."""

    try:
        async with mcp_graph_greeter() as graph:
            messages: List[BaseMessage] = []
            if context_messages:
                messages.extend(context_messages)
            messages.append(HumanMessage(content=message))
            initial_state = {"messages": messages}
            result = await graph.ainvoke(initial_state)
            logger.info("Graph execution completed successfully")
            return result["messages"]
    except Exception as e:
        logger.error(f"Error running MCP Graph Greeter: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return [HumanMessage(content=message), AIMessage(content=f"Error: {str(e)}")]
