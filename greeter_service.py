"""
MCP Graph Greeter Service - Functions for using the greeter in applications

This module provides functions for creating and invoking the MCP Graph Greeter
outside of the LangGraph CLI.
"""

import logging
from typing import List, Optional
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config import FILESYSTEM_SERVER
from mcp_graph_greeter import build_greeter_graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def mcp_graph_greeter() -> AsyncIterator:
    """
    Create a filesystem graph greeter with async context management.

    This function:
    1. Initializes the MCP client and connects to the MCP server using session
    2. Creates a LangGraph workflow with the model and filesystem tools
    3. Yields the graph for use
    4. Ensures the MCP session is properly closed

    Follows the recommended pattern from langchain-mcp-adapters 0.1.0.

    Yields:
        A configured LangGraph for filesystem operations
    """
    # Create MCP client (not using it as a context manager)
    client = MultiServerMCPClient(FILESYSTEM_SERVER)

    # Use the client's session method for the filesystem server
    async with client.session("filesystem") as session:
        # Get tools using the load_mcp_tools function
        try:
            filesystem_tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(filesystem_tools)} filesystem tools")

            # Sensitive tools requiring human approval
            sensitive_tools = ["read_file", "get_file_info"]
            
            # Create the graph with the tools
            graph = build_greeter_graph(filesystem_tools, sensitive_tools)
            logger.info(f"MCP Graph Greeter created successfully with {len(sensitive_tools)} sensitive tools")

            # Yield the graph - the session will remain active during this context
            yield graph
        except Exception as e:
            logger.error(f"Error creating MCP Graph Greeter: {str(e)}")
            raise
        finally:
            # Clean up is handled by async context manager of the session
            logger.info("MCP Graph Greeter closed")


async def invoke_greeter(
    message: str,
    context_messages: Optional[List[BaseMessage]] = None,
) -> List[BaseMessage]:
    """
    Invoke the MCP Graph Greeter with a message.

    Args:
        message: User's message
        context_messages: Optional list of previous messages for context

    Returns:
        The complete conversation history
    """
    try:
        # Create the graph with the async context manager
        async with mcp_graph_greeter() as graph:
            # Prepare initial messages
            messages = []

            # Add context messages if provided
            if context_messages:
                messages.extend(context_messages)

            # Add the human message
            messages.append(HumanMessage(content=message))

            # Create initial state
            initial_state = {"messages": messages}

            # Run the graph asynchronously
            result = await graph.ainvoke(initial_state)
            logger.info("Graph execution completed successfully")
            return result["messages"]

    except Exception as e:
        # Log the error
        logger.error(f"Error running MCP Graph Greeter: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

        # Return a simple message with the error
        return [HumanMessage(content=message), AIMessage(content=f"Error: {str(e)}")]