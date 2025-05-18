"""
MCP Graph Greeter Service - Functions for using the greeter in applications

This module provides functions for creating and invoking the MCP Graph Greeter
outside of the LangGraph CLI.
"""

import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config.loader import (
    load_server_configs,
    create_server_map,
    get_allowed_tools_map,
    get_sensitive_tools_map,
    namespace_tool_name,
)
from mcp_graph_greeter import build_greeter_graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def mcp_graph_greeter() -> AsyncIterator:
    """
    Create a filesystem graph greeter with async context management.

    This function:
    1. Loads server configurations from JSON files
    2. Initializes the MCP client and connects to the MCP server
    3. Creates a LangGraph workflow with the model and tools
    4. Yields the graph for use
    5. Ensures the MCP session is properly closed

    Yields:
        A configured LangGraph for filesystem operations
    """
    # Load server configurations
    server_configs = load_server_configs()
    if not server_configs:
        logger.error("No server configurations found")
        raise ValueError("No server configurations found. Please add at least one server configuration.")
    
    # Create maps
    server_map = create_server_map(server_configs)
    allowed_tools_map = get_allowed_tools_map(server_configs)
    sensitive_tools_map = get_sensitive_tools_map(server_configs)
    
    # Use the first server in the list (usually filesystem)
    default_server = server_configs[0]["server_name"]
    logger.info(f"Using {default_server} as the default server for the graph greeter service")
    
    # Create client with all servers
    client = MultiServerMCPClient(server_map)
    
    # Use the first server for the session (typically filesystem for this service)
    async with client.session(default_server) as session:
        try:
            # Get tools for the session
            tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(tools)} tools from {default_server} server")
            
            # Namespace the tools
            namespaced_tools = []
            for tool in tools:
                # Skip if not in allowed tools list
                if tool.name not in allowed_tools_map.get(default_server, []):
                    continue
                
                # Initialize metadata if None
                if tool.metadata is None:
                    tool.metadata = {}
                
                # Preserve original name
                tool.metadata["original_name"] = tool.name
                tool.metadata["server_name"] = default_server
                
                # Add namespace using underscore instead of dot
                namespaced_name = namespace_tool_name(default_server, tool.name)
                tool.name = namespaced_name
                
                namespaced_tools.append(tool)
            
            logger.info(f"Using {len(namespaced_tools)} namespaced tools after filtering")
            
            # Create the graph with tools and sensitive map for this server
            graph = build_greeter_graph(
                namespaced_tools, 
                {default_server: sensitive_tools_map.get(default_server, [])}
            )
            
            logger.info(f"MCP Graph Greeter created successfully")
            
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