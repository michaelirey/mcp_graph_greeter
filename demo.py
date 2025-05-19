#!/usr/bin/env python3
"""
Demo script for MCP Graph Greeter

This script demonstrates the MCP Graph Greeter in an interactive session.
"""
import asyncio
import logging
import sys
import uuid
from typing import List, Dict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from mcp_graph_greeter import build_greeter_graph, GreeterState, GreeterInput
from config.loader import load_server_configs, create_server_map, get_allowed_tools_map, get_sensitive_tools_map
from langchain_mcp_adapters.client import MultiServerMCPClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mcp_graph_greeter_demo")

async def initialize_graph():
    """Initialize the MCP Graph Greeter with proper configuration."""
    # Load server configurations
    server_configs = load_server_configs()
    if not server_configs:
        raise ValueError("No server configurations found. Please add at least one server configuration to the config/servers directory.")
    
    # Create maps for server endpoints, allowed tools, and sensitive tools
    server_map = create_server_map(server_configs)
    allowed_tools_map = get_allowed_tools_map(server_configs)
    sensitive_tools_map = get_sensitive_tools_map(server_configs)
    
    # Create a client for all servers
    client = MultiServerMCPClient(server_map)
    
    # Get all tools from all servers
    all_tools = []
    raw_tools = await client.get_tools()
    
    # Process and namespace tools
    all_sensitive_tools = []
    for tool in raw_tools:
        if tool.metadata is None:
            tool.metadata = {}
        
        server_name = tool.metadata.get("server_name")
        if not server_name:
            for server in server_map.keys():
                if tool.name in allowed_tools_map.get(server, []):
                    server_name = server
                    tool.metadata["server_name"] = server_name
                    break
            
            if not server_name:
                logger.warning(f"Tool {tool.name} missing server_name in metadata and couldn't determine server, skipping")
                continue
        
        if tool.name not in allowed_tools_map.get(server_name, []):
            continue
        
        tool.metadata["original_name"] = tool.name
        tool.name = f"{server_name}_{tool.name}"
        all_tools.append(tool)
        
        if tool.metadata["original_name"] in sensitive_tools_map.get(server_name, []):
            all_sensitive_tools.append(tool.name)
    
    # Create and return the graph
    return build_greeter_graph(all_tools, all_sensitive_tools)

async def interactive_session():
    """Run an interactive session with the MCP Graph Greeter."""
    print("\n📁 Welcome to the MCP Graph Greeter Demo 📁\n")
    print("This demo allows you to interact with a filesystem assistant powered by LangGraph and MCP.")
    print("You can ask about files in your current directory, create new files, and more.")

    try:
        # Initialize the graph
        print("\n🔄 Initializing the assistant...")
        graph = await initialize_graph()
        
        # Initialize state
        state: GreeterState = {"messages": []}
        
        # Initialize config with required checkpointer keys
        config: RunnableConfig = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "checkpoint_ns": "demo_session",
                "checkpoint_id": str(uuid.uuid4())
            }
        }

        # Continue the conversation
        while True:
            # Get user input - required
            while True:
                user_input = input("\n🧑 Your input: ")
                if user_input:
                    break
                print("⚠️  Input is required. Please try again.")

            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Thanks for using the MCP Graph Greeter Demo! Goodbye!")
                break

            # Get response
            print("\n🤖 Thinking...")
            input_data = {"messages": [HumanMessage(content=user_input)]}
            result = await graph.ainvoke(input_data, config)
            state["messages"] = result["messages"]

            # Print just the latest response
            latest_response = state["messages"][-1]
            print(f"\n🤖 Assistant: {latest_response.content}")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Error in interactive session: {str(e)}")
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    # Run the interactive session
    asyncio.run(interactive_session())
