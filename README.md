# MCP Graph Greeter

A demonstration of using LangGraph with Model Context Protocol (MCP) for filesystem operations.

## Overview

This project showcases how to integrate LangGraph with MCP tools to create an interactive assistant that can help users explore and manage their filesystem. The implementation follows the recommended pattern for using the MultiServerMCPClient from the langchain-mcp-adapters library.

## Features

- Interactive greeting and personalization
- Filesystem exploration capabilities
- File reading and creation
- Proper resource management with async context managers
- Structured LangGraph workflow
- Server configuration via JSON files
- Tool namespacing for uniqueness across servers

## Requirements

- Python 3.11+
- LangChain
- LangGraph
- langchain-mcp-adapters
- Node.js (for the MCP filesystem server)
- uvx (for running the LangGraph CLI)

## Installation

1. Clone this repository
2. Install the package and its dependencies:

```bash
# Using uv (recommended)
uv install -e .

# Or using pip
pip install -e .
```

3. Install the LangGraph CLI:

```bash
pip install uvx
```

4. Install the MCP filesystem server:

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

5. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file
   - Optionally change the model name from the default "gpt-4.1" to another OpenAI model

## Server Configuration

The MCP Graph Greeter uses a data-driven approach for configuring MCP servers. Each server has its own JSON configuration file in the `config/servers/` directory.

### Configuration Format

Each server configuration file should follow this structure:

```json
{
  "server_name": "filesystem",
  "endpoint": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"],
    "transport": "stdio"
  },
  "allowed_tools": ["read_file", "write_file", "list_directory"],
  "sensitive_tools": ["write_file"]
}
```

### Adding a New Server

To add a new MCP server:

1. Create a new JSON file in the `config/servers/` directory
2. Define the server_name, endpoint, allowed_tools, and sensitive_tools
3. Restart the application

No code changes are required to add a new server.

### Tool Namespacing

All tools are namespaced using the pattern `<server_name>_<tool_name>` to ensure uniqueness. For example, the `read_file` tool from the `filesystem` server becomes `filesystem_read_file`. This prevents conflicts when multiple servers provide tools with the same name and ensures compatibility with OpenAI's tool name requirements.

## Usage

### Running with LangGraph CLI

The recommended way to run this demo is using the LangGraph CLI, which provides a web UI for interacting with the graph:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

This will:
1. Start the LangGraph development server
2. Launch the MCP servers defined in the configuration files
3. Create the graph with all necessary tools
4. Open a web interface for interacting with the assistant

Once the server is running, you can access the web UI at http://localhost:8000 and:
1. Select the "mcp_greeter" graph
2. Start a new chat
3. Begin with an introduction like "Hello, my name is Alice"
4. Ask about the filesystem with queries like "What files are in the current directory?"

### Running the Demo Script

Alternatively, you can use the included demo script for a command-line experience:

```bash
uv run python demo.py
```

### Using in Your Own Code

You can also use the greeter programmatically in your own code:

```python
import asyncio
from langchain_core.messages import HumanMessage
from greeter_service import invoke_greeter

async def main():
    # Start a conversation
    result = await invoke_greeter("Hello, my name is Alice")
    
    # Print the messages
    for message in result:
        role = "User" if isinstance(message, HumanMessage) else "AI"
        print(f"{role}: {message.content}")
    
    # Continue the conversation
    user_input = "What files are in the current directory?"
    result = await invoke_greeter(user_input, context_messages=result)
    
    # Print the response
    for message in result[len(result)-1:]:  # Just print the new message
        role = "User" if isinstance(message, HumanMessage) else "AI"
        print(f"{role}: {message.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

- `mcp_graph_greeter.py` - Main implementation of the graph and LangGraph CLI entry point
- `greeter_service.py` - Service functions for using the greeter in applications
- `config/` - Configuration package
  - `__init__.py` - Basic configuration and environment variables
  - `loader.py` - Server configuration loader
  - `schema.py` - JSON schema for server configurations
  - `servers/` - Directory containing server configuration files
    - `filesystem.json` - Filesystem server configuration
    - `context7.json` - Context7 server configuration
    - `shell.json` - Shell command server configuration
- `demo.py` - Command-line demo script
- `langgraph.json` - Configuration for the LangGraph CLI
- `__init__.py` - Package definition

## Security Considerations

This demo only allows access to the current working directory for safety reasons. The MCP filesystem server restricts file operations to this directory tree.

The configuration system lets you mark specific tools as sensitive, requiring human approval before execution.

## Acknowledgements

This project builds upon the work of the LangGraph and Model Context Protocol (MCP) projects, using the integration provided by langchain-mcp-adapters.