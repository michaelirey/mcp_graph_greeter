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

## Requirements

- Python 3.11+
- LangChain
- LangGraph
- langchain-mcp-adapters
- Node.js (for the MCP filesystem server)
- uvx (for running the LangGraph CLI)

## Installation

1. Clone this repository
2. Install the LangGraph CLI and dependencies using uvx:

```bash
pip install uvx
```

3. Install the MCP filesystem server:

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file
   - Optionally change the model name from the default "gpt-4.1" to another OpenAI model

## Usage

### Running with LangGraph CLI

The recommended way to run this demo is using the LangGraph CLI, which provides a web UI for interacting with the graph:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

This will:
1. Start the LangGraph development server
2. Launch the MCP filesystem server
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
python demo.py
```

### Using in Your Own Code

You can also use the greeter programmatically in your own code:

```python
import asyncio
from langchain_core.messages import HumanMessage
from mcp_graph_greeter import invoke_greeter

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

- `mcp_graph_greeter.py` - Main implementation of the graph
- `config.py` - Configuration for the MCP server
- `demo.py` - Command-line demo script
- `langgraph.json` - Configuration for the LangGraph CLI
- `__init__.py` - Package definition

## Security Considerations

This demo only allows access to the current working directory for safety reasons. The MCP filesystem server restricts file operations to this directory tree.

## Acknowledgements

This project builds upon the work of the LangGraph and Model Context Protocol (MCP) projects, using the integration provided by langchain-mcp-adapters.
