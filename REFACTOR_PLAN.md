# Refactor Plan: Per-Server Configuration & Namespaced Tools

## Overview

This ticket covers the implementation of a data-driven mechanism for configuring MCP servers and their tools in the MCP Graph Greeter. Currently, server configurations and tool lists are hard-coded in `config.py`, which makes it difficult to add new servers or manage tool permissions.

## Requirements (from PRD)

The goal is to implement a system that:
1. Declares each server's endpoint, allowed tools, and sensitive tools in its own JSON file
2. Namespaces every tool as `<server_name>.<tool_name>` to guarantee uniqueness
3. Dynamically loads these configurations at runtime without code changes
4. Maintains backward compatibility for end-users

## Current Implementation Analysis

The current implementation has these limitations:
- Server endpoints are hard-coded in `config.py`
- All tools are stored in one flat list
- Tool names are not namespaced, risking collisions
- Adding a new server requires code changes and redeployment

Key files involved:
- `mcp_graph_greeter.py`: Main module that defines the graph workflow and factory function
- `config.py`: Contains server configurations and tool lists
- `greeter_service.py`: Service implementation for using the greeter outside CLI

## Implementation Plan

### 1. Create Server Configuration Schema and Directory

- Create a new directory `config/servers/` to store server configurations
- Implement a JSON schema for server configurations with fields:
  - `server_name` (string)
  - `endpoint` (object with command, args, transport)
  - `allowed_tools` (array of strings)
  - `sensitive_tools` (array of strings)

### 2. Create Configuration Loader

- Create a new module `config/loader.py` that:
  - Finds and loads all JSON files in `config/servers/`
  - Validates each file against the schema
  - Returns a list of validated server configurations
  - Handles error cases (duplicate server names, schema validation errors)

### 3. Refactor `mcp_graph_greeter.py`

- Update `graph_factory()` to:
  - Call the configuration loader
  - Build server map, allowed tool map, and sensitive tool map
  - Create `MultiServerMCPClient` with the server map
  - Fetch and namespace tools (`tool.name = f"{server}.{tool.name}"`)
  - Track original tool names in metadata
  - Filter tools based on allowed tools per server

- Update `review_tool_calls()` to:
  - Split the namespaced tool name to get server and tool parts
  - Check if the tool is in the sensitive list for its server

- Add a helper function `split_namespaced(name)` that returns the server and tool parts

### 4. Update Tests

- Update test cases to account for namespaced tools
- Add tests for the configuration loader
- Add tests for tool namespacing and sensitivity detection
- Ensure backward compatibility

### 5. Update Documentation

- Update README.md with information about server configuration
- Document the JSON schema for server configurations
- Add examples of adding new servers

## Technical Details

### JSON Schema

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

### Configuration Loading

The loader should:
- Handle file I/O and JSON parsing errors gracefully
- Validate each configuration against the schema
- Check for duplicate server names
- Return a list of validated configurations

### Tool Namespacing

Change tool names from:
```
"read_file" (from filesystem server)
"read_file" (from another server)
```

To:
```
"filesystem.read_file"
"another_server.read_file"
```

### Sensitive Tool Detection

When reviewing tool calls, extract the server and tool parts:
```python
server_name, tool_name = split_namespaced(tool_call["name"])
if tool_name in sensitive_tools_map.get(server_name, []):
    # Require human review
```

## Success Criteria

- Adding a new MCP server requires zero code changes (just a new JSON file)
- The graph boots successfully with one or more servers
- No namespace collisions are reported in logs
- Sensitive tool review flow still works correctly with namespaced tools
- All unit tests pass
- Functional tests show backward compatibility

## Out of Scope

- UI/CLI changes to expose server information
- Feature parity enhancements for individual tools

## Dependencies

- No external dependencies beyond what's already in the project

## Estimated Effort

- 3-4 days of implementation work

## References

- [PRD Document](file:///Users/michaelirey/Development/mcp_graph_greeter/prd.txt)
- [MultiServerMCPClient Documentation](https://langchain-mcp-adapters.readthedocs.io/)