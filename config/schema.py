"""
JSON schema for MCP server configurations
"""

SERVER_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["server_name", "endpoint", "allowed_tools", "sensitive_tools"],
    "properties": {
        "server_name": {
            "type": "string",
            "description": "Unique name for the server"
        },
        "endpoint": {
            "type": "object",
            "required": ["command", "transport"],
            "properties": {
                "command": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "env": {"type": "object"},
                "transport": {"type": "string", "enum": ["stdio"]}
            }
        },
        "allowed_tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tools allowed from this server"
        },
        "sensitive_tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tools requiring human approval"
        }
    },
    "additionalProperties": False
}