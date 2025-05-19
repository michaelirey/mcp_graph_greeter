"""
Configuration loader for MCP Graph Greeter

Loads server configurations from JSON files in the config/servers directory.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logger = logging.getLogger(__name__)

def load_server_configs() -> List[Dict[str, Any]]:
    """
    Load server configurations from JSON files in the config/servers directory.
    
    Returns:
        List of server configurations
    
    Raises:
        ValueError: If there are validation errors or duplicate server names
    """
    # Get the directory containing server configuration files
    base_dir = Path(__file__).parent
    servers_dir = base_dir / "servers"
    
    if not servers_dir.exists():
        logger.warning(f"Servers directory not found: {servers_dir}. Creating it.")
        servers_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in the servers directory
    config_files = list(servers_dir.glob("*.json"))
    
    if not config_files:
        logger.warning("No server configuration files found.")
        return []
    
    # Load and validate each configuration file
    server_configs = []
    server_names = set()
    workspace_dir = os.getcwd()  # Get the workspace directory for substitution
    
    for config_file in config_files:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            
            # Basic validation
            required_fields = ["server_name", "endpoint", "allowed_tools", "sensitive_tools"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field} in {config_file}")
            
            # Check for duplicate server names
            server_name = config["server_name"]
            if server_name in server_names:
                raise ValueError(f"Duplicate server name found: {server_name} in {config_file}")
            
            # Handle ${WORKSPACE_DIR} replacement in args
            if "args" in config["endpoint"]:
                config["endpoint"]["args"] = [
                    str(arg).replace("${WORKSPACE_DIR}", workspace_dir)
                    for arg in config["endpoint"]["args"]
                ]
            
            server_names.add(server_name)
            server_configs.append(config)
            logger.info(f"Loaded configuration for server: {server_name} from {config_file.name}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON in {config_file}: {e}")
            raise ValueError(f"Invalid JSON in {config_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            raise
    
    logger.info(f"Loaded {len(server_configs)} server configurations")
    return server_configs

def create_server_map(server_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create a map of server names to endpoint configurations for MultiServerMCPClient.
    
    Args:
        server_configs: List of server configurations
    
    Returns:
        Dictionary mapping server names to endpoint configurations
    """
    return {
        config["server_name"]: config["endpoint"]
        for config in server_configs
    }

def get_allowed_tools_map(server_configs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Create a map of server names to their allowed tools.
    
    Args:
        server_configs: List of server configurations
    
    Returns:
        Dictionary mapping server names to lists of allowed tools
    """
    return {
        config["server_name"]: config["allowed_tools"]
        for config in server_configs
    }

def get_sensitive_tools_map(server_configs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Create a map of server names to their sensitive tools.
    
    Args:
        server_configs: List of server configurations
    
    Returns:
        Dictionary mapping server names to lists of sensitive tools
    """
    return {
        config["server_name"]: config["sensitive_tools"]
        for config in server_configs
    }

def namespace_tool_name(server_name: str, tool_name: str) -> str:
    """
    Create a namespaced tool name using underscore separator.
    
    Args:
        server_name: Name of the server
        tool_name: Original tool name
    
    Returns:
        Namespaced tool name with underscore separator
    """
    return f"{server_name}_{tool_name}"

def split_namespaced_tool(tool_name: str) -> tuple[str, str]:
    """
    Split a namespaced tool name into server and tool parts.
    
    Args:
        tool_name: Namespaced tool name (e.g., "filesystem_read_file")
    
    Returns:
        Tuple of (server_name, tool_name)
    
    Raises:
        ValueError: If the tool name is not properly namespaced
    """
    # Split on the first underscore
    parts = tool_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid namespaced tool name: {tool_name}")
    
    return parts[0], parts[1]