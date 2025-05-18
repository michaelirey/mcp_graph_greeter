"""
Tests for the configuration loader module
"""

import os
import json
import tempfile
import pytest
from pathlib import Path

from config.loader import (
    load_server_configs,
    create_server_map,
    get_allowed_tools_map,
    get_sensitive_tools_map,
    split_namespaced_tool,
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create servers directory
        servers_dir = Path(temp_dir) / "servers"
        servers_dir.mkdir()
        
        # Create a sample server configuration
        fs_config = {
            "server_name": "test_filesystem",
            "endpoint": {
                "command": "test_cmd",
                "args": ["arg1", "arg2"],
                "transport": "stdio"
            },
            "allowed_tools": ["read_file", "write_file"],
            "sensitive_tools": ["write_file"]
        }
        
        # Write to a JSON file
        with open(servers_dir / "filesystem.json", "w") as f:
            json.dump(fs_config, f)
        
        # Create another server config
        context_config = {
            "server_name": "test_context",
            "endpoint": {
                "command": "context_cmd",
                "transport": "stdio"
            },
            "allowed_tools": ["get_docs"],
            "sensitive_tools": []
        }
        
        # Write to a JSON file
        with open(servers_dir / "context.json", "w") as f:
            json.dump(context_config, f)
        
        yield temp_dir


def test_load_server_configs(temp_config_dir, monkeypatch):
    """Test loading server configurations from JSON files."""
    # Mock the base directory to use our temporary test directory
    monkeypatch.setattr("config.loader.Path.__new__", lambda cls, *args, **kwargs: Path(temp_config_dir))
    
    # Load configurations
    configs = load_server_configs()
    
    # Check if we loaded two configurations
    assert len(configs) == 2
    
    # Check first configuration
    fs_config = next(c for c in configs if c["server_name"] == "test_filesystem")
    assert fs_config["endpoint"]["command"] == "test_cmd"
    assert "read_file" in fs_config["allowed_tools"]
    assert "write_file" in fs_config["sensitive_tools"]
    
    # Check second configuration
    context_config = next(c for c in configs if c["server_name"] == "test_context")
    assert context_config["endpoint"]["command"] == "context_cmd"
    assert "get_docs" in context_config["allowed_tools"]
    assert len(context_config["sensitive_tools"]) == 0


def test_create_server_map():
    """Test creating a server map from configurations."""
    configs = [
        {
            "server_name": "server1",
            "endpoint": {"command": "cmd1", "transport": "stdio"},
            "allowed_tools": [],
            "sensitive_tools": []
        },
        {
            "server_name": "server2",
            "endpoint": {"command": "cmd2", "transport": "stdio"},
            "allowed_tools": [],
            "sensitive_tools": []
        }
    ]
    
    server_map = create_server_map(configs)
    
    assert len(server_map) == 2
    assert "server1" in server_map
    assert "server2" in server_map
    assert server_map["server1"]["command"] == "cmd1"
    assert server_map["server2"]["command"] == "cmd2"


def test_get_allowed_tools_map():
    """Test getting allowed tools map from configurations."""
    configs = [
        {
            "server_name": "server1",
            "endpoint": {},
            "allowed_tools": ["tool1", "tool2"],
            "sensitive_tools": []
        },
        {
            "server_name": "server2",
            "endpoint": {},
            "allowed_tools": ["tool3"],
            "sensitive_tools": []
        }
    ]
    
    allowed_map = get_allowed_tools_map(configs)
    
    assert len(allowed_map) == 2
    assert "server1" in allowed_map
    assert "server2" in allowed_map
    assert set(allowed_map["server1"]) == {"tool1", "tool2"}
    assert set(allowed_map["server2"]) == {"tool3"}


def test_get_sensitive_tools_map():
    """Test getting sensitive tools map from configurations."""
    configs = [
        {
            "server_name": "server1",
            "endpoint": {},
            "allowed_tools": [],
            "sensitive_tools": ["tool1"]
        },
        {
            "server_name": "server2",
            "endpoint": {},
            "allowed_tools": [],
            "sensitive_tools": ["tool2", "tool3"]
        }
    ]
    
    sensitive_map = get_sensitive_tools_map(configs)
    
    assert len(sensitive_map) == 2
    assert "server1" in sensitive_map
    assert "server2" in sensitive_map
    assert set(sensitive_map["server1"]) == {"tool1"}
    assert set(sensitive_map["server2"]) == {"tool2", "tool3"}


def test_split_namespaced_tool():
    """Test splitting a namespaced tool name."""
    server, tool = split_namespaced_tool("server1.tool1")
    assert server == "server1"
    assert tool == "tool1"
    
    # Test with multiple periods in the tool name
    server, tool = split_namespaced_tool("server1.tool.with.dots")
    assert server == "server1"
    assert tool == "tool.with.dots"
    
    # Test error case
    with pytest.raises(ValueError):
        split_namespaced_tool("invalid_name_without_namespace")