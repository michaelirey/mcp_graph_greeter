"""
Configuration package for MCP Graph Greeter
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application paths
BASE_DIR = Path(__file__).parent.parent
WORKSPACE_DIR = os.getcwd()  # Use current working directory as workspace

# API configuration
# Check for OpenAI API key in environment variables (empty by default, will use system config)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4.1-nano")

logger.info(f"Configured workspace directory: {WORKSPACE_DIR}")
logger.info(f"Using LLM model: {LLM_MODEL_NAME}")

# MCP server configurations - legacy format for backward compatibility
from .loader import load_server_configs

# Load server configurations
try:
    CONFIG_SERVERS = load_server_configs()
    
    # Create a backward-compatible MCP_SERVERS dict
    MCP_SERVERS = {}
    for config in CONFIG_SERVERS:
        MCP_SERVERS[config["server_name"]] = config["endpoint"]
    
    # Backward compatibility
    FILESYSTEM_SERVER = {"filesystem": MCP_SERVERS.get("filesystem", {})}
    
except Exception as e:
    logger.error(f"Error loading server configurations: {e}")
    # Fallback to empty configurations
    CONFIG_SERVERS = []
    MCP_SERVERS = {}
    FILESYSTEM_SERVER = {}