"""
Configuration module for MCP Graph Greeter
"""
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application paths
BASE_DIR = Path(__file__).parent
WORKSPACE_DIR = os.getcwd()  # Use current working directory as workspace

# API configuration
# Check for OpenAI API key in environment variables (empty by default, will use system config)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4.1")

# Filesystem MCP server configuration
FILESYSTEM_SERVER = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            WORKSPACE_DIR  # Only allow access to current working directory
        ],
        "transport": "stdio"  # Required by MultiServerMCPClient
    }
}

logger.info(f"Configured workspace directory: {WORKSPACE_DIR}")
logger.info(f"Using LLM model: {LLM_MODEL_NAME}")