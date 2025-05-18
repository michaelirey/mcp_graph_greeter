"""Wrappers and helpers for launching MCP servers and loading tools."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


@dataclass
class MCPServerWrapper:
    """Simple wrapper around a single MCP server."""

    name: str
    config: Dict
    allowed_tools: Optional[List[str]] = None

    async def get_tools(self) -> List[BaseTool]:
        """Return namespaced tools for this server."""
        async with MultiServerMCPClient({self.name: self.config}) as client:
            tools = await client.get_tools()
        if self.allowed_tools:
            tools = [t for t in tools if t.name in self.allowed_tools]
        for tool in tools:
            tool.name = f"{self.name}:{tool.name}"
        return tools


def build_wrappers(servers: Dict[str, Dict], allowed: Optional[Dict[str, List[str]]] = None) -> Dict[str, MCPServerWrapper]:
    """Create wrappers for a mapping of servers."""
    allowed = allowed or {}
    return {
        name: MCPServerWrapper(name, cfg, allowed.get(name)) for name, cfg in servers.items()
    }


async def load_all_tools(wrappers: Dict[str, MCPServerWrapper]) -> List[BaseTool]:
    """Load tools from all wrappers."""
    all_tools: List[BaseTool] = []
    for wrapper in wrappers.values():
        all_tools.extend(await wrapper.get_tools())
    return all_tools
