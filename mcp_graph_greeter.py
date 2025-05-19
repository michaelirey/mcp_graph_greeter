"""
MCP Graph Greeter - A LangGraph demo using Model Context Protocol for filesystem operations

This module demonstrates integrating filesystem MCP tools into a LangGraph workflow
with proper resource management and async execution patterns.
"""

from contextlib import asynccontextmanager
from typing import List, Dict
import logging
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.constants import END, START
from langgraph.prebuilt.tool_node import ToolNode

# Import MCP server configuration
from config import (
    MCP_SERVERS,  # backward compatibility
    OPENAI_API_KEY,
    LLM_MODEL_NAME,
    load_server_configs,
    ServerConfig,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define input and state types
class GreeterInput(TypedDict):
    """Input type for the MCP graph greeter."""

    messages: List[BaseMessage]  # Required input message(s)


class GreeterOutput(TypedDict):
    """Output type for the MCP graph greeter."""

    messages: List[BaseMessage]  # Output messages


class GreeterState(TypedDict, total=False):
    """State type for the MCP graph greeter."""

    messages: Annotated[List[BaseMessage], add_messages]


def split_namespaced(name: str) -> tuple[str, str]:
    """Split a namespaced tool name into (server, tool)."""
    if "." not in name:
        raise ValueError(f"Tool name '{name}' is not namespaced")
    return name.split(".", 1)




def should_continue(state: GreeterState) -> str:
    """
    Determine if we should continue to tools or end.

    Args:
        state: Current state of the graph

    Returns:
        Next node to execute: "tools" or "respond"
    """
    # Check if the last message has tool calls
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, continue to tools
    if isinstance(last_message, AIMessage) and getattr(
        last_message, "tool_calls", None
    ):
        return "tools"
    # Otherwise proceed to respond (no more processing needed)
    return "respond"


async def respond(state: GreeterState, config: RunnableConfig) -> Dict:
    """
    Final response node, just passes through messages.

    Args:
        state: Current graph state
        config: Configuration for the model

    Returns:
        Unchanged state
    """
    # This node doesn't modify the state, just a passthrough
    logger.info("Final response node reached")
    return {}


def build_greeter_graph(
    tools: List[BaseTool], sensitive_tools: List[str] = None
) -> StateGraph:
    """
    Build a greeter graph with the provided filesystem tools.

    Args:
        tools: List of filesystem tools to use
        sensitive_tools: List of tool names that require human approval before execution

    Returns:
        A compiled graph ready to use
    """
    # Default to empty list if None
    if sensitive_tools is None:
        sensitive_tools = []
    # Import required modules for human-in-the-loop
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command

    # Initialize the model with tools bound using config values
    model = ChatOpenAI(
        model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY if OPENAI_API_KEY else None
    )
    model_with_tools = model.bind_tools(tools)

    # Define the system prompt
    system_prompt = """You are a helpful filesystem assistant that can help users navigate and manage their files.
    You can provide information about files, read file contents, create new files, and more.

    When asked about directories, provide a clear listing of files and subdirectories.
    When asked about file contents, provide the contents in a nicely formatted way.
    When creating files, confirm the creation and provide the file path.

    Always be friendly, helpful, and focused on filesystem operations.
    """

    # Define the agent function
    def agent(state: GreeterState, config: RunnableConfig) -> Dict:
        """Call the model with tools to respond to filesystem queries."""
        messages = state["messages"]

        # Add system message if not already present
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = model_with_tools.invoke(messages, config)
        return {"messages": [response]}

    # Define a human review node for tool calls
    def review_tool_calls(state: GreeterState) -> Command:
        """Review tool calls before executing them."""
        last_message = state["messages"][-1]

        # If there are no tool calls, move on
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return Command(goto="tools")

        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        try:
            server, base_tool = split_namespaced(tool_name)
        except ValueError:
            server, base_tool = None, tool_name

        # Only request human approval for specific sensitive tools defined at the graph_factory level
        # If the tool is not in our sensitive list, skip review
        if tool_name not in sensitive_tools:
            logger.info(f"Tool '{tool_name}' not sensitive - skipping review")
            return Command(goto="tools")

        # Interrupt and wait for human review of the tool call
        # This will be displayed in the UI for user approval
        human_review = interrupt(
            {
                "question": f"Review required for sensitive tool '{tool_name}'. Approve? (Enter 'true' to approve or 'false' to cancel)",
                "tool_call": tool_call,
            }
        )

        # Handle boolean responses (true/false)
        if human_review is True:
            # Continue with the original tool call
            return Command(goto="tools")
        elif human_review is False:
            # Cancel the tool call and go back to the agent
            feedback = {
                "role": "tool",
                "content": "Tool call was canceled by the user.",
                "name": tool_name,
                "tool_call_id": tool_call["id"],
            }
            return Command(goto="agent", update={"messages": [feedback]})

        # Also handle dictionary responses for modification
        elif isinstance(human_review, dict):
            # If args are provided, update the tool call
            if "args" in human_review:
                updated_message = AIMessage(
                    content=last_message.content,
                    tool_calls=[
                        {
                            "id": tool_call["id"],
                            "name": tool_name,
                            "args": human_review["args"],
                            "type": "tool_call",
                        }
                    ],
                    id=last_message.id,
                )
                return Command(goto="tools", update={"messages": [updated_message]})
            # If action is provided
            elif "action" in human_review:
                if human_review["action"] == "approve":
                    return Command(goto="tools")
                else:
                    feedback = {
                        "role": "tool",
                        "content": "Tool call was canceled by the user.",
                        "name": tool_name,
                        "tool_call_id": tool_call["id"],
                    }
                    return Command(goto="agent", update={"messages": [feedback]})

        # Default: continue with the tool call
        return Command(goto="tools")

    # Create the graph with input and output types
    workflow = StateGraph(GreeterState, input=GreeterInput, output=GreeterOutput)

    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("respond", respond)
    workflow.add_node("review_tool_calls", review_tool_calls)

    # Add tools node
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    # Add edges - start directly with agent instead of greeter
    workflow.add_edge(START, "agent")

    # Add conditional edges for the agent
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "review_tool_calls", "respond": "respond"}
    )

    # Add edge from review to tools and from tools back to agent
    workflow.add_edge("review_tool_calls", "tools")
    workflow.add_edge("tools", "agent")

    # Add edge from respond to end
    workflow.add_edge("respond", END)

    # Compile the graph with memory to support interrupts
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


@asynccontextmanager
async def graph_factory():
    """
    Graph factory function for LangGraph CLI.

    This function is the entry point configured in langgraph.json.
    It creates and yields the MCP Graph Greeter for the CLI to use.

    Yields:
        A configured LangGraph for filesystem operations
    """
    logger.info("Initializing MCP Graph Greeter for LangGraph CLI")

    configs = load_server_configs()
    server_map = {c.server_name: c.endpoint for c in configs}
    allowed_map = {c.server_name: set(c.allowed_tools) for c in configs}
    sensitive_map = {c.server_name: set(c.sensitive_tools) for c in configs}

    client = MultiServerMCPClient(server_map)

    try:
        tools = await client.get_tools()
        logger.info(f"Loaded {len(tools)} total tools")

        namespaced_tools: List[BaseTool] = []
        seen = set()

        for tool in tools:
            meta = getattr(tool, "metadata", {}) or {}
            server = meta.get("server") or meta.get("server_name")
            if not server:
                # Fall back to config lookup if metadata missing
                for srv, allowed in allowed_map.items():
                    if tool.name in allowed:
                        server = srv
                        break
            if not server:
                raise ValueError(f"Tool {tool.name} missing server metadata")

            if tool.name not in allowed_map.get(server, set()):
                continue

            namespaced = f"{server}.{tool.name}"
            if namespaced in seen:
                raise ValueError(f"Duplicate tool after namespacing: {namespaced}")
            seen.add(namespaced)

            tool.metadata["original_name"] = tool.name
            tool.name = namespaced
            namespaced_tools.append(tool)

        sensitive_names = [f"{srv}.{name}" for srv, names in sensitive_map.items() for name in names]

        graph = build_greeter_graph(namespaced_tools, sensitive_names)
        logger.info(
            f"MCP Graph Greeter created with {len(namespaced_tools)} tools ({len(sensitive_names)} requiring approval)"
        )

        yield graph

    except Exception as e:
        logger.error(f"Error creating MCP Graph Greeter: {str(e)}")
        raise
    finally:
        logger.info("MCP Graph Greeter factory closing")
