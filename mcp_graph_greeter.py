"""
MCP Graph Greeter - A LangGraph demo using Model Context Protocol for filesystem operations

This module demonstrates integrating filesystem MCP tools into a LangGraph workflow
with proper resource management and async execution patterns.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Dict, Optional
import logging
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.constants import END, START
from langgraph.utils.runnable import RunnableCallable
from langgraph.prebuilt.tool_node import ToolNode

# Import MCP server configuration
from config import FILESYSTEM_SERVER, OPENAI_API_KEY, LLM_MODEL_NAME

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


async def greeter(state: GreeterState, config: RunnableConfig) -> Dict:
    """Generate a personalized greeting and ask about files."""

    messages = state["messages"]
    return {"messages": messages }


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


def build_greeter_graph(tools: List[BaseTool]) -> StateGraph:
    """
    Build a greeter graph with the provided filesystem tools.

    Args:
        tools: List of filesystem tools to use

    Returns:
        A compiled graph ready to use
    """
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
        
        # Only request human approval for specific sensitive tools
        sensitive_tools = ["read_file", "get_file_info"]
        
        # If the tool is not in our sensitive list, skip review
        if tool_name not in sensitive_tools:
            logger.info(f"Tool '{tool_name}' not sensitive - skipping review")
            return Command(goto="tools")
            
        # Interrupt and wait for human review of the tool call
        # This will be displayed in the UI for user approval
        human_review = interrupt({
            "question": f"Review required for sensitive tool '{tool_name}'. Approve? (Enter 'true' to approve or 'false' to cancel)",
            "tool_call": tool_call
        })
        
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
                "tool_call_id": tool_call["id"]
            }
            return Command(goto="agent", update={"messages": [feedback]})
        
        # Also handle dictionary responses for modification
        elif isinstance(human_review, dict):
            # If args are provided, update the tool call
            if "args" in human_review:
                updated_message = AIMessage(
                    content=last_message.content,
                    tool_calls=[{
                        "id": tool_call["id"],
                        "name": tool_name,
                        "args": human_review["args"],
                        "type": "tool_call"
                    }],
                    id=last_message.id
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
                        "tool_call_id": tool_call["id"]
                    }
                    return Command(goto="agent", update={"messages": [feedback]})
        
        # Default: continue with the tool call
        return Command(goto="tools")

    # Create the graph with input and output types
    workflow = StateGraph(GreeterState, input=GreeterInput, output=GreeterOutput)

    # Add nodes
    workflow.add_node("greeter", greeter)
    workflow.add_node("agent", agent)
    workflow.add_node("respond", respond)
    workflow.add_node("review_tool_calls", review_tool_calls)

    # Add tools node
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "greeter")
    workflow.add_edge("greeter", "agent")

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

    # Define which tools to include (comment this line to include all tools)
    allowed_tools = ["list_directory", "read_file", "list_allowed_directories", "get_file_info", "search_files", "directory_tree"]

    client = MultiServerMCPClient(FILESYSTEM_SERVER)

    try:
        # Use the client's session method for the filesystem server
        async with client.session("filesystem") as session:
            # Get all tools using the load_mcp_tools function
            all_tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(all_tools)} filesystem tools")
            
            # Filter tools by name if allowed_tools is defined
            filesystem_tools = [t for t in all_tools if not allowed_tools or t.name in allowed_tools]
            logger.info(f"Using {len(filesystem_tools)}/{len(all_tools)} tools")

            # Create the graph with the filtered tools
            graph = build_greeter_graph(filesystem_tools)
            logger.info("MCP Graph Greeter created successfully")

            # Yield the graph - the session will remain active during this context
            yield graph
    except Exception as e:
        logger.error(f"Error creating MCP Graph Greeter: {str(e)}")
        raise
    finally:
        logger.info("MCP Graph Greeter factory closing")
