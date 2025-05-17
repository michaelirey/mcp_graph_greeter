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


# Define state type
class GreeterState(TypedDict, total=False):
    """State type for the MCP graph greeter."""

    messages: Annotated[List[BaseMessage], add_messages]
    greeting: str
    file_info: str
    name: str


async def greeter(state: GreeterState, config: RunnableConfig) -> Dict:
    """
    Generate a personalized greeting and ask about files.

    Args:
        state: Current graph state
        config: Configuration for the model

    Returns:
        Updated state with greeting
    """
    # Extract input from the last human message
    messages = state["messages"]
    name = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            # Extract name from input (assuming format "Hello, my name is [name]")
            content = message.content
            if "my name is" in content.lower():
                name = content.split("my name is", 1)[1].strip()
            break

    # Create a greeting
    if name:
        greeting = f"""Hello, {name}! Nice to meet you!

I'm a filesystem assistant that can help you explore your files and directories.
Try asking me something like:
- "What files are in the current directory?"
- "Show me the contents of a specific file"
- "Create a new file for me"

What would you like to know about your filesystem?"""
    else:
        greeting = """Hello there! I'm a filesystem assistant that can help you explore your files and directories.

Try asking me something like:
- "What files are in the current directory?"
- "Show me the contents of a specific file"
- "Create a new file for me"

What would you like to know about your filesystem?"""

    # Create response message
    response = AIMessage(content=greeting)

    # Return updated state
    logger.info(f"Generated greeting: {greeting}")
    return {
        "messages": [response],
        "greeting": greeting,
        "name": name if name else "User",
    }


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

    # Async version of agent
    async def async_agent(state: GreeterState, config: RunnableConfig) -> Dict:
        """Async version of the agent function."""
        messages = state["messages"]

        # Add system message if not already present
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = await model_with_tools.ainvoke(messages, config)
        return {"messages": [response]}

    # Create the graph
    workflow = StateGraph(GreeterState)

    # Add nodes
    workflow.add_node("greeter", greeter)
    workflow.add_node("agent", RunnableCallable(agent, async_agent))
    workflow.add_node("respond", respond)

    # Add tools node
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "greeter")
    workflow.add_edge("greeter", "agent")

    # Add conditional edges for the agent
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "respond": "respond"}
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Add edge from respond to end
    workflow.add_edge("respond", END)

    # Compile the graph
    return workflow.compile()


@asynccontextmanager
async def mcp_graph_greeter() -> AsyncIterator:
    """
    Create a filesystem graph greeter with async context management.

    This function:
    1. Initializes the MCP client and connects to the MCP server using session
    2. Creates a LangGraph workflow with the model and filesystem tools
    3. Yields the graph for use
    4. Ensures the MCP session is properly closed

    Follows the recommended pattern from langchain-mcp-adapters 0.1.0.

    Yields:
        A configured LangGraph for filesystem operations
    """
    # Create MCP client (not using it as a context manager)
    client = MultiServerMCPClient(FILESYSTEM_SERVER)

    # Use the client's session method for the filesystem server
    async with client.session("filesystem") as session:
        # Get tools using the load_mcp_tools function
        try:
            filesystem_tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(filesystem_tools)} filesystem tools")

            # Create the graph with the tools
            graph = build_greeter_graph(filesystem_tools)
            logger.info("MCP Graph Greeter created successfully")

            # Yield the graph - the session will remain active during this context
            yield graph
        except Exception as e:
            logger.error(f"Error creating MCP Graph Greeter: {str(e)}")
            raise
        finally:
            # Clean up is handled by async context manager of the session
            logger.info("MCP Graph Greeter closed")


async def invoke_greeter(
    message: str,
    context_messages: Optional[List[BaseMessage]] = None,
) -> List[BaseMessage]:
    """
    Invoke the MCP Graph Greeter with a message.

    Args:
        message: User's message
        context_messages: Optional list of previous messages for context

    Returns:
        The complete conversation history
    """
    try:
        # Create the graph with the async context manager
        async with mcp_graph_greeter() as graph:
            # Prepare initial messages
            messages = []

            # Add context messages if provided
            if context_messages:
                messages.extend(context_messages)

            # Add the human message
            messages.append(HumanMessage(content=message))

            # Create initial state
            initial_state = {
                "messages": messages,
                "greeting": "",
                "file_info": "",
                "name": "",
            }

            # Run the graph asynchronously
            result = await graph.ainvoke(initial_state)
            logger.info("Graph execution completed successfully")
            return result["messages"]

    except Exception as e:
        # Log the error
        logger.error(f"Error running MCP Graph Greeter: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

        # Return a simple message with the error
        return [HumanMessage(content=message), AIMessage(content=f"Error: {str(e)}")]


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

    # Create MCP client (not using it as a context manager)
    client = MultiServerMCPClient(FILESYSTEM_SERVER)

    try:
        # Use the client's session method for the filesystem server
        async with client.session("filesystem") as session:
            # Get tools using the load_mcp_tools function
            filesystem_tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(filesystem_tools)} filesystem tools")

            # Create the graph with the tools
            graph = build_greeter_graph(filesystem_tools)
            logger.info("MCP Graph Greeter created successfully")

            # Yield the graph - the session will remain active during this context
            yield graph
    except Exception as e:
        logger.error(f"Error creating MCP Graph Greeter: {str(e)}")
        raise
    finally:
        logger.info("MCP Graph Greeter factory closing")
