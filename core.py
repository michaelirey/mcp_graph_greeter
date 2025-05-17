"""Core functionality for building the greeter graph."""

import logging
from typing import List, Dict
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool

try:
    from .mcp_graph_greeter import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover - fallback for direct import
    from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.constants import END, START
from langgraph.utils.runnable import RunnableCallable
from langgraph.prebuilt.tool_node import ToolNode

from config import OPENAI_API_KEY, LLM_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GreeterState(TypedDict, total=False):
    """State type for the MCP graph greeter."""

    messages: Annotated[List[BaseMessage], add_messages]


async def greeter(state: GreeterState, config: RunnableConfig) -> Dict:
    """Generate a personalized greeting and ask about files."""

    messages = state["messages"]
    name = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if "my name is" in content.lower():
                name = content.split("my name is", 1)[1].strip()
            break

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
    response = AIMessage(content=greeting)

    logger.info(f"Generated greeting: {greeting}")
    return {"messages": messages + [response]}


def should_continue(state: GreeterState) -> str:
    """Determine if we should continue to tools or end."""

    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and getattr(
        last_message, "tool_calls", None
    ):
        return "tools"
    return "respond"


async def respond(state: GreeterState, config: RunnableConfig) -> Dict:
    """Final response node, just passes through messages."""

    logger.info("Final response node reached")
    return {}


class GreeterGraphBuilder:
    """Build the LangGraph workflow for the greeter."""

    def __init__(self, tools: List[BaseTool]):
        self.tools = tools

    def build(self) -> StateGraph:
        model = ChatOpenAI(model=LLM_MODEL_NAME, api_key=OPENAI_API_KEY or None)
        model_with_tools = model.bind_tools(self.tools)

        system_prompt = """You are a helpful filesystem assistant that can help users navigate and manage their files.
    You can provide information about files, read file contents, create new files, and more.

    When asked about directories, provide a clear listing of files and subdirectories.
    When asked about file contents, provide the contents in a nicely formatted way.
    When creating files, confirm the creation and provide the file path.

    Always be friendly, helpful, and focused on filesystem operations.
    """

        def agent(state: GreeterState, config: RunnableConfig) -> Dict:
            messages = state["messages"]
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages
            response = model_with_tools.invoke(messages, config)
            return {"messages": [response]}

        async def async_agent(state: GreeterState, config: RunnableConfig) -> Dict:
            messages = state["messages"]
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages
            response = await model_with_tools.ainvoke(messages, config)
            return {"messages": [response]}

        workflow = StateGraph(GreeterState)
        workflow.add_node("greeter", greeter)
        workflow.add_node("agent", RunnableCallable(agent, async_agent))
        workflow.add_node("respond", respond)

        tool_node = ToolNode(self.tools)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "greeter")
        workflow.add_edge("greeter", "agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "respond": "respond"}
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)

        return workflow.compile()


def build_greeter_graph(tools: List[BaseTool]) -> StateGraph:
    """Convenience wrapper to build the greeter graph."""

    return GreeterGraphBuilder(tools).build()
