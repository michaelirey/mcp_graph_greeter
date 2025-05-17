#!/usr/bin/env python3
"""
Demo script for MCP Graph Greeter

This script demonstrates the MCP Graph Greeter in an interactive session.
"""
import asyncio
import logging
import sys
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage

from greeter_service import invoke_greeter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mcp_graph_greeter_demo")


async def interactive_session():
    """Run an interactive session with the MCP Graph Greeter."""
    print("\n📁 Welcome to the MCP Graph Greeter Demo 📁\n")
    print(
        "This demo allows you to interact with a filesystem assistant powered by LangGraph and MCP."
    )
    print(
        "You can ask about files in your current directory, create new files, and more."
    )

    # Start with an introduction
    messages: List[BaseMessage] = []

    try:
        # First message - introduce yourself - required
        while True:
            user_input = input("\n🧑 Please introduce yourself (e.g., 'Hello, my name is Alice'): ")
            if user_input:
                break
            print("⚠️  Introduction is required. Please try again.")

        # Get greeting response
        print("\n🤖 Contacting the assistant...")
        messages = await invoke_greeter(user_input)

        # Print the response
        for message in messages:
            if isinstance(message, HumanMessage):
                print(f"\n🧑 {message.content}")
            else:
                print(f"\n🤖 Assistant: {message.content}")

        # Continue the conversation
        while True:
            # Get user input - required
            while True:
                user_input = input("\n🧑 Your input: ")
                if user_input:
                    break
                print("⚠️  Input is required. Please try again.")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Thanks for using the MCP Graph Greeter Demo! Goodbye!")
                break

            # Get response
            print("\n🤖 Thinking...")
            new_messages = await invoke_greeter(user_input, context_messages=messages)

            # Store all messages for context
            messages = new_messages

            # Print just the latest response
            latest_response = messages[-1]
            print(f"\n🤖 Assistant: {latest_response.content}")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Error in interactive session: {str(e)}")
        print(f"\n❌ An error occurred: {str(e)}")


if __name__ == "__main__":
    # Run the interactive session
    asyncio.run(interactive_session())
