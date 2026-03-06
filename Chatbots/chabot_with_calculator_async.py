from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
import operator
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3"
        }
    }
)
tools = [calculator]
llm_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def build_graph():
    async def chat_node(state: ChatState):
        """LLM node that may answer or call a certain tool"""
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools","chat_node")
    chatbot = graph.compile()

    return chatbot

async def main():
    chatbot = build_graph()
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="What is product of 2 and 4")]})
    print(result['messages'][-1].content())