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
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=256,
#     temperature=0.7
# )

# model = ChatHuggingFace(llm=llm)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyDGA7dD-WxuwjsXou-RlMc5YN54p5NJK6s"
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool 
def calculator(first_num:float , second_num:float, operation: str)->dict:
    """
     Perform a basic arithmetic operation on two numbers.
     Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num + second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation."}
        
        return {"first_num":first_num, "second_num":second_num, "operation":operation, "result":result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_Stock_price(symbol: str)->dict:
    """
     Fetch latest stock price for a given symbol (e.g 'AAPL', 'TSLA')
     using AlphaVantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=CKH7TKNKTXBRY6HM9"
    r = requests.get(url)
    return r.json()

tools = [get_Stock_price, search_tool, calculator]
llm_with_tools = model.bind_tools(tools)

def chat_node(state: ChatState):
    """LLM node that may answer or call a certain tool"""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db', check_same_thread = False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def get_all_threads():
    """
    Return all thread_ids ordered by most recently updated (latest first).
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT thread_id
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY MAX(rowid) DESC
        """
    )
    rows = cursor.fetchall()
    return [r[0] for r in rows]

def delete_thread(thread_id: str) -> None:
    """Permanently delete a conversation (all checkpoints and writes) by thread_id."""
    with conn:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))