from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
import operator
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages': [response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)
