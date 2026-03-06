from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.langchain import wait_for_all_tracers

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)

# Ensure traces are sent to LangSmith before script exits
wait_for_all_tracers()
