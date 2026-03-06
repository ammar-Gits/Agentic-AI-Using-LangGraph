from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7
)

chat_model = ChatHuggingFace(llm=llm)

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: float = Field(description="Score out of 10 for the essay", ge=0, le=10)


parser = PydanticOutputParser(pydantic_object=EvaluationSchema)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert essay evaluator.\n"
     "Return ONLY valid JSON that matches the given schema.\n"
     "Do not include explanations or extra text.\n"
     "{format_instructions}"
    ),
    ("human", "Evaluate the following essay:\n\n{essay}")
])

chain = prompt | chat_model | parser

class EssayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

def evaluate_language(state: EssayState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an expert essay evaluator.\n"
        "Return ONLY valid JSON that matches the given schema.\n"
        "Do not include explanations or extra text.\n"
        "Focus ONLY on language: grammar, vocabulary, sentence structure, and word choice.\n"
        "The JSON must have two keys: 'feedback' (string) and 'score' (number from 0 to 10).\n"
        "{format_instructions}"
        ),
        ("human",
        "Evaluate the language quality of the following essay. "
        "Provide feedback specifically about grammar, vocabulary, and sentence structure, "
        "and assign a score out of 10:\n\n{essay}"
        )
    ])
    chain = prompt | chat_model | parser
    result = chain.invoke({
        "essay": state["essay"],
        "format_instructions": parser.get_format_instructions(),
    })

    return {'language_feedback': result.feedback, 'individual_scores':[result.score]}

def evaluate_analysis(state: EssayState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an expert essay evaluator.\n"
        "Return ONLY valid JSON that matches the given schema.\n"
        "Do not include explanations or extra text.\n"
        "Focus ONLY on language: grammar, vocabulary, sentence structure, and word choice.\n"
        "The JSON must have two keys: 'feedback' (string) and 'score' (number from 0 to 10).\n"
        "{format_instructions}"
        ),
        ("human",
        "Evaluate the depth of analysis of the following essay. "
        "Provide feedback specifically about logic, reasoning, use of evidence, and critical thinking, "
        "and assign a score out of 10:\n\n{essay}"
        )
    ])
    chain = prompt | chat_model | parser
    result = chain.invoke({
        "essay": state["essay"],
        "format_instructions": parser.get_format_instructions(),
    })

    return {'analysis_feedback': result.feedback, 'individual_scores':[result.score]}

def evaluate_thought(state: EssayState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an expert essay evaluator.\n"
        "Return ONLY valid JSON that matches the given schema.\n"
        "Do not include explanations or extra text.\n"
        "Focus ONLY on language: grammar, vocabulary, sentence structure, and word choice.\n"
        "The JSON must have two keys: 'feedback' (string) and 'score' (number from 0 to 10).\n"
        "{format_instructions}"
        ),
        ("human",
        "Evaluate the clarity of thought of the following essay. "
        "Provide feedback specifically about coherence, logical flow, organization, and clarity of ideas, "
        "and assign a score out of 10:\n\n{essay}"
        )
    ])
    chain = prompt | chat_model | parser
    result = chain.invoke({
        "essay": state["essay"],
        "format_instructions": parser.get_format_instructions(),
    })

    return {'clarity_feedback': result.feedback, 'individual_scores':[result.score]}

def final_evaluation(state: EssayState):
    prompt = f"Based on following feedbacks create a summarized feedback \n language feedback - {state['language_feedback']} \n depth of analysis feedback - {state['analysis_feedback']} \n clarity of thought feedback {state['clarity_feedback']}"
    overall_feedback = chat_model.invoke(prompt)

    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

graph = StateGraph(EssayState)
 
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('final_evaluation', final_evaluation)
graph.add_edge(START,'evaluate_language')
graph.add_edge(START,'evaluate_thought')
graph.add_edge(START,'evaluate_analysis')
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('final_evaluation',END)

workflow = graph.compile()

out = chat_model.invoke("write an essay on topic AI in pakistan with bad language structure, very good clarity of thought but poor analysis")
essay = out.content

initial_state = {
    'essay': essay
}
workflow.invoke(initial_state)