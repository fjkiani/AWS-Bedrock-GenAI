import os
from dotenv import load_dotenv
import requests
from langchain.chat_models import ChatOpenAI
import sys
print(sys.executable)

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from io import StringIO

memory = MemorySaver()

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

class AgentState(TypedDict):
    task: str
    student_data: str
    learning_materials: List[str]
    analysis: str
    personalized_path: str
    feedback: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Define prompts for each node
ANALYZE_STUDENT_DATA_PROMPT = """You are an educational expert. Analyze the provided student performance data and identify strengths and weaknesses. Provide detailed analysis."""
RETRIEVE_LEARNING_MATERIALS_PROMPT = """You are an educational content researcher. Based on the student’s needs, retrieve relevant learning materials from the database."""
BUILD_LEARNING_PATH_PROMPT = """You are an adaptive learning specialist. Create a personalized learning path for the student based on the analysis and the available content."""
FEEDBACK_PROMPT = """You are a teacher providing feedback. Review the generated learning path and suggest improvements or additions based on educational principles."""
RESEARCH_ADDITIONAL_CONTENT_PROMPT = """You are tasked with finding additional content to address gaps identified in the feedback. Generate queries to gather relevant educational resources."""

def analyze_student_data_node(state: AgentState):
    student_data_str = state["student_data"]
    messages = [
        SystemMessage(content=ANALYZE_STUDENT_DATA_PROMPT),
        HumanMessage(content=student_data_str),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}

def retrieve_learning_materials_node(state: AgentState):
    messages = [
        SystemMessage(content=RETRIEVE_LEARNING_MATERIALS_PROMPT),
        HumanMessage(content=state["analysis"]),
    ]
    response = model.invoke(messages)
    materials = response.content.splitlines()
    return {"learning_materials": materials}

def build_learning_path_node(state: AgentState):
    learning_materials_str = "\n".join(state["learning_materials"])
    messages = [
        SystemMessage(content=BUILD_LEARNING_PATH_PROMPT),
        HumanMessage(content=f"{state['analysis']}\n\n{learning_materials_str}"),
    ]
    response = model.invoke(messages)
    return {"personalized_path": response.content}

def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["personalized_path"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}

import random

def research_additional_content_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_ADDITIONAL_CONTENT_PROMPT),
        HumanMessage(content=state["feedback"]),
    ]
    response = model.invoke(messages)
    queries = response.content.splitlines()
    content = state.get("content", [])

    for q in queries:
        q = q.strip()
        if not q:
            print("Skipping empty query")
            continue
        
        # Mocking the response instead of calling an external API
        mock_results = [
            f"Mock educational content for query: '{q}' - Part {i}"
            for i in range(1, 3)
        ]
        content.extend(mock_results)

    return {"content": content}


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"

# Define the workflow using LangGraph
builder = StateGraph(AgentState)

builder.add_node("analyze_student_data", analyze_student_data_node)
builder.add_node("retrieve_learning_materials", retrieve_learning_materials_node)
builder.add_node("build_learning_path", build_learning_path_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_additional_content", research_additional_content_node)

builder.set_entry_point("analyze_student_data")

builder.add_conditional_edges(
    "build_learning_path",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

builder.add_edge("analyze_student_data", "retrieve_learning_materials")
builder.add_edge("retrieve_learning_materials", "build_learning_path")
builder.add_edge("collect_feedback", "research_additional_content")
builder.add_edge("research_additional_content", "build_learning_path")

# Compile the graph
graph = builder.compile()

# ==== For Console Testing ====
import streamlit as st

def main():
    st.title("Edulga's Adaptive Learning Agent")

    task = st.text_input(
        "Enter the task:",
        "Generate a personalized learning path based on student data",
    )
    student_data = st.text_area("Enter student performance data:")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2)

    if st.button("Start Learning Path Generation") and student_data:
        initial_state = {
            "task": task,
            "student_data": student_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
        }
        thread = {"configurable": {"thread_id": "1"}}

        final_state = None
        for s in graph.stream(initial_state, thread):
            st.write(s)
            final_state = s

        if final_state and "personalized_path" in final_state:
            st.subheader("Personalized Learning Path")
            st.write(final_state["personalized_path"])

if __name__ == "__main__":
    main()
