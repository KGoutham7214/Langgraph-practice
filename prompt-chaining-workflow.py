"""
Task: Content Generation Pipeline with Quality Control

Input:
1. Topic
2. Quality Requirements

Steps: 
1. Generate a initial draft
2. fact check the draft
3.improve based on feedback
4. Format the publication
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
import os
import certifi

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

class ContentState(TypedDict):
    topic: str
    requirements: str
    draft: str
    fact_check_results: str
    improved_content: str
    final_draft: str

llm = ChatOpenAI(model="gpt-5.4-mini")

#Define Nodes
def generate_draft(state: ContentState) -> str:
    """Generates an initial draft based on the topic and requirements."""

    prompt = f"""
        Generate a 200 worddraft on the topic: {state['topic']} 
        with the following requirements: {state['requirements']}
        Focus on creating informative and engaging content."""
    
    draft = llm.invoke(prompt).content
    print(" Step 1: Draft Generated:\n", draft[:150] + "...")
    
    return{
        "draft": draft
    }

def fact_check(state: ContentState) -> str:
    """Check draft for factual accuracy and consistency."""
    prompt = f"""
        Check the following draft for factual accuracy and consistency:
        {state['draft']}
        Identify any factual errors, inconsistencies, or areas that require improvement and statements that need citations.
        """
    fact_check_results = llm.invoke(prompt).content
    print(" Step 2: Fact Check Results:\n", fact_check_results[:150] + "...")
    return{
        "fact_check_results": fact_check_results
    }

def improve_content(state: ContentState) -> str:
    """Improve the content based on fact check results."""
    prompt = f"""
        Based on the following fact check results, improve the draft: {state['draft']}
        fact check results:
        {state['fact_check_results']}
        Focus on correcting factual errors, enhancing clarity, and ensuring the content meets the requirements.
        """
    improved_content = llm.invoke(prompt).content
    print(" Step 3: Improved Content:\n", improved_content[:150] + "...")
    return{
        "improved_content": improved_content
    }

def format_ouput(state: ContentState) -> str:

    """Format content with HTML tags and elements"""
    prompt = f"""
        Format the following blog post for publication, using appropriate HTML tags and elements:
        {state['improved_content']}
        Ensure the formatting enhances readability and presentation.
        Include a meta description, a title, and appropriate headings.
        Output the formatted HTML
        """
    final_draft = llm.invoke(prompt).content
    print(" Step 4: Final Draft:\n", final_draft[:150] + "...")
    return{
        "final_draft": final_draft
    }


graph = StateGraph(ContentState)

graph.add_node(generate_draft)
graph.add_node(fact_check)
graph.add_node(improve_content)
graph.add_node(format_ouput)

graph.add_edge(START, "generate_draft")
graph.add_edge("generate_draft", "fact_check")
graph.add_edge("fact_check", "improve_content")
graph.add_edge("improve_content", "format_ouput")
graph.add_edge("format_ouput", END)

initial_state = {
    "topic": "The Future of Renewable Energy",
    "requirements": "Focus on recent advancements, challenges, and potential impact on the environment. Include at least 3 key points and ensure the content is engaging and informative."
}

compiled_graph = graph.compile()

result = compiled_graph.invoke(initial_state)
print("\nFinal Output:\n", result['final_draft'][:500] + "...")
