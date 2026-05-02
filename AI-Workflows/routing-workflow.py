from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import certifi

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

class SupportState(TypedDict):
    customer_query: str
    query_category: str
    response : str
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-5.4-mini")

class QueryClassification(BaseModel):
    """Classify the customer query into a category."""
    category: Literal["billing", "technical", "refund", "general"]
    confidence : float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="why this category was chosen")

def classify_query(state: SupportState) -> SupportState:
    """Classify the customer query into a category."""
    
    classifier_llm = llm.with_structured_output(QueryClassification)
    prompt = f"""Classify the following customer support query into one
    of these categories:

    - billing: Questions about invoices, payments, charges, pricing
    - technical: Technical issues, bugs, feature problems, integration issues
    - refund: Refund requests, returns, cancellations
    - account: Account access, password reset, profile changes, login issues
    - general: General questions, feature inquiries, information requests

    Customer Query: {state['customer_query']}

    Classify accurately based on the primary intent."""

    classification_result = classifier_llm.invoke(prompt)

    print(f"Classification Result: {classification_result.category}")

    return{
        "query_category": classification_result.category,
    }


def handle_billing(state: SupportState) -> SupportState:
    print("= BILLING HANDLER: Processing billing query")
    prompt = f"""You are a billing specialist. Handle this customer
    query:
    {state['customer_query' ]}
    Provide a helpful response that:
    - References billing policies and procedures
    - Offers to check their account details
    - Provides clear next steps
    - Mentions relevant payment options
    Keep it professional and reassuring."""

    response = llm.invoke(prompt).content
    return{
        "response": response,
        "tools_used": ["billing_db", "payment"]
    }

def handle_technical(state: SupportState) -> SupportState:
    print("> TECHNICAL HANDLER: Processing billing query")
    prompt = f"""You are a technical support specialist. Handle this
    customer query:
    {state ['customer_query' ]}
    Provide a helpful response that:
    - Offers specific troubleshooting steps
    - References relevant documentation
    - Asks clarifying questions if needed
    - Provides workarounds if applicable
    Be clear and technical but accessible."""
    response = llm.invoke(prompt).content
    return{
        "response": response,
        "tools_used": ["technical_knowledge_base", "troubleshooting_guide"]
    }

def handle_refund(state: SupportState) -> SupportState:
    print("> REFUND HANDLER: Processing refund query")
    prompt = f"""You are a refund specialist. Handle this customer
    query:
    {state ['customer_query' ]}
    Provide a helpful response that:
    - References refund policies and eligibility
    - Offers to check their order details
    - Provides clear next steps for requesting a refund
    - Mentions any relevant timelines or conditions
    Be empathetic and clear."""
    response = llm.invoke(prompt).content
    return{
        "response": response,
        "tools_used": ["refund_policy_db", "order_management_system"]
    }

def handle_general(state: SupportState) -> SupportState:
    """Handle general queries"""
    prompt = f"""You are a general support specialist. Handle this
    customer query:
    state['customer_query' ]H
    Provide a helpful response that:
    - Answers their question clearly
    - Provides relevant links or resources
    - Offers additional help if needed
    - Suggests related features they might find useful
    Be friendly and informative."""

    response = llm. invoke(prompt).content
    return{
        "response": response,
        "tools_used": ["general_faq", "product_docs"]
    }

def route_query(state: SupportState) -> Literal["billing", "technical", "refund", "general"]:
    """Route the query to the appropriate handler based on the category."""
    return state['query_category']


builder = StateGraph(SupportState)
builder.add_node("classify_query", classify_query)
builder.add_node("handle_billing", handle_billing)
builder.add_node("handle_technical", handle_technical)
builder.add_node("handle_refund", handle_refund)
builder.add_node("handle_general", handle_general)

builder.add_edge(START, "classify_query")
builder.add_conditional_edges("classify_query", route_query, {
    "billing": "handle_billing",
    "technical": "handle_technical",
    "refund": "handle_refund",
    "general": "handle_general"
})

builder.add_edge("handle_billing", END)
builder.add_edge("handle_technical", END)
builder.add_edge("handle_refund", END)
builder.add_edge("handle_general", END)

graph = builder.compile()
initial_state = {
    "customer_query": "I was charged twice for my subscription. Can you help me fix this?"
}

final_state = graph.invoke(initial_state)
print("Final State:", final_state['response'])