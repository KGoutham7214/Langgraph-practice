from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from langgraph.types import Send
from operator import add
from dotenv import load_dotenv
import os  
import certifi
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

class OverallState(TypedDict):
    research_topic: str
    sources: List[str]
    worker_findings: Annotated[List[dict], add]
    final_report: str

class WorkerState(TypedDict):
    source: str
    worker_id: int
    reasearch_topic: str 

class ReasearchPlan(BaseModel):
    sources: List[str] = Field(description="List of sources to research", max_length=5)
    reasoning: str = Field(description="Reasoning behind choosing these sources")


llm = ChatOpenAI(model="gpt-5.4-mini")

#orchestrator
def plan_reasearch(state: OverallState) -> OverallState:
    print("\n"+"="*70)
    print(f"ORCHESTRATOR: Planning research strategy for topic: {state['research_topic']}")
    print("\n"+"="*70)

    prompt = f"""You are a research strategist planning a comprehensive
    investigation.

    Research Topic: {state['research_topic']}

    CRITICAL INSTRUCTION: Generate between 3-5 specific research sources
    or aspects to investigate.
    DO NOT generate more than 5 sources.

    Each source should be:
    - Specific and focused on a distinct aspect
    - Relevant to the overall topic
    - Complementary to other sources (minimal overlap)
    - Concrete enough to guide targeted research

    Examples of good sources:
    - "Clinical trial results and efficacy data"
    - "Economic impact and cost-benefit analysis"
    - "Regulatory framework and compliance requirements"
    - "Patient outcomes and quality of life metrics"
    - "Industry adoption rates and market trends"

    Generate sources that will provide comprehensive coverage of the
    topic."""

    planner_llm = llm.with_structured_output(ReasearchPlan)
    research_plan = planner_llm.invoke(prompt)
    print(f"Generated Research Plan: {research_plan.sources}")

    for idx, source in enumerate(research_plan.sources):
        print(f"{idx}. {source}")
    
    print(f"Reasoning: {research_plan.reasoning}") 

    return{
        "sources": research_plan.sources
    }

#worker
def research_worker(state: WorkerState) -> WorkerState:
    worker_id = state['worker_id']
    source = state['source']
    print("\n"+"-"*50)
    prompt = f"""You are a specialized researcher investigating: {state
    ['research_topic']}

    Your specific focus area: {source}

    Conduct thorough research on this aspect and provide:

    1. KEY FINDINGS (3-5 specific points)
    - What are the most important discoveries or facts?

    2. DATA & STATISTICS
    - Relevant numbers, percentages, or quantitative information

    3. INSIGHTS & ANALYSIS
    - What does this information mean?
    - How does it relate to the broader topic?

    4. SOURCES & CREDIBILITY
    - Types of sources you would consult (academic, industry,
    government, etc.)

    5. IMPLICATIONS
    - Why does this matter for understanding the overall topic?

    Be specific, factual, and provide depth on this particular aspect."""

    response = llm.invoke(prompt).content
    findings = {
        "worker_id": worker_id,
        "source": source,
        "content": response
    }

    return {
        "worker_findings": [findings]
    }

#synthesizer
def synthesize_report(state: OverallState) -> OverallState:
    print("\n"+"="*70)
    print("SYNTHESIZER: Compiling final report from worker findings")
    print("\n"+"="*70)

    all_findings = "\n\n".join([
        f"Worker {finding['worker_id']} - Source: {finding['source']}\n{finding['content']}"
        for finding in state['worker_findings']
    ])

    prompt = f"""You are an expert synthesizer compiling a comprehensive
    report on: {state['research_topic']}

    You have the following findings from specialized workers:

    {all_findings}

    Your task is to create a cohesive, well-structured report that:
    - Integrates the key findings from all workers
    - Provides clear explanations and connections between different
    sources
    - Highlights the most important insights and implications
    - Presents the information in a logical flow

    Generate a final report that effectively communicates the overall
    understanding of the research topic based on the workers' findings."""

    final_report = llm.invoke(prompt).content

    return {
        "final_report": final_report
    }

#conditional edge function
def create_research_workers(state: OverallState) -> List[WorkerState]:
    print("\n"+"="*70)
    print("ORCHESTRATOR: Creating research workers for each source")
    print("\n"+"="*70)

    return[
        Send("research_worker", {
            "source": source,
            "worker_id": idx + 1,
            "research_topic": state['research_topic']
        })
        for idx, source in enumerate(state['sources'])
    ]


builder = StateGraph(OverallState)
builder.add_node("plan_reasearch", plan_reasearch)
builder.add_node("research_worker", research_worker)
builder.add_node("synthesize_report", synthesize_report)
builder.add_edge(START, "plan_reasearch")
builder.add_conditional_edges("plan_reasearch", create_research_workers, ["research_worker"])
builder.add_edge("research_worker", "synthesize_report")
builder.add_edge("synthesize_report", END)

graph = builder.compile()

initial_state = {
    "research_topic": "The impact of Artificial Intelligence on healthcare outcomes",
    "sources": [],
    "worker_findings": [],
    "final_report": ""
}
final_state = graph.invoke(initial_state)
print("\n"+"="*70)
print("FINAL SYNTHESIZED REPORT:")
print(final_state['final_report']) 

