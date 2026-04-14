from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from typing  import TypedDict, List, Literal
from pydantic import BaseModel, Field
import os
import certifi
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

class OptimizationState(TypedDict):
    product_name: str
    product_features: List[str]
    target_audience: str
    current_description: str
    evaluation_result: dict 
    feedback: str
    iteration_count: int
    max_iterations: int
    is_approved: bool
    iteration_history: List[dict]

class ProductDescription(BaseModel):
    headline: str = Field(description="catchy headline with 10 words")
    description: str = Field(description="detailed product description with 100 words")
    key_benefits: List[str] = Field(description="list of 3 key benefits of the product")
    call_to_action: str = Field(description="compelling call to action")

class Evaluation(BaseModel):
    overall_score: int = Field(description="Overall score (1-10)", ge=1, le=10)
    clarity_score: int = Field(description="Clarity score (1-10)", ge=1, le=10)
    persuasiveness_score: int = Field(description="Persuasiveness score (1-10)", ge=1, le=10)
    audience_fit_score: int = Field(description="Audience fit score (1-10)", ge=1, le=10)
    is_approved: bool = Field(description="Whether the description is approved for use (score >= 8)")
    strengths: List[str] = Field(description="List of strengths of the current description")
    weaknesses: List[str] = Field(description="List of weaknesses of the current description")
    specific_feedback: str = Field(description="Detailed, actionable, Specific feedback for improvement")
    
llm = ChatOpenAI(model="gpt-5.4-mini")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

def generate_description(state: OptimizationState) -> OptimizationState:
    print("\n"+"="*70)
    print(f"ITERATION {state['iteration_count']}: Generating product description for {state['product_name']}")
    print("\n"+"="*70)

    iteration = state['iteration_count'] 

    optimizer_llm = llm.with_structured_output(ProductDescription)
    if iteration == 1:
        print("Creating initial product description ... ")

        prompt = f"""Create a compelling product description for:

        Product: {state['product_name' ]}
        Features: {', '.join(state['product_features' ] ) }
        Target Audience: {state['target_audience' ]}

        Requirements :
        - Headline: Catchy and concise (max 10 words)
        - Description: Engaging and informative (100-150 words)
        - Key Benefits: 3-5 clear, compelling benefits
        - Call-to-Action: Strong, action-oriented CTA

        Make it persuasive and tailored to the target audience."""
    else:
        print("Refining description based on evaluation feedback ...")
        prompt = f"""Improve this product description based on feedback:

        Product: {state['product_name' ]}
        Target Audience: {state['target_audience' ]}

        CURRENT DESCRIPTION:
        {state ['current_description' ]}

        EVALUATION SCORES:
        - Overall: {state['evaluation_result'].get('overall_score', 0)}/10
        - Clarity: {state['evaluation_result' ] .get('clarity_score', 0)}/10
        - Persuasiveness: {state['evaluation_result'] .get('persuasiveness_score', 0)}/10
        - Audience Fit: {state['evaluation_result' ] .get('audience_fit_score', 0)}/10

        FEEDBACK TO ADDRESS:
        {state ['feedback' ]}

        CRITICAL: Focus on the specific weaknesses mentioned. Make targeted improvements to:
        1. Address each point in the feedback
        2. Maintain the strengths that were working
        3. Increase scores in weak areas

        Generate a revised description that is more likely to be approved and addresses all feedback."""

    description = optimizer_llm.invoke(prompt)
    formatted_description = f"""
        HEADLINE: {description. headline}

        DESCRIPTION:
        {description. description}

        KEY BENEFITS:

        {chr(10) .join( [f"* {benefit}" for benefit in description.key_benefits])}

        CALL TO ACTION:
        {description. call_to_action}

    """
    print("Generated Description:\n", formatted_description)
    print("\n"+"="*70)

    return{
        "current_description": formatted_description,
        "iteration_count": iteration + 1
    }

def evaluate_description(state: OptimizationState) -> OptimizationState:
    print("\n"+"="*70)
    print(f"EVALUATION: reviewing description ...")
    print("\n"+"="*70)
    evaluation_llm = gemini_llm.with_structured_output(Evaluation)
    prompt = f"""Evaluate this product description objectively:

    Product: {state['product_name' ]}
    Target Audience: {state['target_audience' ]}

    DESCRIPTION TO EVALUATE:
    {state ['current_description' ]}

    Evaluate on these criteria (1-10 scale) :
    1. CLARITY: Is it clear and easy to understand?
    2. PERSUASIVENESS: Does it effectively sell the product?
    3. AUDIENCE FIT: Does it resonate with the target audience?

    APPROVAL CRITERIA: Overall score must be 8 or higher to approve.
    Provide:
    - Scores for each criterion
    - Overall score (average of criteria)
    - Whether it's approved (score >= 8)
    -"Specific strengths (what's working well)
    - Specific weaknesses (what needs improvement)
    - Actionable feedback for the next iteration

    Be objective and constructive."""

    evaluation = evaluation_llm.invoke(prompt)
    print("Evaluation Result:\n", evaluation.model_dump())
    print("\n"+"="*70)
    iteration_record = {
        "iteration": state['iteration_count'],
        "description": state['current_description'],
        "scores": {
            "overall": evaluation.overall_score,
            "clarity": evaluation.clarity_score,
            "persuasiveness": evaluation.persuasiveness_score,
            "audience_fit": evaluation.audience_fit_score
        },
        "feedback": evaluation.specific_feedback,
        "is_approved": evaluation.is_approved
    }

    history = state.get('iteration_history', [])
    history.append(iteration_record)

    return{
        "evaluation_result": evaluation.model_dump(),
        "feedback": evaluation.specific_feedback,
        "iteration_history": history,
        "is_approved": evaluation.is_approved
    }

def should_continue(state: OptimizationState) -> Literal["optimizer", "end"]:
    if state['is_approved']:
        print("\n"+"="*70)
        print("DESCRIPTION APPROVED! Ending optimization process.")
        print("\n"+"="*70)
        return "end"
    elif state['iteration_count'] >= state['max_iterations']:
        print("\n"+"="*70)
        print("MAX ITERATIONS REACHED. Ending optimization process.")
        print("\n"+"="*70)
        return "end"
    else:
        return "optimizer"
    
builder = StateGraph(OptimizationState)
builder.add_node("optimizer", generate_description)
builder.add_node("evaluator", evaluate_description)
builder.add_edge(START, "optimizer")
builder.add_edge("optimizer", "evaluator")
builder.add_conditional_edges("evaluator", should_continue, {"optimizer": "optimizer", "end": END})

graph = builder.compile()
result = graph.invoke({
    "product_name": "SmartHome AI  Assistant Security",
    "product_features": ["Voice control", "Home automation", "Energy monitoring", "Security alerts"],
    "target_audience": "Tech-savvy homeowners looking to automate their homes",
    "current_description": "",
    "evaluation_result": {},
    "feedback": "",
    "iteration_count": 1,
    "max_iterations": 5,
    "is_approved": False,
    "iteration_history": []
})
print("\n"+"="*70)
print("FINAL OPTIMIZED DESCRIPTION:\n", result['current_description'])
print("\n"+"="*70)


