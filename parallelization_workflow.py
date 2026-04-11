from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

class OverallState(TypedDict):
    """A structured representation of the overall state."""
    topic: str
    instagram_post: str
    twitter_post: str
    linkedin_post: str
    final_output: str

llm = ChatOpenAI(model="gpt-5.4-mini")

def generate_instagram_post(state: OverallState) -> str:
    """ Generate an engaging Instagram post with emojis and hashtags. """
    topic = state["topic"]
    prompt = f"""
    Create an engaging Instagram post about the topic: {topic}. Use emojis and relevant hashtags to make it appealing.
    Requirements:
    - Engaging and visual language
    - 2-3 short paragraphs (150-200 words max)
    - Include relevant emojis
    - End with 5-8 relevant hashtags
    - Casual, friendly tone
    - Call-to-action to engage with the post
    
    """ 
    insta_post = llm.invoke(prompt).content

    print("Generated Instagram Post")
    return {
        "instagram_post": insta_post
    }

def generate_twitter_post(state: OverallState) -> str:
    """ Generate a concise and impactful Twitter post. """
    topic = state["topic"]
    prompt = f"""
    Create a concise and impactful Twitter post about the topic: {topic}. 
    Requirements:
    - 280 characters max
    - Engaging and attention-grabbing language
    - Include relevant emojis
    - End with 2-3 relevant hashtags
    - Casual, friendly tone
    - Call-to-action to engage with the post
    
    """ 
    twitter_post = llm.invoke(prompt).content

    print("Generated Twitter Post")
    return {
        "twitter_post": twitter_post
    }

def generate_linkedin_post(state: OverallState) -> str:
    """ Generate a professional and informative LinkedIn post. """
    topic = state["topic"]

    prompt = f"""
    Create a professional and informative LinkedIn post about the topic: {topic}. 
    Requirements:
    - 150-200 words max
    - Professional and informative language
    - End with 2-3 relevant hashtags
    - Professional tone
    - Call-to-action to engage with the post
    
    """ 
    linkedin_post = llm.invoke(prompt).content

    print("Generated LinkedIn Post")
    return {
        "linkedin_post": linkedin_post
    }

def aggregate_posts(state: OverallState) -> str:
    """ Aggregate the generated posts into a final output. """
    topic = state["topic"]
    instagram_post = state["instagram_post"]
    twitter_post = state["twitter_post"]
    linkedin_post = state["linkedin_post"]

    final_output = f"""
    Topic: {topic}
    {'='*50}
    Instagram Post:
    {instagram_post}
    {'='*50}
    Twitter Post:
    {twitter_post}
    {'='*50}
    LinkedIn Post:
    {linkedin_post}
    
    """ 
    print("Aggregated Final Output")
    return {
        "final_output": final_output
    }

builder = StateGraph(OverallState)

builder.add_node(generate_instagram_post)
builder.add_node(generate_twitter_post)
builder.add_node(generate_linkedin_post)
builder.add_node(aggregate_posts)

builder.add_edge(START, "generate_instagram_post")
builder.add_edge(START, "generate_twitter_post")
builder.add_edge(START, "generate_linkedin_post")

builder.add_edge("generate_instagram_post", "aggregate_posts")
builder.add_edge("generate_twitter_post", "aggregate_posts")
builder.add_edge("generate_linkedin_post", "aggregate_posts")
builder.add_edge("aggregate_posts", END)

graph = builder.compile()

topic = "The Benefits of Remote Work"
initial_state = OverallState(
    {"topic": topic,
    "instagram_post": "",
    "twitter_post": "",
    "linkedin_post": "",
    "final_output": "",}
)

result = graph.invoke(initial_state)
print(result["final_output"])