from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
import random

class WeatherState(TypedDict):
    city: str
    temperature: float
    conditions: str

class ApiError(Exception):
    """Simulated API error"""
    pass

def fetch_weather(state: WeatherState) -> dict:
    print(f"Fetching weather for {state['city']}...")
    # Simulate a 50% chance of API failure
    if random.random() < 0.5:
        print("API call failed!")
        raise ApiError("Failed to fetch weather data")
    
    # Simulate successful API response
    state["temperature"] = round(random.uniform(60, 100), 1)  # Random temperature between 60 and 100
    state["conditions"] = random.choice(["Sunny", "Cloudy", "Rainy"]) 
    print("API call succeeded!")
    return state

graph = StateGraph(WeatherState)
graph.add_node(
    "fetch_weather", 
    fetch_weather, 
    retry_policy=RetryPolicy(
        max_attempts=5, 
        initial_interval=1,
        max_interval=10,
        jitter=True,
        retry_on=ApiError,
        backoff_factor=2
    )
)
graph.add_edge(START, "fetch_weather")
graph.add_edge("fetch_weather", END)

agent = graph.compile()
initial_state = {"city": "New York", "temperature": 0.0, "conditions": ""}
try:
    final_state = agent.invoke(initial_state)
    print(final_state)
except ApiError as e:
    print(f"All retry attempts failed: {e}")
