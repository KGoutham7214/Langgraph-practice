from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()



@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    
    weather_data = {
        "New York": "22  cloudy",
        "Los Angeles": "28  sunny",
        "Chicago": "18  rainy",
    }

    return weather_data.get(city,"Weather data not available for this city.")

@tool
def calculate_tip(total_bill: float, tip_percentage: float) -> float:
    """Calculate the tip amount based on the total bill and tip percentage."""
    return round(total_bill * (tip_percentage / 100), 2)


llm = ChatOpenAI(model="gpt-5.4-mini")

llm_with_tools = llm.bind_tools([
    get_weather,
    calculate_tip
])

weather_propmpt = "What is the current weather in New York?"
tip_prompt = "Calculate the tip for a total bill of $50 with a tip percentage of 20%."

response = llm_with_tools.invoke(weather_propmpt)

tool_calls = response.tool_calls

for tool_call in tool_calls:
    if tool_call['name'] == "get_weather":
        result = get_weather.invoke(tool_call['args'])
    elif tool_call['name'] == "calculate_tip":
        result = calculate_tip.invoke(tool_call['args'])
    else:
        result = "Unknown tool called."

print(result)



