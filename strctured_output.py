from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv
load_dotenv()

class ProductReview(BaseModel):
    """A structured representation of a product review."""
    product_name: str = Field(description="Name of the product")
    sentiment: str = Field(description="Sentiment of the review (positive, negative, neutral)")
    rating: int = Field(description="Rating given by the reviewer (1-5)", ge=1, le=5)
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="A brief summary of the review") 

llm = ChatOpenAI(model="gpt-5.4-mini")

structured_llm = llm.with_structured_output(ProductReview)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a product review analyzer, Extract structured information from reviews."),
    ("user", "{review_text}")
])

chain = prompt | structured_llm

result = chain.invoke({"review_text": "I recently bought the XYZ headphones. The sound quality is amazing and the battery life is excellent. However, they are a bit uncomfortable to wear for long periods. Overall, I would give them 4 out of 5 stars."})
print(json.dumps(result.model_dump(), indent=2))