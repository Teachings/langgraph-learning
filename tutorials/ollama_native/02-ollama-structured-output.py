from enum import Enum
import random
from ollama import chat, Client
from pydantic import BaseModel, Field
from termcolor import colored

# Configuration: API Endpoint and Ollama Client
API_URL = "http://localhost:11434"
client = Client(host=API_URL)

class TemperatureTone(str, Enum):
    HOT = "hot"
    WARM = "warm"
    NEUTRAL = "neutral"
    COOL = "cool"
    COLD = "cold"

# Pydantic Schema for Response Evaluation
class Evaluation(BaseModel):
    result: bool = Field(description="True or False, with some leniency in response adequacy.", required=True)
    explanation: str = Field(description="Explanation of the evaluation result.", required=True)
    temperatureTone: TemperatureTone = Field(description="Specifies whether the weather conditions are hot, warm, neutral, cool or cold", required=True)

def evaluate_response(initial_request, tool_response):
    """Evaluate the adequacy of the tool response with loosened criteria."""
    evaluation_prompt = f"""
    **Initial Query:** {initial_request}
    **Tool Response:** {tool_response}
    
    Evaluate the response based on:
    1. Core Relevance: Does the response partially or fully address the user's query?
    2. Accuracy: Is the information reasonable and aligns with typical weather patterns?
    3. Clarity: Is the response generally understandable?

    Provide a result (True/False, allowing partial correctness) with an explanation and Heat Scale based on the response.
    """
    
    response = client.chat(
        messages=[
            {"role": "system", "content": "You are a skilled evaluator. Assess the tool response with leniency."},
            {"role": "user", "content": evaluation_prompt}
        ],
        model="qwen2.5-coder:32b",
        format=Evaluation.model_json_schema()
    )

    return Evaluation.model_validate_json(response.message.content)

# Dummy method to simulate processing a request with tools
def process_request_with_tools(query):
    weather_responses = [
        "Temperature is 30 degrees Fahrenheit and it is blazing hot outside.",
        "It's currently 75 degrees Fahrenheit, feeling quite warm today.",
        "The temperature is around 50 degrees Fahrenheit, just right.",
        "It's a cool 45 degrees Fahrenheit with a gentle breeze.",
        "Brr, it's only 20 degrees Fahrenheit and quite cold out there.",
        "Sunny day ahead with temperatures reaching up to 85 degrees Fahrenheit.",
        "Expect a mild day with temperatures in the mid-60s Fahrenheit.",
        "It's chilly today with temperatures dropping to the low 30s Fahrenheit.",
        "A perfect day for outdoor activities with temperatures at 68 degrees Fahrenheit.",
        "The weather is brisk, with temperatures around 55 degrees Fahrenheit.",
        "It's a balmy day with temperatures in the high 70s Fahrenheit.",
        "Stay cozy indoors; it's freezing outside at just 15 degrees Fahrenheit.",
        "Cloudy skies today, but not too cold with temperatures in the low 60s Fahrenheit.",
        "Spring is here with temperatures climbing to around 72 degrees Fahrenheit.",
        "Winter chill prevails with temperatures barely reaching 32 degrees Fahrenheit."
    ]
    response = random.choice(weather_responses)
    print(colored(f"Randomly Picked Tool Response: {response}\n", "cyan"))
    return response

def display_evaluation(evaluation: Evaluation):
    """Display the evaluation result in a user-friendly format."""
    print(colored("Evaluation:", attrs=['bold']))
    print(f"Result: {colored(evaluation.result, 'green' if evaluation.result else 'red')}")
    print(f"Explanation: {evaluation.explanation}")
    print(f"Temperature Tone: {colored(evaluation.temperatureTone.value.capitalize(), attrs=['bold'])}")

def run_demo():
    """Run the AI assistant demo with predefined queries."""
    queries = [
        "What is the temperature in Chicago, is it cold?",
        "Is it warm in New York today?",
        "Tell me if it is hot in Los Angeles.",
        "What's the weather like in Seattle, cool or cold?",
        "How neutral is the temperature in Miami right now?"
    ]

    for query in queries:
        print(colored(f"\nProcessing Query: {query}\n", "blue", attrs=['bold']))
        tool_response = process_request_with_tools(query)
        evaluation = evaluate_response(query, tool_response)
        display_evaluation(evaluation)
        input(colored("\nPress Enter to continue...", "yellow"))

if __name__ == "__main__":
    run_demo()