from ollama import chat, Client
from pydantic import BaseModel, Field
from termcolor import colored

# Configuration: API Endpoint and Ollama Client
API_URL = "http://ai.mtcl.lan:11434"
client = Client(host=API_URL)

# Pydantic Schema for Response Evaluation
class Evaluation(BaseModel):
    result: bool = Field(description="True or False, with some leniency in response adequacy.", required=True)
    explanation: str = Field(description="Explanation of the evaluation result.", required=True)

def evaluate_response(initial_request, tool_response):
    """Evaluate the adequacy of the tool response with loosened criteria."""
    evaluation_prompt = f"""
    **Initial Query:** {initial_request}
    **Tool Response:** {tool_response}
    
    Evaluate the response based on:
    1. Core Relevance: Does the response partially or fully address the user's query?
    2. Accuracy: Is the information reasonable and aligns with typical weather patterns?
    3. Clarity: Is the response generally understandable?

    Provide a result (True/False, allowing partial correctness) with an explanation.
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
    response = "It's chilly today with temperatures dropping to the low 30s Fahrenheit."
    print(colored(f"Dummy Tool Response: {response}\n", "cyan"))
    return response

def display_evaluation(evaluation: Evaluation):
    """Display the evaluation result in a user-friendly format."""
    print(colored("Evaluation:", attrs=['bold']))
    print(f"Result: {colored(evaluation.result, 'green' if evaluation.result else 'red')}")
    print(f"Explanation: {evaluation.explanation}")

def run_demo():
    """Run the AI assistant demo with predefined queries."""

    query = "What is the temperature in Chicago, is it cold?"

    print(colored(f"\nProcessing Query: {query}\n", "blue", attrs=['bold']))
    tool_response = process_request_with_tools(query)
    evaluation = evaluate_response(query, tool_response)
    display_evaluation(evaluation)

if __name__ == "__main__":
    run_demo()