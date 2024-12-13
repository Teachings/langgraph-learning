from ollama import chat, Client
from pydantic import BaseModel, Field
from tools import get_current_weather, get_system_time
from termcolor import colored

# Configuration: API Endpoint
API_URL = "http://ai.mtcl.lan:11434"

# Tools and Tool Functions
TOOLS = [
    get_current_weather.tool_definition,
    get_system_time.tool_definition
]

tool_functions = {
    "get_current_weather": get_current_weather,
    "get_system_time": get_system_time
}

# Client Initialization
client = Client(host=API_URL)

# Pydantic Schema for Response Evaluation
class Evaluation(BaseModel):
    result: bool = Field(description="True or False", required=True)
    explanation: str = Field(description="Explanation of the evaluation result.", required=True)

def process_request_with_tools(initial_request):
    """Process a user request and return the AI's response."""
    system_message = """
    You are a knowledgeable and helpful AI assistant. 
    Always strive to provide the best possible response to user requests. 
    If a request requires information or actions beyond your current capabilities, utilize the available tools judiciously.

    **Examples:**
    **Generic Questions:**
    * **User Request:** "Tell me a joke."
    * **Your Response:** Provide a humorous response directly, without invoking any tools.
    * **User Request:** "What is the meaning of life?"
    * **Your Response:** Provide a thoughtful and informative response, drawing on philosophical and existential concepts.

    Remember to use tools only when necessary. If a request can be answered directly, do so without invoking any tools.
    """

    # AI Response
    response = client.chat(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": initial_request}
        ],
        model="qwen2.5-coder:32b",
        tools=TOOLS,
        format="json"
    )

    print(colored("System Response (Raw):", "cyan", attrs=['bold']))
    print(colored(response.message.content, "white"))

    # Process Tool Calls
    if response.message.tool_calls:
        print(colored("\nProcessing Tool Calls...", "magenta", attrs=['bold']))
        for tool_call in response.message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_function = tool_functions.get(function_name)

            try:
                result = tool_function(**arguments)
                print(colored(f"Tool: {function_name}", "yellow", attrs=['bold']))
                print(colored(f"Arguments: {arguments}", "white"))
                print(colored(f"Result: {result}", "green"))
                # Append tool result to the response content
                response.message.content += f"\nTool Result ({function_name}): {result}"
            except Exception as e:
                print(colored(f"Error executing tool '{function_name}': {e}", "red"))

    return response.message.content


def run_demo():
    """Run the AI assistant demo with predefined queries."""
    queries = [
        "Tell me a joke about programming.",
        "What is the current temperature and forecast for tomorrow in Woodbury, MN?",
        "Turn on the living room lights and adjust them to 50% brightness.",
        "What is the exact time in New York City right now?",
        "Solve this math problem: What is 2 + 2 multiplied by 3?",
        "What are the weather conditions in San Francisco this weekend?",
        "Can you turn off all the lights in the house?",
        "What is the square root of 144?",
        "Give me a detailed weather report for Paris, France.",
        "Play some relaxing music in the bedroom.",
        "What is the capital of France?",
        "How long would it take to fly from New York to Tokyo?",
        "Set an alarm for 6:30 AM tomorrow.",
        "Tell me an interesting fact about space.",
        "What's the current system time on this device?"
    ]

    for query in queries:
        print(colored(f"\nProcessing Query: {query}\n", "blue", attrs=['bold']))
        tool_response = process_request_with_tools(query)
        input(colored("\nPress Enter to continue...", "yellow"))

if __name__ == "__main__":
    run_demo()