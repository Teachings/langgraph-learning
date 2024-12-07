from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from termcolor import colored
from tqdm import tqdm
import random
import json

# Import tools from tools.py
from tools import get_current_weather, get_system_time

# Initialize the model
model = ChatOllama(
    base_url="http://ai.mtcl.lan:11435",
    model="llama3.3",
    format="json"
)

model_with_tools = model.bind_tools(
    tools=[get_current_weather, get_system_time],
)

tool_mapping = {
    'get_current_weather': get_current_weather,
    'get_system_time': get_system_time,
}

# Define prompt template
prompt = PromptTemplate(
    template="""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
        You are a smart Agent. 
        You are a master at understanding what a customer wants and utilize available tools only if you have to.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
        Conduct a comprehensive analysis of the request provided. \n
        USER REQUEST:\n\n {initial_request} \n\n
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["initial_request"],
)

agent_request_generator = prompt | model_with_tools

# Test suite size configuration
test_suite_size = int(input("Enter the number of test cases to generate for the test suite: "))

# Cities and specific requests
cities_and_states = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
    "Miami, FL", "San Francisco, CA", "Seattle, WA", "Boston, MA",
    "Austin, TX", "Denver, CO", "Philadelphia, PA", "Phoenix, AZ",
    "Dallas, TX", "San Diego, CA", "Atlanta, GA", "Washington D.C., DC",
    "Orlando, FL", "Nashville, TN", "Minneapolis, MN", "Las Vegas, NV"
]

specific_requests = [
    "What is the weather today in [location]?",
    "Can you tell me the forecast for tomorrow in [location]?",
    "How's the weather in [location] today?",
    "What's the temperature like right now in [location]?",
    "Tell me about the conditions for tonight in [location]?",
    "Is it going to rain tomorrow in [location]?",
]

# Generate test sentences based on user-defined test suite size
test_sentences = [
    random.choice(specific_requests).replace("[location]", random.choice(cities_and_states))
    for _ in range(test_suite_size)
]

# Display generated test sentences
print(colored("Generated Test Sentences:", "cyan"))
for sentence in test_sentences:
    print(colored(sentence, "white"))

# Initialize counters
function_call_count = 0
results = []

# Test suite execution
print(colored("\nRunning Test Suite...\n", "cyan"))
for i, sentence in enumerate(tqdm(test_sentences, desc="Testing Sentences"), start=1):
    print(colored(f"\nTest #{i}: {sentence}", "yellow"))
    try:
        result = agent_request_generator.invoke({"initial_request": sentence})
        
        # Check for AI message and tool calls
        if isinstance(result, AIMessage) and hasattr(result, 'tool_calls'):
            last_tool_calls = result.tool_calls
            print(colored(f"AI Response: {result}", "green"))
            if last_tool_calls:
                for tool_call in last_tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    print(colored(f"Tool Invoked: {tool_name} with args {tool_args}", "blue"))
                    tool_output = tool_mapping[tool_name.lower()].invoke(tool_args)
                    print(colored(f"Tool Output: {tool_output}", "magenta"))
                    function_call_count += 1
            else:
                print(colored("No tools invoked by the model.", "red"))
        else:
            print(colored("No AI response or tool calls detected.", "red"))

        # Save results
        results.append({"sentence": sentence, "result": str(result), "tool_calls": last_tool_calls or []})

    except Exception as e:
        print(colored(f"Error during test execution: {str(e)}", "red"))
        results.append({"sentence": sentence, "error": str(e)})

# Summary
success_rate = (function_call_count / len(test_sentences)) * 100
print(colored(f"\nTest Suite Completed.", "cyan"))
print(colored(f"Total Sentences Tested: {len(test_sentences)}", "cyan"))
print(colored(f"Function Calls Made: {function_call_count}", "green"))
print(colored(f"Success Rate: {success_rate:.2f}%", "green"))

# Save results to a file
with open("test_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(colored("\nResults saved to 'test_results.json'.", "cyan"))
