from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.prompts import PromptTemplate
import random

# Import tools from tools.py
from tools import get_current_weather, get_system_time

# using OllamaFunctions from experimental because it supports function binding with llms
model = OllamaFunctions(
    base_url="http://ai.mtcl.lan:11436",
    model="llama3.1:70b",
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
    template=
    """
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

# List of different sentences to test
# List of cities and states for more specific location entries
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
    # "What time does the museum open in [location]?"
]

# Generate test sentences
test_sentences = []
test_suite_size = 25
for _ in range(test_suite_size):
    city_and_state = random.choice(cities_and_states)
    specific_request = random.choice(specific_requests).replace("[location]", city_and_state)
    test_sentences.append(specific_request)

# Print the generated sentences for reference
print("Generated Test Sentences:")
for sentence in test_sentences:
    print(sentence)

# Initialize a counter for function calls
function_call_count = 0
counter=0
# Loop through the test sentences and invoke the model with each one
for sentence in test_sentences:
    counter = counter+1
    print("----------- Loop # ",counter, "--------------")
    result = agent_request_generator.invoke({"initial_request": sentence})
    
    # Extract the last AI message from messages
    last_ai_message = None
    if isinstance(result, AIMessage):
        last_ai_message = result

    # Print information about the tool calls
    if last_ai_message and hasattr(last_ai_message, 'tool_calls'):
        print("Sentence: ", sentence)
        print("Last AI Message:", last_ai_message)
        if len(last_ai_message.tool_calls) > 0:
            # Increment the function call counter for each tool call
            function_call_count += 1
            print("Tool Name :: ", last_ai_message.tool_calls[-1]["name"])
            print("Tool Args :: ", last_ai_message.tool_calls[-1]["args"])
    else:
        print("No tool calls found for Sentence:", sentence)
    
    # Run the tool if there are any tool calls
    if hasattr(last_ai_message, 'tool_calls'):
        for tool_call in last_ai_message.tool_calls:
            tool = tool_mapping[tool_call["name"].lower()]
            tool_output = tool.invoke(tool_call["args"])
            print("Tool Output:", tool_output)
    else:
        print("No tools to run for Sentence:", sentence)
    
    # Add a newline for better readability between different sentences' results
    print("\n")
    print("Total function calls made by the AI model:", function_call_count)
    print("\n \n \n")

#print percentage of function calls vs test_suite_size
print("Success Rate :: ", function_call_count/test_suite_size*100, " percent")

