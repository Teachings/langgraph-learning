import random
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.prompts import PromptTemplate

# using OllamaFunctions from experimental because it supports function binding with llms
model = OllamaFunctions(
    base_url="http://ai.mtcl.lan:11436",
    model="llama3.1:70b",
    format="json"
    )

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a specified location.
    
    This tool simulates checking the weather by randomly selecting from three possible outcomes: sunny, cold, or rainy. 
    The chance of each outcome is equal (1/3). If the random check fails, it may return an unexpected result to simulate real-world unpredictable conditions.
    
    Args:
        location (str): The name of the location for which to check the weather.
        
    Returns:
        str: A string describing the current weather in the specified location, randomly chosen from three possible outcomes.
    """
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return "Sunny, 78F"
    elif random.randint(0, 2) == 1:
        return "Cold, 22F"
    else:
        return "Rainy, 60F"

@tool
def get_system_time(location: str = "Minnesota") -> str:
    """Get the current system time. If no location is provided, use default location as 'Woodbury, Minnesota'.
    
    This tool simulates retrieving the system time by randomly selecting from three possible outcomes: morning, afternoon, or evening. 
    The chance of each outcome is equal (1/3). If the random check fails, it may return an unexpected result to simulate real-world unpredictable conditions.
    
    Args:
        location (str): Optional. The name of the location for which to retrieve the system time. It defaults to Minnesota if not provided.  
        
    Returns:
        str: A string describing the current system time in the specified or default location, randomly chosen from three possible outcomes.
    """
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return "2:00 AM"
    elif random.randint(0, 2) == 1:
        return "3:00 PM"
    else:
        return "6:15 PM"

model_with_tools = model.bind_tools(
    tools=[get_current_weather, get_system_time],
)

tool_mapping = {
    'get_current_weather': get_current_weather,
    'get_system_time': get_system_time,
}

# messages = [HumanMessage("What is the weather in Chicago IL?")]
# llm_response = model_with_tools.invoke(messages)
# messages.append(llm_response)


# Define prompt
prompt = PromptTemplate(
    template="""system
    You are a smart Agent. 
    You are a master at understanding what a customer wants and utilize available tools only if you have to.

    user
    Conduct a comprehensive analysis of the request provided\

    USER REQUEST:\n\n {initial_request} \n\n
    
    assistant
    """,
    input_variables=["initial_request"],
)

agent_request_generator = prompt | model_with_tools

result = agent_request_generator.invoke(
    {"initial_request": "What is the weather in woodbury in MN?"}
    )

# Extract the last AI message from messages
last_ai_message = None
if isinstance(result, AIMessage):
    last_ai_message = result

# print("Last AI Message:", last_ai_message)

if last_ai_message and hasattr(last_ai_message, 'tool_calls'):
    print("last_ai_message:", last_ai_message)
    print("tool name :: ", last_ai_message.tool_calls[-1]["name"])
    print("tool args :: ", last_ai_message.tool_calls[-1]["args"])
else:
    print("No tool calls found.")
