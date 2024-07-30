import random
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph import END, MessageGraph
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from typing import Any, Callable, Dict, Type, TypedDict, Annotated, Sequence, Union
from langchain_core.messages import BaseMessage
import operator
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel, Field


# using OllamaFunctions from experimental because it supports function binding with llms
model = OllamaFunctions(
    base_url="http://ai.mtcl.lan:11436",
    model="gemma2:27b", #dolphin-llama3:70b
    format="json"
    )

@tool
def get_current_weather(location: str) -> str:
    """Check the weather in the specified location"""
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return "Sunny, 78F"
    elif random.randint(0, 2) == 1:
        return "Cold, 22F"
    else:
        return "Rainy, 60F"

@tool
def get_system_time(location: str) -> str:
    """Get current system time, if no location is provided then pick location as Minnesota"""
    # ignore any passed arguments
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

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    api_call_count: int = 0
    


# messages = [HumanMessage("What is the time right now?")]
# llm_response = model_with_tools.invoke(messages)
# messages.append(llm_response)
# # print(messages)

messages = [HumanMessage("What is the weather in Chicago IL?")]
llm_response = model_with_tools.invoke(messages)
messages.append(llm_response)


# messages = [HumanMessage("I am not feeling good right now, can you help me with the system time?")]
# llm_response = model_with_tools.invoke(messages)
# messages.append(llm_response)
# # print(messages)


for tool_call in llm_response.tool_calls:
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

print(messages)

# Extract the last AI message from messages
last_ai_message = None
for msg in reversed(messages):
    if isinstance(msg, HumanMessage):
        continue
    elif isinstance(msg, AIMessage):
        last_ai_message = msg
        break

print("Last AI Message:", last_ai_message)
if last_ai_message and hasattr(last_ai_message, 'tool_calls'):
    print("Tool Calls:", last_ai_message.tool_calls)
else:
    print("No tool calls found.")