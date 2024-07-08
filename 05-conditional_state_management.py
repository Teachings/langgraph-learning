import random
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import END, StateGraph
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
    base_url="http://ai.mtcl.lan:11434",
    model="mixtral", #dolphin-llama3:70b
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
    
def should_continue(state: AgentState):
    print("STATE:", state)
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        input()
        return "continue"


def call_model(state: AgentState):
    print("STATE:", state)
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "api_call_count": state["api_call_count"]}


def call_tool(state: AgentState):
    print("STATE:", state)
    messages = state["messages"]
    last_message = messages[-1]
    tool_call = last_message.tool_calls[0]
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    state["api_call_count"] += 1
    print("Tool output:", tool_output)
    print("API call count after this tool call:", state["api_call_count"])
    tool_message = ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
    return {"messages": [tool_message], "api_call_count": state["api_call_count"]}

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")
workflow.set_entry_point("agent")

app = workflow.compile()

#helper method to visualize graph
def save_graph_to_file(runnable_graph, output_file_path):
    png_bytes = runnable_graph.get_graph().draw_mermaid_png()
    with open(output_file_path, 'wb') as file:
        file.write(png_bytes)

save_graph_to_file(app, "output-05.png")

system_message = SystemMessage(
    content="You are responsible for answering user questions. You use tools for that."
)

human_message = HumanMessage(content="How is the weather in munich today?")
messages = [system_message, human_message]

result = app.invoke({"messages": messages, "api_call_count": 0})

print(result["messages"][-1].content)
print(result["api_call_count"])