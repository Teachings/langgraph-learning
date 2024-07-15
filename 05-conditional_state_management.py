import random
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from typing import Any, Callable, Dict, Type, TypedDict, Annotated, Sequence, Union
from langchain_core.messages import BaseMessage
import operator
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


# using OllamaFunctions from experimental because it supports function binding with llms
model = OllamaFunctions(
    base_url="http://ai.mtcl.lan:11434",
    model="gemma2:27b-instruct-q8_0", #dolphin-llama3:70b
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

# Define Agent prompt
prompt = PromptTemplate(
    template="""system
    You are a smart Agent. You are a master at understanding what a customer wants and utilize available tools only if you have to.

    user
    Conduct a comprehensive analysis of the request provided\

    USER REQUEST:\n\n {initial_request} \n\n
    
    assistant
    """,
    input_variables=["initial_request"],
)

agent_request_generator = prompt | model_with_tools
# result = agent_request_generator.invoke({"initial_request": "What is the weather in woodbury in MN?"})

# Define should continue prompt
category_generator_prompt = PromptTemplate(
    template="""system
    You are a Smart Router Agent. You are a master at reviewing whether the original question that customer asked was answered in the tool response.

     user
    Conduct a comprehensive analysis of the Initial Request from user and Tool Response and route the request into one of the following categories:
        continue - used when INITIAL REQUEST is not answered by TOOL RESPONSE or when TOOL RESPONSE is empty \
        end - used when INITIAL REQUEST is somewhat answered by TOOL RESPONSE \
        

            Output a single cetgory only from the types ('continue', 'end') \
            eg:
            'continue' \

    INITIAL REQUEST:\n\n {research_question} \n\n
    TOOL RESPONSE:\n\n {tool_response} \n\n
    
    assistant
    """,
    input_variables=["research_question", "tool_response"],
)

category_generator = category_generator_prompt | model | StrOutputParser()
# result = category_generator.invoke({"research_question": "What is the weather in woodbury in MN?", "tool_response":"65C, Sunny"})
# print(result)
# input("...")


class AgentState(TypedDict):
    research_question: str
    tool_response: str
    agent_response: AIMessage
    api_call_count: int = 0

  
def agent(state: AgentState):
    # print("STATE at agent start:", state)
    # input()
    last_ai_message = agent_request_generator.invoke({"initial_request": state["research_question"]})
    
    #append the response to the agent_response list in the state
    if last_ai_message is not None:
        state["agent_response"] = last_ai_message     
    return state
   
    
def should_continue(state: AgentState):
    # print("STATE:", state)
    # input()
    result = category_generator.invoke({"research_question": state["research_question"], "tool_response":state["tool_response"]})

    if isinstance(result, str) and result == "continue":
        print("Return continue")
        return "continue"
    else:
        print("Return end")
        return "end"



def call_tool(state: AgentState):
    # print("STATE:", state)
    agent_response = state["agent_response"]
    
    tool_call = agent_response.tool_calls[0]
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    state["api_call_count"] += 1
    print("Tool output:", tool_output)
    tool_message = ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
    if tool_output is not None:
        state["tool_response"] = tool_output
        # print("STATE:", state)
        # input()
    return state

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
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

app = workflow.compile()

#helper method to visualize graph
def save_graph_to_file(runnable_graph, output_file_path):
    png_bytes = runnable_graph.get_graph().draw_mermaid_png()
    with open(output_file_path, 'wb') as file:
        file.write(png_bytes)

save_graph_to_file(app, "output-05.png")


research_question = "How is the weather in munich today?"


state : AgentState = {"research_question": research_question,
                      "tool_response": [] ,
                      "agent_response": [],
                      "api_call_count": 0}
result = app.invoke(state)
print(result["tool_response"])
print(result["api_call_count"])