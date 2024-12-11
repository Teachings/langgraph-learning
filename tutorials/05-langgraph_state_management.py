from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from typing import TypedDict
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from termcolor import colored
import json
# Import tools from tools.py
from tools import get_current_weather, get_system_time

# using OllamaFunctions from experimental because it supports function binding with llms
model = ChatOllama(
    base_url="http://ai.mtcl.lan:11436",
    model="qwen2.5-coder:32b", # qwen2.5-coder:32b llama3.1:70b gemma2:27b-instruct-q8_0
    format="json"
    )

model_with_tools = model.bind_tools(
    tools=[get_current_weather, get_system_time],
)

tool_mapping = {
    'get_current_weather': get_current_weather,
    'get_system_time': get_system_time,
}

# Define the system message
system_message = SystemMessagePromptTemplate.from_template(
    "You are an intelligent assistant designed to analyze user requests accurately. "
    "You must:"
    "- Always analyze the user's request to understand its intent."
    "- Only use available tools when the request explicitly requires external information or actions you cannot perform directly."
    "- Avoid using tools for general questions or tasks you can handle without external assistance (e.g., answering general knowledge questions, casual conversations, or creative requests)."
    "- When using a tool, ensure it is relevant to the request and provide the necessary arguments accurately."
    "- Do not invoke a tool if it is not listed in your available tools."
    "- Your available tools are:"
    "  - `get_current_weather`: Provides the current weather information for a specified location."
    "  - `get_system_time`: Provides the current system time."
    "- If no tools are needed or the requested tool is unavailable, respond directly to the user's request without invoking any tools."
)

user_message = HumanMessagePromptTemplate.from_template(
    "Conduct a comprehensive analysis of the request provided."
    "USER REQUEST:{initial_request}"
)

# Define Agent Prompt template for llama3
agent_request_generator_prompt = ChatPromptTemplate.from_messages([system_message, user_message])

agent_request_generator = agent_request_generator_prompt | model_with_tools
# result = agent_request_generator.invoke({"initial_request": "What is the weather in woodbury in MN?"})
# print(result)
# input("...")

# Pydantic Schema for structured response
class Evaluation(BaseModel):
    result: bool = Field(description="True or False", required=True)


category_system_message = SystemMessagePromptTemplate.from_template(
    "You are an intelligent assistant designed to analyze user requests accurately. "
    "You must:"
    "- Always analyze the user's request to understand its intent."
    "- Only use available tools when the request explicitly requires external information or actions you cannot perform directly."
    "- Avoid using tools for general questions or tasks you can handle without external assistance (e.g., answering general knowledge questions, casual conversations, or creative requests)."
    "- When using a tool, ensure it is relevant to the request and provide the necessary arguments accurately."
    "- Do not invoke a tool if it is not listed in your available tools."
    "- Your available tools are:"
    "  - `get_current_weather`: Provides the current weather information for a specified location."
    "  - `get_system_time`: Provides the current system time."
    "- If no tools are needed or the requested tool is unavailable, respond directly to the user's request without invoking any tools."
)
category_system_message = SystemMessagePromptTemplate.from_template(
    "You are a Smart Router Agent. "
    "You are a master at reviewing whether the original question that customer asked was answered in the tool response. "
    "You understand the context and question below and return your answer in JSON."
)

category_user_message = HumanMessagePromptTemplate.from_template(
    "CONTEXT: Conduct a comprehensive analysis of the Initial Request from user and Tool Response and route the request into boolean true or false: "
    "    True - used when INITIAL REQUEST appears to be answered by TOOL RESPONSE. "
    "    False - used when INITIAL REQUEST is not answered by TOOL RESPONSE or when TOOL RESPONSE is empty. "
    "        Output either True or False "
    "        eg: "
    "        'True' "
    " INITIAL REQUEST: {research_question}"
    " TOOL RESPONSE:{tool_response}"
)

# Define Agent Prompt template
category_generator_prompt = ChatPromptTemplate.from_messages([category_system_message, category_user_message])
structured_llm = model.with_structured_output(Evaluation)
category_generator = category_generator_prompt | structured_llm

# result = category_generator.invoke({"research_question": "What is the weather in woodbury in MN?", "tool_response":"65C, Sunny"})
# print(result)
# input("...")


class AgentState(TypedDict):
    research_question: str
    tool_response: str
    agent_response: AIMessage
    agent_call_count: int = 0
    tool_call_count: int = 0

  
def agent(state: AgentState):
    print(colored("STATE at agent start:", "magenta"), colored(state, "cyan"))
    input("Paused ... Hit Enter to Execute Agent Logic...")
    last_ai_message = agent_request_generator.invoke({"initial_request": state["research_question"]})
    state["agent_call_count"] += 1
    #append the response to the agent_response list in the state
    if last_ai_message is not None:
        state["agent_response"] = last_ai_message 
        if last_ai_message.content is not None and last_ai_message.content != "" :
            state["tool_response"]=last_ai_message.content
    print(colored("STATE at agent end:", "magenta"), colored(state, "cyan"))
    input("Paused Hit Enter to go to Should Continue Logic...")    
    return state
   
    
def should_continue(state: AgentState):
    print(colored("STATE at should_continue start:", "magenta"), colored(state, "cyan"))
    input("Paused at Should Continue Start")
    print(colored("Evaluating whether the Question is Answered by the tool response or not... Please wait...", "red"))
    result = category_generator.invoke({"research_question": state["research_question"],
                                         "tool_response":state["tool_response"]
                                        })
    if isinstance(result, Evaluation):
        # Access the 'result' attribute from Evaluation
        print(colored("Is tool response good and should the flow go to END node? ", "cyan"), colored(result.result, "yellow"))
        input("Paused at Should Continue Mid")
        
        if result.result:  # If result is True
            print(colored("Return end", "red"))
            return "end"
        else:  # If result is False
            print(colored("Return continue", "green"))
            return "continue"
    else:
        print("Result is not an Evaluation instance, returning 'end' as default.")
        return "end"

def call_tool(state: AgentState):
    print(colored("STATE at call_tool start:", "magenta"), colored(state, "cyan"))
    input("Paused at call_tool Start")
    agent_response = state["agent_response"]
    
    if hasattr(agent_response, 'tool_calls') and len(agent_response.tool_calls) > 0:
        tool_call = agent_response.tool_calls[0]
        tool = tool_mapping[tool_call["name"].lower()]
        try:
            tool_output = tool.invoke(tool_call["args"])
            state["tool_call_count"] += 1
            print(colored("Tool output:", "magenta"), colored(tool_output, "green"))
            if tool_output is not None:
                state["tool_response"] = tool_output
        except Exception as e:
            print(f"Error invoking tool: {e}")
            # Handle the error or log it as needed
    else:
        print("No tool calls found in agent response.")
    print(colored("STATE at call_tool end:", "magenta"), colored(state, "cyan"))
    input("Paused at call_tool End")
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


# research_question = "What's the current system time?"
research_question = "Tell me a joke?"
# research_question = "How is the weather in Woodbury MN today?"
# research_question = "What is the cause of earthquakes?"



state : AgentState = {"research_question": research_question,
                      "tool_response": [] ,
                      "agent_response": [],
                      "agent_call_count": 0,
                      "tool_call_count": 0
                      }
result = app.invoke(state)
print("\n")
print(colored("FINAL STATE at end:", "magenta"), colored(result, "cyan"))

print(colored("FINAL RESPONSE at end:", "magenta"), colored(result["tool_response"], "cyan"))