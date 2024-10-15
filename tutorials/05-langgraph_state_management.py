from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from typing import TypedDict
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from termcolor import colored
import json
# Import tools from tools.py
from tools import get_current_weather, get_system_time

# using OllamaFunctions from experimental because it supports function binding with llms
model = OllamaFunctions(
    base_url="http://ai.mtcl.lan:11436",
    model="llama3.1:70b", #llama3.1:70b gemma2:27b-instruct-q8_0
    format="json"
    )

model_with_tools = model.bind_tools(
    tools=[get_current_weather, get_system_time],
)

tool_mapping = {
    'get_current_weather': get_current_weather,
    'get_system_time': get_system_time,
}

# Define Agent Prompt template for llama3
agent_request_generator_prompt = PromptTemplate(
    template=
    """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
        You are a Smart Agent. 
        You are a master at understanding what a customer wants.
        You evaluate every request and utilize available tools if you have to.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Conduct a comprehensive analysis of the request provided\

    USER REQUEST:\n\n {initial_request} \n\n

    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["initial_request"],
)

agent_request_generator = agent_request_generator_prompt | model_with_tools
# result = agent_request_generator.invoke({"initial_request": "What is the weather in woodbury in MN?"})
# print(result)
# input("...")

# Pydantic Schema for structured response
class Evaluation(BaseModel):
    result: bool = Field(description="True or False", required=True)

# Prompt template llama3
category_generator_prompt = PromptTemplate(
    template=
    """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
        You are a Smart Router Agent. You are a master at reviewing whether the original question that customer asked was answered in the tool response.
        You understand the context and question below and return your answer in JSON.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    CONTEXT: Conduct a comprehensive analysis of the Initial Request from user and Tool Response and route the request into boolean true or false:
        True - used when INITIAL REQUEST appears to be answered by TOOL RESPONSE. \
        False - used when INITIAL REQUEST is not answered by TOOL RESPONSE or when TOOL RESPONSE is empty \

            Output either True or False \
            eg:
            'True' \n\n
    INITIAL REQUEST:\n\n {research_question} \n\n
    TOOL RESPONSE:\n\n {tool_response} \n\n

    JSON:
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
 input_variables=["research_question", "tool_response"],
)

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