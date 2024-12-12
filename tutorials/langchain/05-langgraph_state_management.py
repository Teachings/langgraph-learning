from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from typing import TypedDict
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from termcolor import colored
# Import tools from tools.py
from tools import get_current_weather, get_system_time

# using OllamaFunctions from experimental because it supports function binding with llms
model = ChatOllama(
    base_url="http://ai.mtcl.lan:11436",
    model="nemotron", # qwen2.5-coder:32b llama3.1:70b gemma2:27b-instruct-q8_0
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
    """
    You are a knowledgeable and helpful AI assistant. Always strive to provide the best possible response to user requests. If a request requires information or actions beyond your current capabilities, utilize the available tools to fulfill the user's needs. 

    **Available Tools:**
    - `get_current_weather`: Provides current weather information for a specified location.
    - `get_system_time`: Provides the current system time.

    **Examples:**

    **Tool-Based Requests:**
    * **User Request:** "What's the weather like in Tokyo right now?"
    * **Your Response:** Invoke the `get_current_weather` tool with the argument "Tokyo" .

    **Generic Questions:**
    * **User Request:** "Tell me a joke."
    * **Your Response:** Provide a humorous response directly, without invoking any tools.
    * **User Request:** "What is the meaning of life?"
    * **Your Response:** Provide a thoughtful and informative response, drawing on philosophical and existential concepts.

    Remember to use tools judiciously and only when necessary. If a request can be answered directly, do so without invoking any tools.
    """
)

user_message = HumanMessagePromptTemplate.from_template(
    """
    Conduct a comprehensive analysis of the request provided.
    USER REQUEST:{initial_request}
    """
)

# Define Agent Prompt template for llama3
agent_request_generator_prompt = ChatPromptTemplate.from_messages([system_message, user_message])

agent_request_generator = agent_request_generator_prompt | model_with_tools
# result = agent_request_generator.invoke({"initial_request": "What is the system time?"})
# print(result)
# input("...")

# Pydantic Schema for structured response
class Evaluation(BaseModel):
    result: bool = Field(description="True or False", required=True)

category_system_message = SystemMessagePromptTemplate.from_template(
    """
    You are a skilled evaluator. Your task is to assess whether the provided tool response adequately addresses the initial user query. 

    **Consider the following:**
    * **Relevance:** Does the response directly answer the core question?
    * **Comprehensiveness:** Does the response provide a complete and thorough answer?
    * **Accuracy:** Is the information in the response correct and up-to-date?
    * **Clarity:** Is the response easy to understand and free of ambiguity?
    * **Conciseness:** Is the response concise and to the point?

    **Provide an evaluation, including:**
    * A clear **boolean result** (True/False) indicating whether the response is adequate.

    **Examples:**

    **Example 1:**
    * **INITIAL REQUEST:** "What is the capital of France?"
    * **Tool Response:** "Paris is the capital of France."
    * **Your Evaluation:**
        * **TOOL RESPONSE:** True

    **Example 2:**
    * **INITIAL REQUEST:** "Explain quantum computing in simple terms."
    * **Tool Response:** "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition and entanglement, to perform calculations."
    * **Your Evaluation:**
        * **TOOL RESPONSE:** False
    """
)
category_user_message = HumanMessagePromptTemplate.from_template(
    """
    CONTEXT: Conduct a comprehensive analysis of the Initial Request from user and Tool Response and route the request into boolean true or false: 
    INITIAL REQUEST: {research_question}
    TOOL RESPONSE:{tool_response}
    """
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