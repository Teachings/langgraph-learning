from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.prompts import PromptTemplate

# Import tools from tools.py
from tools import get_current_weather, get_system_time

# using OllamaFunctions from experimental because it supports function binding with llms
model = OllamaFunctions(
    base_url="http://ai.mtcl.lan:11436",
    model="llama3.1:70b", #llama3-groq-tool-use:70b llama3.1:70b
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

#run tool
for tool_call in result.tool_calls:
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    print(tool_output)