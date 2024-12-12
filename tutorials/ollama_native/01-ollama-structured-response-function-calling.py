from ollama import chat, Client
from pydantic import BaseModel, Field
from tools import get_current_weather, get_system_time

# Define the OpenAI endpoint and API key
api_url = "http://localhost:11434"

# Collect tool definitions from decorated functions
tools = [get_current_weather.tool_definition, 
         get_system_time.tool_definition]

# Define a dictionary of tool functions
tool_functions = {
    "get_current_weather": get_current_weather,
    "get_system_time": get_system_time
}
client = Client(host=api_url)

def process_request(initial_request, tools):
    system_message = """
    You are a knowledgeable and helpful AI assistant. 
    Always strive to provide the best possible response to user requests. 
    If a request requires information or actions beyond your current capabilities, utilize the available tools to fulfill the user's needs. 

    **Examples:**

    **Tool-Based Requests:**
    * **User Request:** "What's the weather like in Chicago right now?"
    * **Your Response:** Invoke the `get_current_weather` tool with the argument "Chicago, IL".

    **Generic Questions:**
    * **User Request:** "Tell me a joke."
    * **Your Response:** Provide a humorous response directly, without invoking any tools.
    * **User Request:** "What is the meaning of life?"
    * **Your Response:** Provide a thoughtful and informative response, drawing on philosophical and existential concepts.

    Remember to use tools judiciously and only when necessary. If a request can be answered directly, do so without invoking any tools.
    """

    user_message = f"{initial_request}"

    response = client.chat(
        messages=[
            {
                'role': 'system',
                'content': system_message,
            },
            {
                'role': 'user',
                'content': user_message,
            }
        ],
        model='qwen2.5-coder:32b',
        tools=tools,
        format='json'
    )

    print(response.message)

    try:
        if response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                tool_function = tool_functions[function_name]
                result = tool_function(**arguments)
                response.message.content += f"\n{result}"
    except Exception as e:
        print(f"Error executing tool call: {e}")
        # You might want to handle the error differently, e.g., by logging it or notifying the user.

    return response.message.content


# print(process_request("what is the weather in woodbury MN?", tools))
# print(process_request("Tell me a joke", tools))

# input("...")

# Pydantic Schema for structured response
class Evaluation(BaseModel):
    result: bool = Field(description="True or False", required=True)
    explanation: str = Field(description="explanation of the reason why it is true or false.", required=True)


def evaluate_response(initial_request, tool_response):
    category_system_message = f"""
    You are a skilled evaluator. Your task is to assess whether the provided tool response *adequately* addresses the initial user query. Focus on whether the response provides a *fundamentally correct and relevant* answer. Minor omissions or formatting issues should not necessarily lead to a "False" result.

    **Consider the following (in order of importance):**

    1.  **Core Relevance:** Does the response address the central point of the user's request? (Most important)
    2.  **Fundamental Accuracy:** Is the core information provided correct? (Crucial)
    3.  **Clarity:** Is the response understandable? (Important, but minor clarity issues shouldn't automatically fail)
    4.  **Conciseness:** Is the response reasonably concise? (Less important)
    5. **Units:** For temperature, either Fahrenheit (F) or Celsius (C) is acceptable unless the user specifically requests a particular unit.

    **Provide an evaluation, including:**

    *   A clear **boolean result** (True/False) indicating whether the response is *fundamentally adequate*.
    *   A brief **explanation** justifying your evaluation.

    **Examples:**

    # ... (Previous examples remain the same)

    **Example 8 (Acceptable Fahrenheit):**

    *   **INITIAL REQUEST:** "What's the weather in Woodbury, MN?"
    *   **Tool Response:** "Sunny, 78F"
    *   **Your Evaluation:**
        *   **result:** True
        *   **explanation:** The response provides relevant weather information, including the temperature in Fahrenheit, which is acceptable.

    **Example 9 (Acceptable Celsius):**

    *   **INITIAL REQUEST:** "What's the weather in Woodbury, MN?"
    *   **Tool Response:** "Sunny, 26C"
    *   **Your Evaluation:**
        *   **result:** True
        *   **explanation:** The response provides relevant weather information, including the temperature in Celsius, which is acceptable.

    **Example 10 (Explicit Celsius Request):**

        *   **INITIAL REQUEST:** "What's the weather in Woodbury, MN in Celsius?"
        *   **Tool Response:** "Sunny, 78F"
        *   **Your Evaluation:**
            *   **result:** False
            *   **explanation:** The user explicitly requested Celsius, but the response provided Fahrenheit.

    """

    category_user_message = f"""
        CONTEXT: Conduct a comprehensive analysis of the Initial Request from user and Tool Response and route the request into boolean true or false: 
        
        **Initial Query:** {initial_request}
        **Tool Response:** {tool_response}
        """
    print("Tool Response :: ", tool_response)
    response = client.chat(
        messages=[
            {
                'role': 'system',
                'content': category_system_message,
            },
            {
                'role': 'user',
                'content': category_user_message,
            }
        ],
        model='qwen2.5-coder:32b',
        format=Evaluation.model_json_schema(),
    )

    evaluation = Evaluation.model_validate_json(response.message.content)
    return evaluation

initial_request = "Tell me a joke?"
tool_response = process_request(initial_request, tools)

evaluation = evaluate_response(initial_request, tool_response)
print(evaluation)

input("\n hit enter to go to next question \n")

initial_request = "What is the temprature in Chicago, is it sunny cold or rainy??"
tool_response = process_request(initial_request, tools)

evaluation = evaluate_response(initial_request, tool_response)
print(evaluation)