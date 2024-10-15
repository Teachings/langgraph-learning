from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions

# Chain
model = OllamaFunctions(
    base_url="http://ai:11436",
    model="llama3.1:70b", 
    format="json", 
    temperature=0
    )
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

# result = category_generator.invoke({
#     "research_question": "What is 2+2",
#     "tool_response": "4"
#     })

result = category_generator.invoke({
    "research_question": "What is the weather in woodbury in MN?",
    "tool_response":"65F, Sunny"})


print(result.result)