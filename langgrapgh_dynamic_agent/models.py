from typing import TypedDict
from pydantic import BaseModel, Field

# Define the structure of the review result using Pydantic
class CodeReviewResult(BaseModel):
    result: str = Field(..., description="The result of the code review: 'correct' or 'incorrect'.")
    message: str = Field(..., description="Optional message returned by the review agent.")


# Define the state
class AgentState(TypedDict):
    initial_request: str
    preprocessor_agent_result: str
    generated_code_result: str
    code_extraction_status: str
    extracted_python_code: str
    code_review_result: CodeReviewResult
    code_review_status: str
    final_output: str