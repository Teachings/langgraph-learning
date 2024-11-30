from termcolor import colored
import re
import docker
import tempfile
import os
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from models import CodeReviewResult, AgentState
from utils import pretty_print_state_enhanced


# Import the prompt templates from the new file
from prompts import preprocessor_prompt_template, code_generation_prompt_template, code_review_prompt_template

# Define model
model = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2" #deepseek-coder-v2 nemotron qwen2.5-coder:32b llama3.2
)

# Define model
model_json = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2",
    format="json"
)

# Initialize models for preprocessor, code generation, and code review agents
preprocessor_model = model
code_generator_model = model
code_review_model = model_json.with_structured_output(CodeReviewResult)

# Initialize chains for preprocessor, code generation, and code review agents
preprocessor_agent_generator = preprocessor_prompt_template | preprocessor_model
agent_code_generator = code_generation_prompt_template | code_generator_model
code_review_agent_generator = code_review_prompt_template | code_review_model

def agent_preprocessor(state: AgentState):
    print(colored("DEBUG: Preprocessing User Request...", "magenta"))
    result = preprocessor_agent_generator.invoke({"user_request": state["initial_request"]})
    # print(colored(f"DEBUG: Preprocessor Result: {result.content}", "magenta"))
    state["preprocessor_agent_result"] = result.content
    print(colored("DEBUG: agent_preprocessor state", "magenta"))
    pretty_print_state_enhanced(state)
    return state

def agent_code_generation(state: AgentState):
    print(colored("DEBUG: Generating Python Code...", "blue"))
    
    # Check and reset the state at the beginning of the method if needed
    if (state["generated_code_result"] == "regenerate" or 
        state["code_review_result"] == "regenerate"):
        print(colored("DEBUG: Resetting agent state due to regenerate flag...", "yellow"))
        reset_keys = [
            "generated_code_result", 
            "extracted_python_code", 
            "code_review_result", 
            "final_output"
        ]
        for key in reset_keys:
            state[key] = ""
    else:
        print(colored("DEBUG: Initial Generation of code. No need to reset agent state.", "green"))
    
    # Continue with the rest of your code generation logic...
    result = agent_code_generator.invoke({"task": state["preprocessor_agent_result"]})
    # print(colored(f"DEBUG: Code Generation Result: {result.content}", "blue"))
    state["generated_code_result"] = result.content
    
    # Continue with the rest of your code generation logic...
    print(colored("DEBUG: agent_code_generation state", "magenta"))
    pretty_print_state_enhanced(state)
    return state

def agent_extract_code(state: AgentState):
    print(colored("DEBUG: Extracting Python Code...", "magenta"))
    # print(colored(f"DEBUG: Generated Code Result: {state['generated_code_result']}", "green"))

    code_result = state["generated_code_result"]

    code_block = re.search(r"```(?!python)(.*?)```", code_result, re.DOTALL)
    code_block_with_lang = re.search(r"```python(.*?)```", code_result, re.DOTALL)
    single_backtick_code = re.search(r"`(.*?)`", code_result, re.DOTALL)
    
    # 1. Try to extract code from triple backticks without the word 'python'
    if code_block:
        extracted_code = code_block.group(1).strip()
        state["extracted_python_code"] = extracted_code
        # print(colored(f"DEBUG: Extracted Python Code from triple backticks: {state['extracted_python_code']}", "green"))
        print(colored("DEBUG: Extracted Python Code from triple backticks", "green"))
        state["code_extraction_status"] = "continue"
    
    # 2. If that fails, try to extract from triple backticks with 'python'
    elif code_block_with_lang:
        extracted_code = code_block_with_lang.group(1).strip()
        state["extracted_python_code"] = extracted_code
        # print(colored(f"DEBUG: Extracted Python Code from triple backticks with 'python': {state['extracted_python_code']}", "green"))
        print(colored("DEBUG: Extracted Python Code from triple backticks with 'python'", "green"))
    
        state["code_extraction_status"] = "continue"
    
    # 3. If that fails, try to extract from single backticks
    elif single_backtick_code:
        extracted_code = single_backtick_code.group(1).strip()
        state["extracted_python_code"] = extracted_code
        # print(colored(f"DEBUG: Extracted Python Code from single backticks: {state['extracted_python_code']}", "green"))
        print(colored("DEBUG: Extracted Python Code from single backtick", "green"))
        state["code_extraction_status"] = "continue"
    
    # 4. Fallback: Assume the entire result is the code if no backticks are found
    elif code_result:
        print(colored("DEBUG: No backticks found. Assuming entire result is the code.", "yellow"))
        state["extracted_python_code"] = code_result.strip()
        # print(colored(f"DEBUG: Fallback Extracted Python Code: {state['extracted_python_code']}", "yellow"))
        print(colored("DEBUG: Fallback Extraction", "green"))
        state["code_extraction_status"] = "continue"
    else:
        state["code_extraction_status"] = "regenerate"  # Extraction failed, regenerate
    
    print(colored("DEBUG: agent_extract_code state", "magenta"))
    pretty_print_state_enhanced(state)

    return state  # Always return state

def conditional_should_continue_after_extraction(state: AgentState):
    # Check if the extraction was successful and we have some code to work with
    if state["code_extraction_status"] == "continue":
        return "continue"
    else:
        return "regenerate"

def agent_code_review(state: AgentState):
    print(colored("DEBUG: Reviewing Python Code...", "magenta"))
    
    code_review_result = code_review_agent_generator.invoke({"generated_code": state["extracted_python_code"], "initial_request": state["preprocessor_agent_result"]})

    try:
        
        # Print and store in agent state
        # print(colored("Reviewed Code:", "yellow"))
        if isinstance(code_review_result, CodeReviewResult):
            # print(f"Result: {code_review_result.result}")
            # print(f"Message: {code_review_result.message}")
            state["code_review_result"] = code_review_result
        else:
            print("Unexpected response format from code review agent.")
        
        # Update the review status based on the result
        if code_review_result.result == "correct":
            state["code_review_status"] = "continue"
        else:
            state["code_review_status"] = "regenerate"

    except ValidationError as e:
        print(colored(f"ERROR: Code review validation failed with error: {e}", "red"))
        state["code_review_status"] = "regenerate"
    
    except Exception as e:
        print(colored(f"ERROR: Error parsing JSON: {e}", "red"))
        state["code_review_status"] = "regenerate"
    
    print(colored("DEBUG: agent_code_review state", "magenta"))
    pretty_print_state_enhanced(state)


    return state  # Always return state

def conditional_should_continue_after_code_review(state: AgentState):
    # Check if the extraction was successful and we have some code to work with
    if state["code_review_status"] == "continue":
        return "continue"
    else:
        return "regenerate"

def agent_execute_code_in_docker(state: AgentState):
    print(colored("DEBUG: Running code in Docker...", "magenta"))
    # print(colored(f"DEBUG: Final Python Code to run: {state['extracted_python_code']}", "cyan"))

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_code_file:
        temp_code_file.write(state["extracted_python_code"].encode('utf-8'))
        temp_code_filename = temp_code_file.name

    client = docker.from_env()
    try:
        container_output = client.containers.run(
            image="python:3.9-slim",
            command=f"python {os.path.basename(temp_code_filename)}",
            volumes={os.path.dirname(temp_code_filename): {'bind': '/usr/src/app', 'mode': 'rw'}},
            working_dir="/usr/src/app",
            remove=True,
            stdout=True,
            stderr=True
        )
        state["final_output"] = container_output.decode('utf-8')
        # print(colored("DEBUG: Docker Output:", "cyan"), state["final_output"])
    except docker.errors.ContainerError as e:
        print(colored(f"ERROR: Error running code in container: {str(e)}", "red"))
    
    os.remove(temp_code_filename)

    print(colored("DEBUG: agent_execute_code_in_docker state", "magenta"))
    pretty_print_state_enhanced(state)

    return state
