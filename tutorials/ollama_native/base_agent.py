from abc import ABC, abstractmethod
from ollama import Client
import json
import inspect
from pydantic import BaseModel
from termcolor import colored

class BaseLLMAgent(ABC):
    def __init__(self, system_prompt: str, model_name: str, api_url: str = "http://dual-ai:11434", debug: bool = False):
        """Initialize the BaseLLMAgent with common attributes."""
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.client = Client(host=api_url)
        self.debug = debug

    def debug_message(self, message: str, color: str = "cyan"):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(colored(f"[DEBUG] {message}", color))

    def generate_response(self, messages: list[dict], **kwargs) -> dict:
        """Generate a response from the LLM based on the given messages."""
        try:
            self.debug_message(f"Sending messages: {messages}")
            response = self.client.chat(
                messages=messages,
                model=self.model_name,
                **kwargs
            )
            self.debug_message(f"Received response: {response}")
            return response.message
        except Exception as e:
            error_message = f"Error generating response: {e}"
            print(colored(error_message, "red"))
            raise

    @abstractmethod
    def run(self, messages: list[dict]):
        """Run the agent with the provided messages."""
        pass

class StructuredResponseAgent(BaseLLMAgent):
    @abstractmethod
    def get_pydantic_model(self) -> type[BaseModel]:
        """Return the Pydantic model for structured responses."""
        pass

    def parse_response(self, response_text: str) -> BaseModel:
        """Parse the LLM response into a Pydantic model instance."""
        try:
            self.debug_message(f"Parsing response text: {response_text}")
            pydantic_model = self.get_pydantic_model()
            return pydantic_model.model_validate_json(response_text)
        except json.JSONDecodeError as e:
            error_message = f"Failed to parse JSON response: {e}"
            print(colored(error_message, "red"))
            raise
        except ValueError as e:
            error_message = f"Validation error in Pydantic model: {e}"
            print(colored(error_message, "red"))
            raise

    def run(self, messages: list[dict]) -> BaseModel:
        """Run the agent by generating and parsing the LLM response."""
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *messages
        ]
        pydantic_model = self.get_pydantic_model()
        response_text = self.generate_response(full_messages, format=pydantic_model.model_json_schema()).content
        return self.parse_response(response_text)

class ToolCallingAgent(BaseLLMAgent):
    def __init__(self, system_prompt: str, model_name: str, api_url: str = "http://dual-ai:11434", debug: bool = False):
        super().__init__(system_prompt, model_name, api_url, debug)
        self.tools = []
        self.tool_functions = {}

    def custom_tool(self, func):
        """Decorator to define tool definitions for functions."""
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Extract function signature and docstring
        sig = inspect.signature(func)
        docstring = func.__doc__.strip() if func.__doc__ else "No description provided."

        # Extract parameter details
        parameters = {}
        for param_name, param in sig.parameters.items():
            param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "unknown"
            parameters[param_name] = {
                'type': param_type,
                'description': f'The {param_name} parameter',
            }

        # Create the tool definition
        tool_definition = {
            'type': 'function',
            'function': {
                'name': func.__name__,
                'description': docstring,
                'parameters': {
                    'type': 'object',
                    'properties': parameters,
                    'required': list(sig.parameters.keys()),
                },
            },
        }

        wrapper.tool_definition = tool_definition
        return wrapper

    def register_tool(self, func):
        """Register a tool function with the agent."""
        wrapped_func = self.custom_tool(func)
        self.tools.append(wrapped_func.tool_definition)
        self.tool_functions[func.__name__] = wrapped_func
        return wrapped_func

    def execute_tool_calls(self, tool_calls: list) -> dict:
        """Execute the tool calls included in the LLM's response."""
        results = {}
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_function = self.tool_functions.get(function_name)
            if tool_function:
                try:
                    self.debug_message(f"Executing tool '{function_name}' with arguments: {arguments}")
                    result = tool_function(**arguments)
                    results[function_name] = result
                except Exception as e:
                    error_message = f"Error executing tool '{function_name}': {e}"
                    print(colored(error_message, "red"))
                    results[function_name] = str(e)
            else:
                warning_message = f"Tool '{function_name}' not found."
                print(colored(warning_message, "yellow"))
                results[function_name] = "Tool not found."
        return results

    def run(self, messages: list[dict]) -> str:
        """Run the agent by generating a response and executing any tool calls."""
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *messages
        ]
        message = self.generate_response(full_messages, tools=self.tools)

        if message.tool_calls:
            self.debug_message(f"Tool calls found in response: {message.tool_calls}")
            tool_results = self.execute_tool_calls(message.tool_calls)
            for fn_name, result in tool_results.items():
                message.content += f"\nTool Result ({fn_name}): {result}"

        return message.content
