# tool_decorator.py
import inspect
from typing import Callable

def custom_tool(func: Callable) -> Callable:
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
