# tools.py
import random
from tool_decorator import custom_tool

@custom_tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a specified location.

    Args:
        location (str): The name of the location for which to check the weather.
        
    Returns:
        str: A string describing the current weather in the specified location, randomly chosen from three possible outcomes.
    """
    return f"Weather conditions in {location} is Sunny. Temperature is 60F."

@custom_tool
def get_system_time(location: str = "Minnesota") -> str:
    """Get the current system time. If no location is provided, use default location as 'Woodbury, Minnesota'.
    
    Args:
        location (str): Optional. The name of the location for which to retrieve the system time. It defaults to Minnesota if not provided.  
        
    Returns:
        str: A string describing the current system time in the specified or default location, randomly chosen from three possible outcomes.
    """
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return "2:00 AM"
    elif random.randint(0, 2) == 1:
        return "3:00 PM"
    else:
        return "6:15 PM"
