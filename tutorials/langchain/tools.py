# tools.py
import random
from langchain_core.tools import tool

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a specified location.
    
    This tool simulates checking the weather by randomly selecting from three possible outcomes: sunny, cold, or rainy. 
    The chance of each outcome is equal (1/3). If the random check fails, it may return an unexpected result to simulate real-world unpredictable conditions.
    
    Args:
        location (str): The name of the location for which to check the weather.
        
    Returns:
        str: A string describing the current weather in the specified location, randomly chosen from three possible outcomes.
    """
    # start a random check for 1/3 of times to simulate a failure
    if random.randint(0, 2) == 0 :
        return "Sunny, 78F"
    elif random.randint(0, 2) == 1:
        return "Cold, 22F"
    else:
        return "Rainy, 60F"

@tool
def get_system_time(location: str = "Minnesota") -> str:
    """Get the current system time. If no location is provided, use default location as 'Woodbury, Minnesota'.
    
    This tool simulates retrieving the system time by randomly selecting from three possible outcomes: morning, afternoon, or evening. 
    The chance of each outcome is equal (1/3). If the random check fails, it may return an unexpected result to simulate real-world unpredictable conditions.
    
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
