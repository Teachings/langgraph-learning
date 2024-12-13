from enum import Enum
from pydantic import BaseModel, Field
from termcolor import colored
from ollama import Client
from tools import get_current_weather, get_system_time

# Configuration: API Endpoint
API_URL = "http://ai.mtcl.lan:11434"

# Tools and Tool Functions
TOOLS = {
    "weather": get_current_weather.tool_definition,
    "get_system_time": get_system_time.tool_definition
}

tool_functions = {
    "get_current_weather": get_current_weather,
    "get_system_time": get_system_time
}

# Client Initialization
client = Client(host=API_URL)

# Enums for clarity
class AgentType(str, Enum):
    BASIC = "basic"
    SPECIALIZED = "specialized"

class TaskType(str, Enum):
    WEATHER = "weather"
    SYSTEM_TIME = "get_system_time"
    GENERIC = "generic"

# Router Decision Schema
class RouterDecision(BaseModel):
    agent_type: AgentType
    task_type: TaskType

# BasicAgent: Handles generic queries
class BasicAgent:
    def handle_query(self, query: str) -> str:
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a BasicAgent. Answer the query directly."},
                {"role": "user", "content": query}
            ],
            model="qwen2.5-coder:32b",
        )
        print(colored(f"BasicAgent Response: {response.message.content}\n", "cyan"))
        return response.message.content

# SpecializedAgent: Dynamically assigns and invokes tools
class SpecializedAgent:
    def handle_query(self, query: str, task_type: TaskType) -> str:
        tool = TOOLS.get(task_type.value)
        if not tool:
            print(colored(f"No tool available for task type '{task_type}'. Routing to BasicAgent.\n", "red"))
            return BasicAgent().handle_query(query)

        print(colored(f"SpecializedAgent handling query with tool: {task_type}\n", "magenta"))

        # AI Response with the tool
        response = client.chat(
            messages=[
                {"role": "system", "content": f"You have access to the {task_type} tool. Use it judiciously."},
                {"role": "user", "content": query}
            ],
            model="qwen2.5-coder:32b",
            tools=[tool],
            format="json"
        )

        # Process tool calls
        if response.message.tool_calls:
            print(colored("\nProcessing Tool Calls...", "magenta", attrs=['bold']))
            for tool_call in response.message.tool_calls:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                tool_function = tool_functions.get(function_name)

                try:
                    result = tool_function(**arguments)
                    print(colored(f"Tool: {function_name}", "yellow", attrs=['bold']))
                    print(colored(f"Arguments: {arguments}", "white"))
                    print(colored(f"Result: {result}", "green"))
                    response.message.content += f"\nTool Result ({function_name}): {result}"
                except Exception as e:
                    print(colored(f"Error executing tool '{function_name}': {e}", "red"))

        return response.message.content

# Router: Decides whether a Basic or Specialized Agent is needed
class Router:
    def __init__(self):
        self.basic_agent = BasicAgent()
        self.specialized_agent = SpecializedAgent()

    def query_ollama_for_decision(self, query: str) -> RouterDecision:
        """Interact with Ollama to determine which agent and task type to route the query to."""
        available_task_types = ", ".join([f"'{key}'" for key in TOOLS.keys()] + ["'generic'"])
        ollama_prompt = f"""
        You are a routing decision engine. Your task is to decide:
        - Which agent (basic or specialized) should handle the query.
        - Which task type (weather, get_system_time, or generic) applies to the query.

        Only the following task types are supported: {available_task_types}.
        If the query pertains to an unsupported task type, route it to the basic agent with task_type 'generic'.

        Examples:
        Query: "What is the weather outside?"
        Response: {{
            "agent_type": "specialized",
            "task_type": "weather"
        }}

        Query: "What is the time right now?"
        Response: {{
            "agent_type": "specialized",
            "task_type": "get_system_time"
        }}

        Query: "Tell me a joke."
        Response: {{
            "agent_type": "basic",
            "task_type": "generic"
        }}

        Query: "What is 2+2?"
        Response: {{
            "agent_type": "basic",
            "task_type": "generic"
        }}

        Now, based on the user's query, provide your decision:
        Query: {query}

        Respond in JSON format:
        {{
            "agent_type": "<basic|specialized>",
            "task_type": "<weather|get_system_time|generic>"
        }}
        """
        response = client.chat(
            messages=[
                {"role": "system", "content": "You are a routing decision engine."},
                {"role": "user", "content": ollama_prompt}
            ],
            model="qwen2.5-coder:32b",
            format=RouterDecision.model_json_schema()
        )
        return RouterDecision.model_validate_json(response.message.content)

    def route(self, query: str) -> str:
        """Route the query to the appropriate agent based on the decision."""
        print(colored("Querying Ollama for routing decision...\n", "yellow"))
        decision = self.query_ollama_for_decision(query)
        print(colored(f"Router Decision: {decision.model_dump_json(indent=2)}\n", "blue"))

        # Route to the appropriate agent
        if decision.agent_type == AgentType.BASIC:
            return self.basic_agent.handle_query(query)
        elif decision.agent_type == AgentType.SPECIALIZED:
            return self.specialized_agent.handle_query(query, decision.task_type)
        else:
            raise ValueError("Invalid routing decision.")

# Demo: Showcase the routing and tool usage flow
def run_demo():
    router = Router()

    queries = [
        "Tell me a joke about programming.",
        "What is the current temperature and forecast for tomorrow in Woodbury, MN?",
        "Turn on the living room lights and adjust them to 50% brightness.",
        "What is the exact time in New York City right now?",
        "Solve this math problem: What is 2 + 2 multiplied by 3?",
        "What are the weather conditions in San Francisco this weekend?",
        "Can you turn off all the lights in the house?",
        "What is the square root of 144?",
        "Give me a detailed weather report for Paris, France.",
        "Play some relaxing music in the bedroom.",
        "What is the capital of France?",
        "How long would it take to fly from New York to Tokyo?",
        "Set an alarm for 6:30 AM tomorrow.",
        "Tell me an interesting fact about space.",
        "What's the current system time on this device?"
    ]

    for query in queries:
        print(colored(f"\nProcessing Query: {query}\n", "blue", attrs=['bold']))
        response = router.route(query)
        # print(colored(f"Final Response:\n{response}\n", "green"))
        input(colored("Press Enter to continue...", "yellow"))

# Entry Point
if __name__ == "__main__":
    run_demo()
