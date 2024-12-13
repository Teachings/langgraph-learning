from enum import Enum
from pydantic import BaseModel, Field
from termcolor import colored
from ollama import Client

# Configuration: API Endpoint
API_URL = "http://ai.mtcl.lan:11434"
client = Client(host=API_URL)

# Enums for clarity
class AgentType(str, Enum):
    BASIC = "basic"  # BasicAgent for simple tasks
    SPECIALIZED = "specialized"  # SpecializedAgent for tool-requiring tasks

class TaskType(str, Enum):
    WEATHER = "weather"
    HOME_AUTOMATION = "home_automation"
    GENERIC = "generic"

# Router decision schema
class RouterDecision(BaseModel):
    agent_type: AgentType = Field(description="Determines which agent to use")
    task_type: TaskType = Field(description="Task type determined by the query")

# Router: Determines which agent to call and task type
class Router:
    def query_ollama_for_decision(self, query: str) -> RouterDecision:
        """Interact with Ollama to determine which agent and task type to route the query to."""
        ollama_prompt = f"""
        You are a routing decision engine. Your task is to decide the type of agent and task for the given user query. Follow these rules:

        1. If the query can be answered without tools and pertains to general information, classify it as:
        - agent_type: "basic"
        - task_type: "generic"

        2. If the query pertains to specific domains such as weather or home automation classify it as:
        - agent_type: "specialized"
        - task_type: "<domain-specific-task>"

        3. A specialized agent cannot handle generic queries, and a basic agent cannot handle domain-specific queries.

        Use the following examples as guidance:

        ### Examples:
        Query: "What is the weather outside?"
        Response: {{
            "agent_type": "specialized",
            "task_type": "weather"
        }}

        Query: "Turn on the lights in the living room."
        Response: {{
            "agent_type": "specialized",
            "task_type": "home_automation"
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

        Query: "Is it going to rain today?"
        Response: {{
            "agent_type": "specialized",
            "task_type": "weather"
        }}

        Now, based on the user's query below, provide your decision:

        Query: {query}

        Respond in the following JSON format:
        {{
            "agent_type": "<basic|specialized>",
            "task_type": "<weather|home_automation|generic>"
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
        """Determine which agent and task type should handle the query."""
        print(colored("Querying Ollama for routing decision...\n", "yellow"))
        decision = self.query_ollama_for_decision(query)
        print(colored(f"Router Decision: {decision.model_dump_json(indent=2)}\n", "blue"))
        return decision

# Demo: Show the routing decisions for various queries
def run_demo():
    router = Router()
    queries = [
        "What is the weather outside?",
        "Turn on the living room lights.",
        "Hey, tell me a joke.",
        "This is a simple question.",
        "How many days are there in a year?",
        "Dim the bedroom lights.",
        "Will it snow tomorrow?",
        "What's the temperature in New York?",
        "What is the square root of 16?",
        "Play some music in the living room.",
        "Tell me something interesting."
    ]

    for query in queries:
        print(colored(f"\nProcessing Query: {query}\n", "blue", attrs=['bold']))
        decision = router.route(query)
        input(colored(f"\nPress Enter to continue...\n", "yellow"))

# Entry Point
if __name__ == "__main__":
    run_demo()
