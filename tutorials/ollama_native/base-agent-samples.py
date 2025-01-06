from pydantic import BaseModel, Field
from base_agent import StructuredResponseAgent, ToolCallingAgent

def get_current_weather(location: str) -> str:
    """Fetch the current weather for a given location."""
    return f"Weather conditions in {location} are Sunny. Temperature is 60°F."

class WeatherCallingAgent(ToolCallingAgent):
    def __init__(self, model_name="qwen2.5-coder:32b", api_url="http://dual-ai:11434", debug=False):
        system_prompt = (
            "You are a weather assistant designed to provide current weather information. "
            "Use the 'get_current_weather' tool to fetch weather data for a specified location."
        )
        super().__init__(system_prompt, model_name, api_url, debug=debug)
        
        # Register the get_current_weather tool
        self.register_tool(get_current_weather)

class EvaluatorAgent(StructuredResponseAgent):
    def __init__(self, model_name="qwen2.5-coder:32b", api_url="http://dual-ai:11434", debug=False):
        system_prompt = (
            "You are a skilled evaluator. Assess the generated steps with attention to relevance, clarity, and alignment."
        )
        super().__init__(system_prompt, model_name, api_url, debug=debug)

    def get_pydantic_model(self):
        class StepEvaluation(BaseModel):
            score: float = Field(
                description="A score between 0 and 1 indicating the quality of the steps.", ge=0.0, le=1.0
            )
            feedback: str = Field(description="Detailed feedback on why the score was assigned.")
        return StepEvaluation

    def execute_task(self, refined_problem, previous_steps, generated_steps):
        evaluation_prompt = f"""
        **Refined Problem Statement:** {refined_problem}
        
        **Previous Steps:**
        {previous_steps}
        
        **Generated Steps:**
        {generated_steps}
        
        Evaluate the generated steps based on:
        1. Relevance: How well do the steps address the refined problem statement?
        2. Clarity: Are the steps clear and understandable?
        3. Alignment: Do the steps logically follow from the previous steps?
        
        Provide a score between 0 and 1, where 1 indicates excellent quality and 0 indicates very poor quality.
        Also, provide detailed feedback explaining the score.
        """
        messages = [{"role": "user", "content": evaluation_prompt}]
        return self.run(messages)

if __name__ == "__main__":
    # Enable debug mode for the WeatherCallingAgent
    weather_agent = WeatherCallingAgent(debug=True)
    
    # Test the WeatherCallingAgent
    response = weather_agent.run(
        messages=[{"role": "user", "content": "What's the weather like in London?"}]
    )
    print(response)

    # Enable debug mode for the EvaluatorAgent
    evaluator_agent = EvaluatorAgent(debug=True)
    
    # Test the EvaluatorAgent
    evaluation = evaluator_agent.execute_task(
        refined_problem="Design a Python function to sort a list of integers.",
        previous_steps="Analyzed the requirements and decided to use built-in sorting methods.",
        generated_steps="Implement a function that uses the sorted() function with custom key parameters."
    )
    print(evaluation)
