from langgraph.graph import StateGraph, END
from termcolor import colored

from agents import AgentState
from agents import agent_preprocessor, agent_code_generation, agent_extract_code, agent_code_review, agent_execute_code_in_docker 
from agents import conditional_should_continue_after_extraction, conditional_should_continue_after_code_review


# Create a StateGraph to model the workflow
workflow = StateGraph(AgentState)

# Add nodes for each step
workflow.add_node("agent_preprocessor", agent_preprocessor)
workflow.add_node("agent_code_generation", agent_code_generation)
workflow.add_node("agent_extract_code", agent_extract_code)
workflow.add_node("agent_code_review", agent_code_review)
workflow.add_node("agent_execute_code_in_docker", agent_execute_code_in_docker)

# Set entry point
workflow.set_entry_point("agent_preprocessor")

# Add edges between nodes
workflow.add_edge("agent_preprocessor", "agent_code_generation")
workflow.add_edge("agent_code_generation", "agent_extract_code")

# Add conditional edges 
workflow.add_conditional_edges(
    "agent_extract_code",
    conditional_should_continue_after_extraction,
    {
        "continue": "agent_code_review",
        "regenerate": "agent_code_generation"
    }
)

workflow.add_conditional_edges(
    "agent_code_review",
    conditional_should_continue_after_code_review,
    {
        "continue": "agent_execute_code_in_docker",
        "regenerate": "agent_code_generation"
    }
)

workflow.add_edge("agent_execute_code_in_docker", END)

# Compile and run the workflow with debug messages
app = workflow.compile()

#helper method to visualize graph
def save_graph_to_file(runnable_graph, output_file_path):
    png_bytes = runnable_graph.get_graph().draw_mermaid_png()
    with open(output_file_path, 'wb') as file:
        file.write(png_bytes)

save_graph_to_file(app, "output.png")


# List of initial requests
initial_requests = [
    "Calculate the factorial of 10.",
    "Generate Python code to calculate the square root of 144.",
    "Create a function that returns the Fibonacci sequence up to the 10th element.",
    "Write Python code to reverse a given string.",
    "Generate Python code to convert Celsius to Fahrenheit.",
    "Create a function that checks if a number is prime.",
    "Write Python code to sort a list of numbers in ascending order.",
    "Create a function that calculates the area of a circle given the radius.",
    "Write Python code to merge two dictionaries.",
    "Create a function to check if a string is a palindrome.",
    "Generate Python code to implement binary search in a sorted array.",
    "Create Python code that solves a quadratic equation using the quadratic formula.",
    "Generate Python code to create a file, add a poem in it and print its contents line by line.",
    "Generate Python code to create a bunch of files with names starting with a_ and m_, and then delete only the files that start with m, and then list the files that start with a and with m separately.",
    "Generate Python code to create a file named 'data.txt' with some sample data, then read the contents of this file into memory and print them to the console.",
    "Generate Python code to simulate the rolling of two dice 100 times and count how often each sum occurs.",
    "Create a function that reads a CSV file and prints the data in tabular format, ensuring proper alignment of columns.",
    "Write Python code to encrypt and decrypt a message using a Caesar cipher with a shift of 3.",
    "Generate Python code to scrape the latest headlines from a news website using the BeautifulSoup library.",
    "Create a Python script that pings a list of websites and logs their response time to a file.",
    "Write Python code to send an email with an attachment using the smtplib and email libraries.",
    "Generate Python code to monitor a directory for file changes, and log every new file that gets added or deleted.",
    "Create a Python function that downloads an image from a URL and saves it to a local directory.",
    "Write Python code to generate a random password with a mix of uppercase, lowercase, digits, and special characters.",
    "Generate Python code to implement a simple HTTP server that serves static HTML files from a given directory.",
    "Create Python code to find all the anagrams of a given word from a list of words.",
    "Write Python code to extract text from a PDF file using the PyPDF2 library.",
    "Create Python code to plot a sine and cosine wave on the same graph using matplotlib.",
    "Write Python code to generate a QR code for a given string using the qrcode library.",
    "Generate Python code to fetch weather data from an API and display the current temperature and humidity.",
    "Create Python code to connect to a PostgreSQL database, query all rows from a specific table, and print the results.",
    "Write Python code to create a recursive function that solves the Towers of Hanoi puzzle.",
    "Generate Python code to list all the installed packages in the current environment using pip."
]

# Iterate over each request
for request in initial_requests:
    initial_state = {
        "initial_request": request,
        "preprocessor_agent_result": "",
        "generated_code_result": "",
        "extracted_python_code": "",
        "code_review_result": "",
        "final_output": ""
    }

    try:
        # Run the workflow and observe the debug outputs
        result = app.invoke(initial_state)
        print(colored("", "white"))  # Adding a newline with white color for separation
        print(colored("FINAL Result:", "magenta"), colored(result["final_output"], "light_yellow"))
    
    except Exception as e:
        # Catch and log the error, then continue with the next request
        print(colored(f"ERROR: Failed to process request: '{request}'", "red"))
        print(colored(f"ERROR DETAILS: {str(e)}", "red"))
    
    # Pause for user input before moving to the next request
    input(colored("\nPress Enter to continue to the next request...\n", "yellow"))

