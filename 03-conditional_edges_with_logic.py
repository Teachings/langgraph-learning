from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph
from langchain_openai import ChatOpenAI

chatmodel = ChatOpenAI(base_url="http://ai.mtcl.lan:11436/v1", api_key="fake-api-key", model="llama3.1")
# chatmodel = ChatOpenAI(base_url="http://ai.mtcl.lan:8000/v1", api_key="fake-api-key", model="phi3")
# chatmodel = OllamaLLM(base_url="http://ai.mtcl.lan:11436", model="llama3.1")

# Create the MessageGraph
graph = MessageGraph()

joke_call_count = 0

node1_id = "agent"
node2_id = "tell_joke"

def agent(input: list[HumanMessage]):
    return input

def tell_joke(input: list[HumanMessage]):
    global joke_call_count
    joke_call_count += 1
    print("joke_call_count :: " + str(joke_call_count))
    print(chatmodel.invoke(input).content)
    return input


def router_node1_node_2_or_end(input: list[HumanMessage]):
    
    if joke_call_count < 10:
        return "tell_joke_condition"
    else:
        return "end_condition"



# Adding nodes with their respective functions
graph.add_node(node1_id, agent)
graph.add_node(node2_id, tell_joke)

# Adding edges between nodes
graph.add_conditional_edges(
    node1_id, 
    router_node1_node_2_or_end, 
    {"tell_joke_condition": node2_id, "end_condition" : END}
    )

graph.add_edge(node2_id, node1_id)


# Set the entry point of the graph
graph.set_entry_point(node1_id)

# Compile the graph to make it runnable
runnable_graph = graph.compile()


#helper method to visualize graph
def save_graph_to_file(runnable_graph, output_file_path):
    png_bytes = runnable_graph.get_graph().draw_mermaid_png()
    with open(output_file_path, 'wb') as file:
        file.write(png_bytes)

save_graph_to_file(runnable_graph, "output-03.png")

#run the graph with input ai or human
runnable_graph.invoke([HumanMessage(content="tell me a joke about nature! keep it funny!")], {"recursion_limit": 100})