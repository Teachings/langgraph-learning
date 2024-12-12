from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

# Create the MessageGraph
graph = MessageGraph()

node1_id = "entry"
node2_id = "human"
node3_id = "ai"
node4_id = "finish"

def entry(input: list[HumanMessage]):
    return input

def human(input: list[HumanMessage]):
    input[0].content += " is not Amazing"
    return input

def ai(input: list[HumanMessage]):
    input[0].content += " is Amazing"
    return input

def finish(input: list[HumanMessage]):
    input[0].content += " always!"
    return input

def router_node1_node_2_or_node_3(input: list[HumanMessage]):
    if input[0].content == "human":
        return "human_node"
    else:
        return "ai_node"



# Adding nodes with their respective functions
graph.add_node(node1_id, entry)
graph.add_node(node2_id, human)
graph.add_node(node3_id, ai)
graph.add_node(node4_id, finish)

# Adding edges between nodes
graph.add_conditional_edges(
    node1_id, 
    router_node1_node_2_or_node_3, 
    {"human_node": node2_id, "ai_node" : node3_id}
    )

graph.add_edge(node2_id, node4_id)
graph.add_edge(node3_id, node4_id)
graph.add_edge(node4_id, END)

# Set the entry point of the graph
graph.set_entry_point(node1_id)

# Compile the graph to make it runnable
runnable_graph = graph.compile()

#helper method to visualize graph
def save_graph_to_file(runnable_graph, output_file_path):
    png_bytes = runnable_graph.get_graph().draw_mermaid_png()
    with open(output_file_path, 'wb') as file:
        file.write(png_bytes)

save_graph_to_file(runnable_graph, "output-02.png")

#run the graph with input ai or human
print(runnable_graph.invoke("human"))