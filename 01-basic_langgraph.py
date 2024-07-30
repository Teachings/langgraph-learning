from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

def add_text(input: list[HumanMessage]):
    input[0].content += " Amazing_"
    return input

# Create the MessageGraph
graph = MessageGraph()

node1_id = "node1"
node2_id = "node2"
node3_id = "node3"
node4_id = "node4"

# Adding nodes with their respective functions
graph.add_node(node1_id, add_text)
graph.add_node(node2_id, add_text)
graph.add_node(node3_id, add_text)
graph.add_node(node4_id, add_text)

# Adding edges between nodes
graph.add_edge(node1_id, node2_id)
graph.add_edge(node1_id, node3_id)
graph.add_edge(node2_id, node4_id)
graph.add_edge(node3_id, node4_id)
graph.add_edge(node4_id, END)

# Set the entry point of the graph
graph.set_entry_point(node1_id)

# Compile the graph to make it runnable
runnable_graph = graph.compile()


def save_graph_to_file(runnable_graph, output_file_path):
    png_bytes = runnable_graph.get_graph().draw_mermaid_png()
    with open(output_file_path, 'wb') as file:
        file.write(png_bytes)

save_graph_to_file(runnable_graph, "output-01.png")

print(runnable_graph.invoke("AI is "))