import json
import networkx as nx
import numpy as np
from rl_dispatch.data_handle.data_handle import read_objective, load_data_file

def create_operations_graphs(data:dict) -> list[nx.Graph]:
    """
    Create a list of directed graphs from the specified data dictionary.
    """
  # Create the directed graph
    graphs = []

    # Add nodes and edges from the data
    for train in data["trains"]:
        G = nx.DiGraph()
        for idx, operation in enumerate(train):
            operation['start_lb'] = operation.get('start_lb', 0)
            operation['start_ub'] = operation.get('start_ub', 10E6) #TODO: np.inf will probably mess up numerical stability
            operation['min_duration'] = operation.get('min_duration', 0)
            operation['resources'] = operation.get('resources', [])
            operation['resources'] = [{"resource": resource.get('resource',0),"release_time": resource.get('release_time', 0)} for resource in operation['resources']]
            operation['successors'] = operation.get('successors', [])

            G.add_node(idx, **operation)
            G.nodes[idx]['op'] = idx
            for successor in operation["successors"]:
                G.add_edge(idx, successor)

        graphs.append(G)
    
    return graphs

# WANT TO CREATE A GRAPH OF RESOURCES SO THAT IT CAN BE INPUT AND INFORMATION CAN BE ASCERTAINED FROM IT
def create_resource_graph(data: dict) -> nx.DiGraph:
    """
    Creates a directed graph representing the relationships between resources 
    based on the provided data.
    Args:
        data (dict): A dictionary containing train operation data.
    Returns:
        nx.DiGraph: A directed graph where nodes represent resources and edges 
                    represent relationships between resources within the same 
                    operation and between successive operations.
    """

    
    resource_graph = nx.DiGraph()
    
    for train_data in data["trains"]:
        for op_data in train_data:
            current_resources = {res["resource"] for res in op_data.get("resources", [])}
            
            # Create edges between resources in the same operation
            for res1 in current_resources:
                for res2 in current_resources:
                    if res1 != res2:
                        resource_graph.add_edge(res1, res2)
            
            # Create edges between resources in successive operations
            for successor_idx in op_data.get("successors", []):
                successor_op = train_data[successor_idx]
                successor_resources = {res["resource"] for res in successor_op.get("resources", [])}
                
                for curr_res in current_resources:
                    for succ_res in successor_resources:
                        # if curr_res != succ_res: # Not sure if necessary
                        resource_graph.add_edge(curr_res, succ_res)

    
    
    return resource_graph
