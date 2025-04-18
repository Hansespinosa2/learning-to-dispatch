import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

def build_super_graph(op_graphs:list[nx.DiGraph], op_objs:list[dict], device:str)->tuple[Data,set,set]:
        """
            Create a super graph by combining multiple operation graphs and adding node features.
            Args:
                op_graphs (list[nx.DiGraph]): List of operation graphs to be combined.
            Returns:
                Data: A PyTorch Geometric Data object representing the combined super graph.
        """
        super_G = nx.DiGraph()
        node_offset = 0
        graph_id_list = []
        resource_dict = {}
        total_nodes = sum(len(graph.nodes()) for graph in op_graphs)
        entry_nodes = torch.zeros(total_nodes, dtype=torch.int, device=device)
        exit_nodes = torch.zeros(total_nodes, dtype=torch.int, device=device)

        for train_id_, train in enumerate(op_graphs):
            mapping = {node: node + node_offset for node in train.nodes()}
            train_relabelled = nx.relabel_nodes(train, mapping)
            entry_nodes[node_offset] = 1


            for node in train_relabelled.nodes():
                node_data = train_relabelled.nodes[node]

                train_relabelled.nodes[node]["train_id"] = torch.tensor([train_id_], dtype=torch.float)
                train_relabelled.nodes[node]["operation_features"] = build_super_graph_node_op_features(node_data)
                train_relabelled.nodes[node]["picked_features"] = build_super_graph_node_picked_features(node_data)
                train_relabelled.nodes[node]["objective_features"] = build_super_graph_node_objective_features(op_objs, node_data, train_id_)
                train_relabelled.nodes[node]["time_features"] = torch.zeros(2, dtype = torch.float)
                train_relabelled.nodes[node]["padded_features"] = torch.zeros(10, dtype=torch.float)

                graph_id_list.append(train_id_)

                for resource in node_data["resources"]:
                    resource_key = resource["resource"]
                    if resource_key not in resource_dict:
                        resource_dict[resource_key] = []  # Initialize the list if key doesn't exist
                    
                    op_res_dict = {
                        "train": train_id_,
                        "op": node_data['op'] + node_offset,
                        "release_time": resource['release_time']
                        }
                    resource_dict[resource_key].append(op_res_dict)


            super_G = nx.compose(super_G, train_relabelled)
            node_offset += len(train)
            exit_nodes[node_offset -1] = 1 #may need to add this to the features
            

        group_node_attrs = ["train_id", "operation_features", "picked_features", "objective_features", "time_features", "padded_features"]
        data_super = from_networkx(super_G, group_node_attrs=group_node_attrs)

        data_super.edge_attr, data_super.edge_index = build_super_graph_edges(data_super, resource_dict)
        
        data_super.y = build_super_graph_y()
        

        data_super.to(device)
        return data_super, entry_nodes, exit_nodes
    
def build_super_graph_edges(data_super:Data, resource_dict:dict) ->Data:
    # Create edges based on resource dependencies
    resource_edges = []
    resource_edge_attrs = []
    operation_edge_attrs = torch.zeros(data_super.edge_index.shape[1], 3)
    operation_edge_attrs[:, 0] = 1
    
    for resource, operations in resource_dict.items():
        for i in range(len(operations)):
            for j in range(i + 1, len(operations)):
                op1 = operations[i]
                op2 = operations[j]
                if op1['train'] != op2['train']:
                    resource_edges.append((op1['op'], op2['op']))
                    resource_edge_attrs.append([0, 1, op1['release_time']])
                    resource_edges.append((op2['op'], op1['op']))
                    resource_edge_attrs.append([0, 1, op2['release_time']])

    # Convert resource edges to tensor and add to the super graph
    if resource_edges:
        resource_edge_index = torch.tensor(resource_edges, dtype=torch.int).t().contiguous()
        edge_index = torch.cat([data_super.edge_index, resource_edge_index], dim=1)
        resource_edge_attrs = torch.tensor(resource_edge_attrs, dtype=torch.long)
        edge_attrs = torch.cat([operation_edge_attrs, resource_edge_attrs], dim=0)
    else:
        edge_attrs = operation_edge_attrs
        edge_index = data_super.edge_index
    
    return edge_attrs, edge_index

def build_super_graph_y() -> torch.Tensor:
    """
    Compute and return graph-level features.
    Returns:
        torch.Tensor: A tensor containing the graph-level features.
    """
    # Example graph-level features
    time = torch.tensor([0], dtype=torch.float)
    current_feasible_node_idx = torch.tensor([0], dtype=torch.float)

    # Concatenate all graph-level features into a single tensor
    graph_features = torch.cat([time, current_feasible_node_idx],dim=0)
    return graph_features
    


def build_super_graph_node_op_features(node_data:dict)->torch.Tensor:
    """
    Extracts and processes node features from the given node data with a resource embedding.
    Args:
        node_data (dict): A dictionary containing node information, including resources and other attributes.
    Returns:
        torch.Tensor: A tensor containing the concatenated node features.
    """
    op = torch.tensor([node_data.get("op", 0.0)], dtype=torch.float)  
    start_lb = torch.tensor([node_data.get("start_lb", 0.0)], dtype=torch.float)      
    start_ub = torch.tensor([node_data.get("start_ub", 10E6)], dtype=torch.float)
    min_duration = torch.tensor([node_data.get("min_duration", 0.0)], dtype=torch.float)

    
    op_features = torch.cat([op, start_lb, start_ub, min_duration], dim=0)
    return op_features

def build_super_graph_node_picked_features(node_data:dict)->torch.Tensor:
    """
    Generate a tensor representing the picking status of features based on node data.
    Args:
        node_data (dict): A dictionary containing node information. It must have a key "op" 
                            which determines the picking status.
    Returns:
        torch.Tensor: A tensor of shape (4,) where each element represents:
                        - can_pick: 1 if node_data["op"] == 0, else 0
                        - picked: Always 0
                        - not_yet_picked: 1 if node_data["op"] != 0, else 0
                        - cant_pick: Always 0
    """

    can_pick = torch.tensor([1 if node_data["op"] == 0 else 0], dtype=torch.float)
    currently_picked = torch.tensor([0], dtype=torch.float)
    picked = torch.tensor([0], dtype=torch.float)
    not_yet_picked = torch.tensor([1 if node_data["op"] != 0 else 0], dtype=torch.float)
    cant_pick = torch.tensor([0], dtype=torch.float)
    feasible = torch.tensor([1 if node_data["op"] == 0 else 0], dtype=torch.float)

    picked_features = torch.cat([can_pick, currently_picked, picked, not_yet_picked, cant_pick, feasible], dim=0)
    return picked_features

def build_super_graph_node_objective_features(op_objs:list[dict], node_data:dict, train_id:int)->torch.Tensor:
    """
    Computes the objective features for a given node and train.
    Args:
        node_data (dict): A dictionary containing data about the node, including the operation ('op').
        train_id (int): The ID of the train for which the objective features are being computed.
    Returns:
        torch.Tensor: A tensor containing the concatenated objective features, which include:
            - delay_threshold: The delay threshold for the operation and train.
            - haversine_increment: The haversine increment for the operation and train.
            - linear_coeff: The linear coefficient for the operation and train.
    """

    delay_threshold = torch.tensor([0], dtype=torch.float)
    haversine_increment = torch.tensor([0], dtype=torch.float)
    linear_coeff = torch.tensor([0], dtype=torch.float)
    for obj_component in op_objs:
        if obj_component['train'] == train_id & obj_component['operation'] == node_data['op']:
            delay_threshold = torch.tensor([obj_component.get("threshold", 0)])
            haversine_increment = torch.tensor([obj_component.get("increment", 0)])
            linear_coeff = torch.tensor([obj_component.get("coeff", 0)])

    objective_features = torch.cat([delay_threshold, haversine_increment, linear_coeff])
    return objective_features

def get_operation_edges(state_graph:Data)->torch.Tensor:
    """
    Retrieves operation edges from the state graph, filtering out non-operation edges.
    Returns:
        torch.Tensor: A tensor containing the indices of the operation edges.
    """
    op_edges_mask = state_graph.edge_attr[:, 0] == 1
    return state_graph.edge_index[:, op_edges_mask]

def get_resource_edges(state_graph:Data)->torch.Tensor:
    """
    Retrieves resource edges from the state graph, filtering out non-resource edges.
    Returns:
        torch.Tensor: A tensor containing the indices of the resource edges.
    """
    resource_edges_mask = state_graph.edge_attr[:, 1] == 1
    return state_graph.edge_index[:, resource_edges_mask]