import gymnasium
import torch
from gymnasium import spaces
from rl_dispatch.data_handle.graph import create_operations_graphs
from rl_dispatch.data_handle.data_handle import load_data_file, read_objective
from rl_dispatch.environment.features import build_super_graph, get_operation_edges, get_resource_edges
import json 
import time

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class DisplibEnvV1(gymnasium.Env):
    def __init__(self, problem_data:dict, device:str,verbose:bool=True):
        super(DisplibEnvV1, self).__init__()
        
        self.device = device
        self.verbose = verbose

        self.op_graphs = create_operations_graphs(problem_data)
        self.op_objs = read_objective(problem_data)

        self.state_graph, self.entry_nodes, self.exit_nodes = build_super_graph(self.op_graphs, self.op_objs, self.device)

        self.initial_state_graph = self.state_graph.clone()

        self.operation_edges = get_operation_edges(self.state_graph)
        self.resource_edges = get_resource_edges(self.state_graph)

        self.num_nodes = self.state_graph.num_nodes
        self.num_node_features = self.state_graph.x.shape[1]
        self.num_trains = len(self.op_graphs)
        
        # Two possible actions: 0 = progress time step, 1 = choose the operation node, 2 = skip this node 
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, seed:int=None):
        super().reset(seed=seed)

        self.state_graph = self.initial_state_graph.clone()

        self.time = 0

        self.nodes_can_pick = torch.zeros(self.num_nodes, dtype=torch.int, device=self.device)
        self.nodes_currently_picked = torch.zeros(self.num_nodes, dtype=torch.int, device=self.device)
        self.nodes_picked = torch.zeros(self.num_nodes, dtype=torch.int, device=self.device)
        self.nodes_not_yet_picked = torch.zeros(self.num_nodes, dtype=torch.int, device=self.device)
        self.nodes_cant_pick = torch.zeros(self.num_nodes, dtype=torch.int, device=self.device)

        self.operation_times = torch.ones((self.num_nodes,2), dtype=torch.int, device=self.device) * -1

        self.feasible_nodes = torch.zeros(self.num_nodes, dtype=torch.int, device=self.device)
        self.current_feasible_node_idx = 0

        self.train_first_free_times = torch.zeros(self.num_trains, dtype=torch.int, device=self.device)
        self.train_no_return_times = torch.zeros(self.num_trains, dtype=torch.int, device=self.device)
        
        self.done = False

        self.solution_dict = {}

        self._initialize_nodes()

        self.log_state("Environment reset.")
        return self.state_graph, {}
    
    def log_state(self, message=""):
        if self.verbose == True:
            logger.info(message)
            logger.info(f"Time: {self.time}")
            logger.info(f"Feasible Nodes: {self.feasible_nodes}")
            logger.info(f"Current Feasible Node Index: {self.current_feasible_node_idx}")
            logger.info(f"Nodes Can Pick: {self.nodes_can_pick}")
            logger.info(f"Nodes Currently Picked: {self.nodes_currently_picked}")
            logger.info(f"Nodes Picked: {self.nodes_picked}")
            logger.info(f"Nodes Not Picked Yet: {self.nodes_not_yet_picked}")
            logger.info(f"Nodes Cant Pick: {self.nodes_cant_pick}")
            logger.info(f"Final Nodes: {self.exit_nodes}")
            logger.info(f"Train Free Times: {self.train_first_free_times}")
            logger.info(f"Train No Return Times: {self.train_no_return_times}")
            logger.info(f"Operation Times: {self.operation_times}")
            logger.info(f"Action Space: {self.action_space}")
            logger.info("="*50)

    
    def _initialize_nodes(self):
        predecessors = self.operation_edges[1]
        idx = torch.tensor([i for i in range(self.num_nodes) if i not in predecessors], device=self.device)
        self.nodes_can_pick[idx] = 1
        self.nodes_not_yet_picked[idx] = 0
        self.feasible_nodes = self.nodes_can_pick.clone()


    
    def _update_picked_node(self, picked_node:int):
        """Updates the picked sets after picking a node."""
        self._reset_node_status(picked_node)
        self.nodes_currently_picked[picked_node] = 1
        self.operation_times[picked_node,0] = self.time
        self.state_graph.x[picked_node, -2] = self.time
        
        successors = self.operation_edges[1][self.operation_edges[0] == picked_node]
        for successor in successors:
            # Any successor of the picked node can now be picked
            self._reset_node_status(successor)
            self.nodes_can_pick[successor] = 1

            predecessors = {int(node.item()) for node in self.operation_edges[0][self.operation_edges[1] == successor]} # ON CPU
            for pred in predecessors:
                if pred != picked_node and pred not in successors:
                    self._reset_node_status(pred)
                    self.nodes_cant_pick[pred] = 1     # Cant pick any predecessors that are not also successors
       
    

            
        # Determine the chosen predecessors of the chosen operation
        chosen_predecessors = { # ON CPU
        node.item() for node in self.operation_edges[0][self.operation_edges[1] == picked_node]
        if self.nodes_currently_picked[node.item()]
        }


        for pred in chosen_predecessors: 
            successors_of_pred = {int(node.item()) for node in self.operation_edges[1][self.operation_edges[0] == pred]}

            for successor in successors_of_pred:
                if successor != picked_node:
                    self._reset_node_status(successor)
                    self.nodes_cant_pick[successor] = 1

            self._reset_node_status(pred)
            self.nodes_picked[pred] = 1
            self.operation_times[pred,1] = self.time
            self.state_graph.x[pred, -1] = self.time

        self._remove_unreachable_nodes() 
        # Log after updating picked node status
        self.log_state("_update_picked_node completed.")

        #TODO: Implement logic to change the graph features

    def _remove_unreachable_nodes(self):
        """
        Optimized BFS to find reachable nodes using edge indices.
        No sparse adjacency matrix, fully GPU-accelerated.
        """
        # Get operation edges (src -> dst)
        src, dst = self.operation_edges.to(self.device)

        # Total number of nodes (assuming max node index)
        num_nodes = max(src.max(), dst.max()) + 1

        # BFS Initialization: Start from nodes_can_pick
        queue = torch.nonzero(self.nodes_can_pick, as_tuple=True)[0]  # Start nodes
        reachable = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)

        while queue.numel() > 0:
            reachable[queue] = True  # Mark visited
            
            # Find neighbors: all destinations where src is in the queue
            mask = torch.isin(src, queue)  # Find edges where src is in queue
            neighbors = dst[mask]  # Get corresponding destinations
            
            # Remove already visited nodes
            queue = neighbors[~reachable[neighbors]]

        # Identify "dead" nodes
        dead_nodes = torch.nonzero(self.nodes_not_yet_picked & ~reachable, as_tuple=True)[0]

        # Mark dead nodes as can't pick
        self.nodes_cant_pick[dead_nodes] = 1
        for dn in dead_nodes.tolist():  # Only use .tolist() where necessary
            self._reset_node_status(dn)



    def _reset_node_status(self, node: int):
            self.nodes_can_pick[node] = 0
            self.nodes_currently_picked[node] = 0
            self.nodes_picked[node] = 0
            self.nodes_not_yet_picked[node] = 0
            self.nodes_cant_pick[node] = 0
            self.feasible_nodes[node] = 0


    def _update_train_free_times(self, picked_node:int):
        train_id = int(self.state_graph.x[picked_node][0].item())
        op_min_dur = self.state_graph.x[picked_node][4].item()
        

        successors = self.operation_edges[1][self.operation_edges[0] == picked_node]
        lowest_succ_lb = 10E6
        max_succ_ub = 0  
        for successor in successors:
            lowest_succ_lb = min(self.state_graph.x[successor][2].item(), lowest_succ_lb)
            max_succ_ub = max(self.state_graph.x[successor][3].item(), max_succ_ub)
        
        self.train_first_free_times[train_id] = max(op_min_dur + self.time,lowest_succ_lb)
        self.train_no_return_times[train_id] = max_succ_ub


    def _check_resource_conflicts(self, node:int):
        resource_edges = self.resource_edges

        # Convert lists to tensors for efficient processing
        picked_currently = torch.nonzero(self.nodes_currently_picked, as_tuple=True)[0]
        picked_already = torch.nonzero(self.nodes_picked, as_tuple=True)[0]

        # Check if any of the currently picked nodes share a resource edge
        PassedResourceCurrentlyUsed = True
        node_current_edges = self.resource_edges[0] == node
        for edge in torch.nonzero(node_current_edges, as_tuple=True)[0]:
            if resource_edges[1][edge] in picked_currently:
                PassedResourceCurrentlyUsed = False
                break

        # Go through each picked node and if it shares a resource edge with the node and has not passed its release time then set the flag to false
        PassedResourceFreedUp = True
        node_picked_edges = self.resource_edges[0] == node
        for edge in torch.nonzero(node_picked_edges, as_tuple=True)[0]:
            if resource_edges[1][edge] in picked_already:
                picked_node_idx = torch.nonzero(self.resource_edges[1][edge] == picked_already, as_tuple=True)[0]
                release_time = self.operation_times[picked_already[picked_node_idx].item()][1] + self.state_graph.edge_attr[edge][2].item()
                if self.time < release_time:
                    PassedResourceFreedUp = False
                    break


        return PassedResourceCurrentlyUsed and PassedResourceFreedUp
    
    def _update_feasible_nodes(self):
        self.feasible_nodes.zero_()

        for node in torch.nonzero(self.nodes_can_pick, as_tuple=True)[0]: 
            train_id = int(self.state_graph.x[node][0].item())
            lower_bound = self.state_graph.x[node][2].item()
            upper_bound = self.state_graph.x[node][3].item()
            PassedMinDuration = self.time >= self.train_first_free_times[train_id]
            PassedLowerBound = self.time >= lower_bound
            PassedUpperBound = self.time <= upper_bound
            PassedResourceConflicts = self._check_resource_conflicts(node)
            if self.verbose:
                if not PassedMinDuration:
                    print(f"Node {node} failed PassedMinDuration check.")
                if not PassedLowerBound:
                    print(f"Node {node} failed PassedLowerBound check.")
                if not PassedUpperBound:
                    print(f"Node {node} failed PassedUpperBound check.")
                if not PassedResourceConflicts:
                    print(f"Node {node} failed PassedResourceConflicts check.")
                
            if PassedMinDuration and PassedLowerBound and PassedUpperBound and PassedResourceConflicts:
                self.feasible_nodes[node] = 1


        self.log_state("_update_feasible_nodes completed.")

    
    def _update_state_picked_features(self):
        picked_statuses = torch.zeros(self.num_nodes, 6, device=self.device)
        status_mapping = {
            0: self.nodes_can_pick,
            1: self.nodes_currently_picked,
            2: self.nodes_picked,
            3: self.nodes_not_yet_picked,
            4: self.nodes_cant_pick,
            5: self.feasible_nodes,
        }

        for status, nodes in status_mapping.items():
            picked_statuses[:, status] = nodes.int()

        self.state_graph.x[:, 5:11] = picked_statuses

    def _update_state_graph_features(self):
        self.state_graph.y[0] = self.time
        if self.feasible_nodes.any():
            self.state_graph.y[1] = torch.nonzero(self.feasible_nodes, as_tuple=True)[0][self.current_feasible_node_idx]
        else:
            self.state_graph.y[1] = -1

    
    def step(self, action):
        start_time = time.time()
        if self.done:
            raise RuntimeError("Environment is done. Reset it to start again.")

        reward = -1 

        if action == 0: # Move time one step
            if self.time == self.train_no_return_times.min():
                self.log_state("Invalid action: Upper bound violation; must pick a node")
            else:
                self.time += 1
        elif action == 1: # Pick the current feasible node
            if not self.feasible_nodes.any(): # THIS SHOULDNT HAPPEN UNLESS THERE IS A DEADLOCK
                self.log_state("Invalid action: No feasible nodes available.")
            else:
                start_a1_time = time.time()
                picked_node = torch.nonzero(self.feasible_nodes, as_tuple=True)[0][self.current_feasible_node_idx]
                self._update_picked_node(picked_node)
                self._update_train_free_times(picked_node)
                self._update_state_picked_features()
                self.log_state(f"Node {picked_node} picked.")
                # print(f"Pick node time: {(time.time() - start_a1_time):.4f}") 
                if self.feasible_nodes.any():
                    self.current_feasible_node_idx = (self.current_feasible_node_idx + 1) % self.feasible_nodes.sum().item()
                else:
                    self.current_feasible_node_idx = -1
                    self.done = torch.all(self.nodes_currently_picked == self.exit_nodes)

        elif action == 2:
            if self.feasible_nodes.any():
                self.current_feasible_node_idx = (self.current_feasible_node_idx + 1) % self.feasible_nodes.sum().item()

        self._update_feasible_nodes()
        self._update_state_graph_features()
        
        while self.current_feasible_node_idx == -1 and not self.done:
            self.time += 1
            
            self._update_feasible_nodes()
            self._update_state_graph_features()
            
            if self.feasible_nodes.any():
                self.current_feasible_node_idx = 0
            
        
        # Check if the final nodes are picked
        if self.done:
            reward += 1000
            self.solution_dict = self._build_solution_dict()
            
            with open("data/solutions/solution.json", "w") as f:
                json.dump(self.solution_dict, f, indent=2)

            self.log_state("All nodes picked. Environment done.")
        
        return self.state_graph, reward, self.done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            self.log_state("Rendering current state")

    
    def close(self):
        pass

    def _build_solution_dict(self):
        """
        Build a dictionary mapping each node to scheduling info
        """
        events = []
        for node_id in range(self.num_nodes):
            # If this node was started at some point, we have a "start_time"
            if len(self.operation_times[node_id]) > 0:
                start_time = int(self.operation_times[node_id][0].item())  # first time entry is start
                train_id = int(self.state_graph.x[node_id][0].item())  # index 0 => train_id
                operation = int(self.state_graph.x[node_id][1].item())  # index 1 => original op ID
                events.append({
                    "time": start_time,
                    "train": train_id,
                    "operation": operation
                })

        # Sort events by ascending "time"
        events.sort(key=lambda e: e["time"])

        # Placeholder objective_value (set to 0 or a computed score if you have one)
        solution_json = {
            "objective_value": 0, #TODO: Compute the score here
            "events": events
        }
        return solution_json
    






