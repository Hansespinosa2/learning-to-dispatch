import time
import torch
from rl_dispatch.agent.gnn import GNN
import random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
    
class DoubleDQNAgent:
    def __init__(self, env, device, checkpoint_path=None, training=True):
        self.env = env
        self.device = device
        self.training = training
        self.model = GNN(env.state_graph.num_node_features, env.state_graph.num_edge_features, env.action_space.n).to(self.device)  # Main network
        self.target_model = GNN(env.state_graph.num_node_features, env.state_graph.num_edge_features, env.action_space.n).to(self.device)  # Target network
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.999
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.update_target_frequency = 1000
        self.timestep = 0
        self.losses = []
        self.rewards = []
        self.max_memory_size = 10000

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def remember(self, state, node_idx, action, reward, next_state, done):
        if self.training:
            self.memory.append((state, node_idx, action, reward, next_state, done))
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)  # Remove the oldest memory entry

    def act(self, state, node_idx):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_values = self.model(state, node_idx)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if (not self.training) or (len(self.memory) < batch_size):
            return

        batch = random.sample(self.memory, batch_size)
        states, node_idxs, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists of `Data` objects into `Batch` for efficient processing
        states = Batch.from_data_list(states).to(self.device)
        next_states = Batch.from_data_list(next_states).to(self.device)

        # Convert to tensors
        node_idxs = torch.tensor(node_idxs, dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # Double DQN: Select actions using the current model, evaluate using target
            next_q_values = self.target_model(next_states, node_idxs)
            next_actions = self.model(next_states, node_idxs).argmax(dim=1)
            target_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Compute target values
            targets = rewards + (self.gamma * target_q_values * (~dones))

        # Get Q-values for current states
        q_values = self.model(states, node_idxs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        # Update target network periodically
        if self.timestep % self.update_target_frequency == 0:
            self.update_target_model()

        self.timestep += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'timestep': self.timestep,
            'losses': self.losses,
            'rewards': self.rewards
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.timestep = checkpoint['timestep']
        self.losses = checkpoint['losses']
        self.rewards = checkpoint['rewards']
        self.update_target_model()  # Ensure the target model is updated

