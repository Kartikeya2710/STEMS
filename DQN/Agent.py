import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, n_states: int, n_actions: int, fc_dims: list, seed = 1):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()

        prev_dim = n_states
        for fc_dim in fc_dims:
            self.layers.append(nn.Linear(prev_dim, fc_dim))
            prev_dim = fc_dim
        self.layers.append(nn.Linear(prev_dim, n_actions))

    def forward(self, states):
        """
        Return the Q values of all the possible actions for each observation (BATCH_SIZE, n_states)
        """
        x = states
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
    
class DuelingDQN (DQN):
    def __init__(self, n_states: int, n_actions: int, fc_dims: list, seed=1):
        super().__init__(n_states, n_actions, fc_dims, seed)
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()

        prev_dim = n_states
        for fc_dim in fc_dims:
            self.layers.append(nn.Linear(prev_dim, fc_dim))
            self.advantage_layers.append(nn.Linear(prev_dim,fc_dim))
            prev_dim = fc_dim
        self.layers.append(nn.Linear(prev_dim, 1))
        self.advantage_layers.append(nn.Linear(prev_dim,n_actions))
     
    def forward(self, states):
        """
        Return the Q values of all the possible actions for each observation (BATCH_SIZE, n_states)
        """
        x = states
        y = states
        for layer in self.layers:
            x = F.relu(layer(x))
        for layer in self.advantage_layers:
            y = F.relu(layer(y))
        return x + y - y.mean()

class Agent(object):
    """Interacts with and learns form environment."""
    def __init__(self, n_states: int, n_actions: int, fc_dims: list, gamma = 0.95, lr = 1e-4, batch_size = 32, update_every = 100, seed = 1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.seed = torch.manual_seed(seed)
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.update_every = update_every

        self.policy_network = DQN(n_states, n_actions, fc_dims, seed).to(self.device)
        self.target_network = DQN(n_states, n_actions, fc_dims, seed).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def act(self, state, epsilon):
        if random.random() <= epsilon:
            return random.choice(torch.arange(self.n_actions)).item()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_network(state).argmax().item()


    def learn(self, experiences, t_step):
        states, actions, rewards, next_state = experiences
        
        criterion = nn.HuberLoss()
        self.policy_network.train()
        self.target_network.eval()
        predicted_values = self.policy_network(states).gather(1,actions)

        with torch.no_grad():
            next_state_labels = self.target_network(next_state).detach().max(1)[0].unsqueeze(1)
        
        labels = rewards + self.gamma * next_state_labels
        loss = criterion(predicted_values, labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        if t_step % self.update_every == 0:
            self.target_network.train()
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_model(self, path, episode):
        torch.save(self.policy_network.state_dict(), os.path.join(path, f"checkpoint_{episode}.pt"))

    def load_model(self, model_folder, episode, Agent):
        Agent.policy_network.load_state_dict(torch.load(os.path.join(model_folder, f"checkpoint_{episode}.pt")))
        return Agent
    
class DoubleDQNAgent(Agent):
    def act(self, state,epsilon):
        if random.random() <= epsilon:
            return random.choice(torch.arange(self.n_actions)).item()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_network(state).argmax().item()
            
    def learn(self, experiences, t_step):
        states, actions, rewards, next_state = experiences
        
        criterion = nn.HuberLoss()
        self.policy_network.train()
        self.target_network.eval()
        # Current state ki Q values
        predicted_values = self.policy_network(states).gather(1,actions)
        with torch.no_grad():
            next_state_actions = self.target_network(next_state).argmax(1).unsqueeze(1)
            next_state_labels = self.policy_network(next_state).gather(1, next_state_actions)
        labels = rewards + self.gamma * next_state_labels

        loss = criterion(predicted_values, labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        if t_step % self.update_every == 0:
            self.target_network.train()
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_model (self,path,episode):
        torch.save (self.policy_network.state_dict(),os.path.join(path,f"checkpoint_{episode}.pt"))

    def load_model(self, model_folder, episode, Agent):
        Agent.policy_network.load_state_dict(torch.load(os.path.join(model_folder, f"checkpoint_{episode}.pt")))
        return Agent

    
class DuelingDoubleDQNAgent(DoubleDQNAgent):
    def __init__(self, n_states: int, n_actions: int, fc_dims: list, gamma = 0.95, lr = 1e-4, batch_size = 32, update_every = 100, seed = 1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.seed = torch.manual_seed(seed)
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.update_every = update_every

        self.policy_network = DuelingDQN(n_states, n_actions, fc_dims, seed).to(self.device)
        self.target_network = DuelingDQN(n_states, n_actions, fc_dims, seed).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
    
    def save_model (self,path,episode):
        torch.save(self.policy_network.state_dict(),os.path.join(path,f"checkpoint_{episode}.pt"))