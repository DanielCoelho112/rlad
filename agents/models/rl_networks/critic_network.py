import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init

class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, fc1_dims, fc2_dims, lr, device, checkpoint_dir, target=False):
        super(CriticNetwork, self).__init__()
        
        self.device = device
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/critic_network_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/critic_network_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/critic_network.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/critic_network.pt"

        n_actions = 2
        self.fc1 = nn.Linear(num_inputs + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.fc4 = nn.Linear(num_inputs + n_actions, fc1_dims)
        self.fc5 = nn.Linear(fc1_dims, fc2_dims)
        self.fc6 = nn.Linear(fc2_dims, 1)       

        # init network weights.
        self.apply(weights_init)
        
        self.optimizer = optimzer.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        
        x1 = F.relu(self.fc1(state_action))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)
        
        x2 = F.relu(self.fc4(state_action))
        x2 = F.relu(self.fc5(x2))
        q2 = self.fc6(x2)
        
        return q1, q2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
        
        
    
    
# if __name__ == '__main__':
    
#     critic = CriticNetwork(num_inputs=100, fc1_dims=50,fc2_dims=25, n_actions=3, lr=0.001, device=torch.device('cuda:0'), checkpoint_dir=f'{os.getenv("HOME")}')
    
#     x = torch.rand(size=(10,100)).to(torch.device('cuda:0'))
#     actions = torch.rand(size=(10,3)).to(torch.device('cuda:0'))
    
#     print(critic(x, actions).size())
    
#     critic.save_checkpoint()
    
        