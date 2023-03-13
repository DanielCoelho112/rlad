import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init

class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, image_latent_size, state_size, fc1_dims, fc2_dims, lr, device, checkpoint_dir, target=False):
        super(CriticNetwork, self).__init__()
        
        self.device = device
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/critic_network_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/critic_network_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/critic_network.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/critic_network.pt"


        n_actions = 2
        
        self.preprocess = nn.Sequential(nn.Linear(num_inputs, image_latent_size),
                                        nn.LayerNorm(image_latent_size), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(state_size + n_actions, fc1_dims),
            nn.ReLU(inplace=True),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(inplace=True),
            nn.Linear(fc2_dims, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(state_size + n_actions, fc1_dims),
            nn.ReLU(inplace=True),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(inplace=True),
            nn.Linear(fc2_dims, 1)
        )
        

        # init network weights.
        self.apply(weights_init)
        
        self.optimizer = optimzer.Adam(self.parameters(), lr=lr)
        self.to(self.device)
    
    def forward(self, state, action):
        state_image = self.preprocess(state['image'])
        state = torch.cat([state_image, state['other']], dim=-1)
        state_action = torch.cat([state, action], dim=-1)
        
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)
        
        return q1, q2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        
    
# if __name__ == '__main__':
    
#     critic = CriticNetwork(num_inputs=30000, latent_size=256, fc1_dims=50,fc2_dims=25, lr=0.001, device=torch.device('cuda:0'), checkpoint_dir=f'')
    
#     x = torch.rand(size=(10,30000)).to(torch.device('cuda:0'))
#     actions = torch.rand(size=(10,2)).to(torch.device('cuda:0'))
    
#     print(critic(x, actions)[0].size())
    
    
        