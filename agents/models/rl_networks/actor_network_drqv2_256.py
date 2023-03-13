import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init, TruncatedNormal

class ActorNetwork(nn.Module):
    def __init__(self, state_size, fc1_dims, fc2_dims, lr, device, checkpoint_dir, target=False):
        super(ActorNetwork, self).__init__()
        
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/actor_network_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/actor_network_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/actor_network.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/actor_network.pt"

        
        self.policy = nn.Sequential(nn.Linear(state_size, fc1_dims),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(fc1_dims, fc2_dims),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(fc2_dims, 2))
      
        

        
        self.apply(weights_init)
        
        # scale actions using following convention: speed [0,1], steer [-1,1]
        self.action_scale = torch.tensor([[0.5, 1.0]]) # (max - min)/2.
        self.action_bias = torch.tensor([[0.5, 0.0]]) # (max + min)/2.
        
        self.optimizer = optimzer.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)
        
        
    def forward(self, state, std):
        
        mu = self.policy(state)
        mu = torch.tanh(mu)
        
        mu = mu * self.action_scale + self.action_bias
        
        std = torch.ones_like(mu) * std # same std for both actions.
        
        return mu, std
        
 
    def sample(self, state, std, clip=None):
        mu, std = self.forward(state, std)
        
        dist = TruncatedNormal(mu, std, device=self.device)
        
        action = dist.sample(clip=clip)
        return action, mu
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
        
        
    
    
# if __name__ == '__main__':
    
#     actor = ActorNetwork(num_inputs=70000, image_latent_size=256, state_size=260, fc1_dims=1024,fc2_dims=1024, lr=0.001, device=torch.device('cuda:0'), checkpoint_dir='')
    
#     state_image = (torch.rand(size=(10,70000))*4).to(torch.device('cuda:0'))
#     other_other = (torch.rand(size=(10,4))*8).to(torch.device('cuda:0'))
    
#     state = {'image':state_image,
#              'other': other_other}
    
#     _, mean = actor.sample(state, std=0.5, clip=0.3)
#     # print(f"mean: {mean}")
    
#     # print('sampling...')
    
#     # action, log_prob, mean = actor.sample(state)
#     # print(f"action: {action}")
#     # print(mean.shape)
    
#     # print(f"log_prob: {log_prob}")
#     print(f"mean: {mean}")
    
#     # actor.save_checkpoint()
#     # actor.load_checkpoint()
    
    
        