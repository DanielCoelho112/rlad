import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optimzer

from utilities.networks import weights_init

class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, fc1_dims, fc2_dims, lr, device, checkpoint_dir, log_sig_min=-20, log_sig_max=2, epsilon=1e-6):
        super(ActorNetwork, self).__init__()
        
        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/actor_network.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/actor_network.pt"
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        
        self.fc1 = nn.Linear(num_inputs, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        
        self.mean_linear = nn.Linear(fc2_dims, 2)
        self.log_std_linear = nn.Linear(fc2_dims, 2)
        
        self.apply(weights_init)
        
        # scale actions using following convention: acc [-1,1], steer [-1,1]
        self.action_scale = torch.tensor([[1.0, 1.0]]) # (max - min)/2.
        self.action_bias = torch.tensor([[0.0, 0.0]]) # (max + min)/2.
        
        self.optimizer = optimzer.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)
        
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick (mean + std * N(0,1)).
        y_t = torch.tanh(x_t) # convert action into [-1,1].
        
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # enforcing action bound. 
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        
        
    
    
# if __name__ == '__main__':
    
#     actor = ActorNetwork(num_inputs=250, fc1_dims=50,fc2_dims=25, lr=0.001, device=torch.device('cuda:0'), checkpoint_dir=f'{os.getenv("HOME")}')
    
#     state = torch.rand(size=(1,250)).to(torch.device('cuda:0'))
    
#     mean, log_std = actor(state)
#     # print(f"mean: {mean}")
#     # print(f"log_std: {log_std}")
    
#     # print('sampling...')
    
#     action, log_prob, mean = actor.sample(state)
#     print(f"action: {action}")
#     print(mean.shape)
    
#     # print(f"log_prob: {log_prob}")
#     print(f"mean: {mean}")
    
#     # actor.save_checkpoint()
#     # actor.load_checkpoint()
    
    
        