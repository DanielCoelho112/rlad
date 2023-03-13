import torch
import torch.nn as nn
import torch.optim as optimizer

from utilities.networks import weights_init

class ContinuePredictor(nn.Module):
    def __init__(self, lr, latent_size, n_actions, device, checkpoint_dir):
        super(ContinuePredictor, self).__init__()

        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/continue.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/continue.pt"
        
        self.encoder = nn.Sequential(nn.Linear(latent_size + n_actions, 256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 2))
        
        self.loss = nn.CrossEntropyLoss()
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr)
        
        self.to(self.device)


    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        out = self.encoder(x)
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))



# if __name__ == '__main__':
#     # enc = WaypointLinearEncoder(lr=0, num_waypoints=10, fc_dims=20, out_dims=32,weight_decay=0, device=torch.device('cuda:0'), target=False, checkpoint_dir=None)

#     # import numpy as np 
#     # a = np.full((1, 20), 0.5, dtype=np.float32)

#     # a_torch = torch.from_numpy(a).to(torch.device('cuda:0'))

#     # print(enc(a_torch).size())
#     # print(get_n_params_network(enc))
    
    # light_state = torch.randint(low=0, high=10, size=(10,4))
    # mask = torch.randint(low=0, high=2, size=(10,)).to(torch.bool)
    # print(mask)
    
    # print(light_state)
    
    # new = light_state[mask]
    
    # print(new)
    
    
#     light_state_copy = light_state.clone()
    
#     print(light_state)
#     print(light_state)

#     idx_on = light_state > 0 
    
#     print(idx_on.nelement())
    
#     # print(idx_on)
#     # summary(enc, (3,2))
    
#     light_state[idx_on] = 1
    
#     print(light_state)
    
#     new_ligh_state = light_state_copy[idx_on].reshape(-1, 1)
    
#     n_tl_on = new_ligh_state.nelement()
    
#     print(n_tl_on)
    
#     exit()
    
#     print(new_ligh_state.size())
#     print(new_ligh_state)
#     print(new_ligh_state -1)