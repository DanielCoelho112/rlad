import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init

class TrafficLightV2():
    def __init__(self, lr, image_latent_size, latent_size, fc_dims, weight_decay, device, checkpoint_dir, weights):
        super(TrafficLightV2, self).__init__()

        self.tl_encoder = TrafficLightEncoder(lr=lr, image_state_size=image_latent_size, latent_size=latent_size, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_color = TrafficLightColorWeights(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir, weights=weights)

        
    
    def compute_loss(self, raw_state_batch, latent_image):
        
        tl_state_label = raw_state_batch['traffic_light_state'].reshape(-1)
               
        tl_latent = self.tl_encoder(latent_image)
        tl_color = self.tl_color(tl_latent) 
        loss = self.tl_color.loss(tl_color, tl_state_label)
            
        return loss




class SafeDanger(nn.Module):
    def __init__(self, lr, image_state_size, latent_size, fc_dims, weight_decay, device, checkpoint_dir):
        super(SafeDanger, self).__init__()

        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/safe_danger_encoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/safe_danger_encoder.pt"
        self.latent_size = latent_size
        self.classification = nn.Sequential(nn.Linear(image_state_size, latent_size),
                                     nn.ReLU(),
                                     nn.Linear(latent_size, fc_dims),
                                     nn.ReLU(),
                                     nn.Linear(fc_dims, 2))
        
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.loss = nn.CrossEntropyLoss()
        
        self.to(self.device)

    def forward(self, x):
        out = self.classification(x)
        return out
    
    
    def compute_loss(self, raw_state_batch, latent_image):
        
        danger_label = raw_state_batch['danger'].reshape(-1)
               
        danger_prediction = self.classification(latent_image)
        loss = self.loss(danger_prediction, danger_label)
            
        return loss


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