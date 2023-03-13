import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init



class TrafficLightV0():
    def __init__(self, lr, image_latent_size, latent_size, fc_dims, weight_decay, device, checkpoint_dir):
        super(TrafficLightV0, self).__init__()

        self.tl_encoder = TrafficLightEncoder(lr=lr, image_state_size=image_latent_size, latent_size=latent_size, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_onoff = TrafficLightOnOff(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_color = TrafficLightColor(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_distance = TrafficLightDistance(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
    
    
    def update(self, raw_state_batch, image_encoder):
        # traffic_light_state (Nx1)
        # traffic_light_distance (Nx1)
        
        tl_state_label = raw_state_batch['traffic_light_state']
        tl_distance_label = raw_state_batch['traffic_light_distance']

        # idxs where the traffic light is on.
        idxs_tl_on = tl_state_label>0
        idxs_tl_on = idxs_tl_on.reshape(-1)
        
        tl_state_0_1_label = tl_state_label.clone()
        tl_state_color_label = tl_state_label.clone()
        
        # tl tensor with 0s and 1s
        tl_state_0_1_label[idxs_tl_on] = 1
        tl_state_0_1_label = tl_state_0_1_label.reshape(-1)
        
        # tl tensor only with transitions where tf is on. -1 is to start labeling from 0.
        tl_state_color_label = tl_state_color_label[idxs_tl_on].reshape(-1) - 1
        
        # number of transitions with tl on.
        n_tl_on = tl_state_color_label.nelement()
        
        # tl distances tensor only with transitions where tf is on. 
        tl_distance_label = tl_distance_label[idxs_tl_on].reshape(-1,1)
        
        
        image_batch = raw_state_batch['central_rgb'].to(torch.float32)
        latent_image = image_encoder(image_batch) # Nx256
        
        tl_latent = self.tl_encoder(latent_image)
        
        tl_on_off = self.tl_onoff(tl_latent) #Nx2
        
        image_encoder.optimizer.zero_grad(set_to_none=True)
        self.tl_encoder.optimizer.zero_grad(set_to_none=True)
        self.tl_onoff.optimizer.zero_grad(set_to_none=True)
        self.tl_color.optimizer.zero_grad(set_to_none=True)
        self.tl_distance.optimizer.zero_grad(set_to_none=True)
        
        loss_on_off = self.tl_onoff.loss(tl_on_off, tl_state_0_1_label)
        
        if n_tl_on > 0:
            tl_latent_on = tl_latent[idxs_tl_on].reshape(n_tl_on, self.tl_encoder.latent_size)  
            
            tl_color = self.tl_color(tl_latent_on) 
            loss_color = self.tl_color.loss(tl_color, tl_state_color_label)
            
            tl_distance = self.tl_distance(tl_latent_on)
            loss_distance = F.mse_loss(tl_distance, tl_distance_label)
            
            loss = loss_on_off + loss_color + loss_distance
            
        else:

            loss = loss_on_off
            
        loss.backward()
        
        image_encoder.optimizer.step()
        self.tl_encoder.optimizer.step()
        self.tl_onoff.optimizer.step()
        self.tl_color.optimizer.step()
        self.tl_distance.optimizer.step()
    
    def train(self):
        self.tl_encoder.train()
        self.tl_onoff.train()
        self.tl_color.train()
        self.tl_distance.train()
    
    def eval(self):
        self.tl_encoder.eval()
        self.tl_onoff.eval()
        self.tl_color.eval()
        self.tl_distance.eval()
    
    def save_checkpoint(self):
        self.tl_encoder.save_checkpoint()
        self.tl_onoff.save_checkpoint()
        self.tl_color.save_checkpoint()
        self.tl_distance.save_checkpoint()
        
        
    def load_checkpoint(self):
        self.tl_encoder.load_checkpoint()
        self.tl_onoff.load_checkpoint()
        self.tl_color.load_checkpoint()
        self.tl_distance.load_checkpoint()

class TrafficLightEncoder(nn.Module):
    def __init__(self, lr, image_state_size, latent_size, weight_decay, device, checkpoint_dir):
        super(TrafficLightEncoder, self).__init__()

        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/traffic_light_encoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/traffic_light_encoder.pt"
        self.latent_size = latent_size
        self.encoder = nn.Sequential(nn.Linear(image_state_size, latent_size),
                                     nn.ReLU())
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        out = self.encoder(x)
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))


class TrafficLightOnOff(nn.Module):
    def __init__(self, lr, latent_size, fc_dims, weight_decay, device, checkpoint_dir):
        super(TrafficLightOnOff, self).__init__()

        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/traffic_light_on_off_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/traffic_light_on_off_decoder.pt"

        self.decoder = nn.Sequential(nn.Linear(latent_size, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, 2))
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.loss = nn.CrossEntropyLoss()
                
        self.to(self.device)


    def forward(self, x):
        out = self.decoder(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        
class TrafficLightColor(nn.Module):
    def __init__(self, lr, latent_size, fc_dims, weight_decay, device, checkpoint_dir):
        super(TrafficLightColor, self).__init__()

        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/traffic_light_color_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/traffic_light_color_decoder.pt"

        self.decoder = nn.Sequential(nn.Linear(latent_size, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, 2))
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.loss = nn.CrossEntropyLoss()
        
        self.to(self.device)


    def forward(self, x):
        out = self.decoder(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        
class TrafficLightColorWeights(nn.Module):
    def __init__(self, lr, latent_size, fc_dims, weight_decay, device, checkpoint_dir, weights):
        super(TrafficLightColorWeights, self).__init__()

        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/traffic_light_color_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/traffic_light_color_decoder.pt"

        self.decoder = nn.Sequential(nn.Linear(latent_size, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, 3))
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        if not weights:
            self.loss = nn.CrossEntropyLoss()
        else:
            weights_class = torch.tensor([1.0, 4.0, 2.0])
            self.loss = nn.CrossEntropyLoss(weight=weights_class)
        
        self.to(self.device)


    def forward(self, x):
        out = self.decoder(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))


class TrafficLightDistance(nn.Module):
    def __init__(self, lr, latent_size, fc_dims, weight_decay, device, checkpoint_dir):
        super(TrafficLightDistance, self).__init__()

        self.device = device
        self.checkpoint_file = f"{checkpoint_dir}/weights/traffic_light_distance_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/traffic_light_distance_decoder.pt"

        self.decoder = nn.Sequential(nn.Linear(latent_size, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, 1))
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        out = self.decoder(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


class TrafficLightV1():
    def __init__(self, lr, image_latent_size, latent_size, fc_dims, weight_decay, device, checkpoint_dir):
        super(TrafficLightV1, self).__init__()

        self.tl_encoder = TrafficLightEncoder(lr=lr, image_state_size=image_latent_size, latent_size=latent_size, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_onoff = TrafficLightOnOff(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_color = TrafficLightColor(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_distance = TrafficLightDistance(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
    
    def zero_grad(self):
        self.tl_encoder.optimizer.zero_grad(set_to_none=True)
        self.tl_onoff.optimizer.zero_grad(set_to_none=True)
        self.tl_color.optimizer.zero_grad(set_to_none=True)
        self.tl_distance.optimizer.zero_grad(set_to_none=True)
    
    def step(self):
        self.tl_encoder.optimizer.step()
        self.tl_onoff.optimizer.step()
        self.tl_color.optimizer.step()
        self.tl_distance.optimizer.step()
        
    
    def compute_loss(self, raw_state_batch, latent_image):
        # traffic_light_state (Nx1)
        # traffic_light_distance (Nx1)
        
        tl_state_label = raw_state_batch['traffic_light_state']
        tl_distance_label = raw_state_batch['traffic_light_distance']

        # idxs where the traffic light is on.
        idxs_tl_on = tl_state_label>0
        idxs_tl_on = idxs_tl_on.reshape(-1)
        
        tl_state_0_1_label = tl_state_label.clone()
        tl_state_color_label = tl_state_label.clone()
        
        # tl tensor with 0s and 1s
        tl_state_0_1_label[idxs_tl_on] = 1
        tl_state_0_1_label = tl_state_0_1_label.reshape(-1)
        
        # tl tensor only with transitions where tf is on. -1 is to start labeling from 0.
        tl_state_color_label = tl_state_color_label[idxs_tl_on].reshape(-1) - 1
        
        # number of transitions with tl on.
        n_tl_on = tl_state_color_label.nelement()
        
        # tl distances tensor only with transitions where tf is on. 
        tl_distance_label = tl_distance_label[idxs_tl_on].reshape(-1,1)
               
        tl_latent = self.tl_encoder(latent_image)
        
        tl_on_off = self.tl_onoff(tl_latent) #Nx2
        
        
        loss_on_off = self.tl_onoff.loss(tl_on_off, tl_state_0_1_label)
        
        if n_tl_on > 0:
            tl_latent_on = tl_latent[idxs_tl_on].reshape(n_tl_on, self.tl_encoder.latent_size)  
            
            tl_color = self.tl_color(tl_latent_on) 
            loss_color = self.tl_color.loss(tl_color, tl_state_color_label)
            
            tl_distance = self.tl_distance(tl_latent_on)
            loss_distance = F.mse_loss(tl_distance, tl_distance_label)
            
            loss = loss_on_off + loss_color + loss_distance
            
        else:

            loss = loss_on_off
            
        return loss
        
    
    def train(self):
        self.tl_encoder.train()
        self.tl_onoff.train()
        self.tl_color.train()
        self.tl_distance.train()
    
    def eval(self):
        self.tl_encoder.eval()
        self.tl_onoff.eval()
        self.tl_color.eval()
        self.tl_distance.eval()
    
    def save_checkpoint(self):
        self.tl_encoder.save_checkpoint()
        self.tl_onoff.save_checkpoint()
        self.tl_color.save_checkpoint()
        self.tl_distance.save_checkpoint()
        
        
    def load_checkpoint(self):
        self.tl_encoder.load_checkpoint()
        self.tl_onoff.load_checkpoint()
        self.tl_color.load_checkpoint()
        self.tl_distance.load_checkpoint()

class TrafficLightV2():
    def __init__(self, lr, image_latent_size, latent_size, fc_dims, weight_decay, device, checkpoint_dir, weights):
        super(TrafficLightV2, self).__init__()

        self.tl_encoder = TrafficLightEncoder(lr=lr, image_state_size=image_latent_size, latent_size=latent_size, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_color = TrafficLightColorWeights(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir, weights=weights)

    def zero_grad(self):
        self.tl_encoder.optimizer.zero_grad(set_to_none=True)
        self.tl_color.optimizer.zero_grad(set_to_none=True)
    
    def step(self):
        self.tl_encoder.optimizer.step()
        self.tl_color.optimizer.step()
        
    
    def compute_loss(self, raw_state_batch, latent_image):
        
        tl_state_label = raw_state_batch['traffic_light_state'].reshape(-1)
               
        tl_latent = self.tl_encoder(latent_image)
        tl_color = self.tl_color(tl_latent) 
        loss = self.tl_color.loss(tl_color, tl_state_label)
            
        return loss
        
    
    def train(self):
        self.tl_encoder.train()
        self.tl_color.train()
    
    def eval(self):
        self.tl_encoder.eval()
        self.tl_color.eval()
    
    def save_checkpoint(self):
        self.tl_encoder.save_checkpoint()
        self.tl_color.save_checkpoint()
        
        
    def load_checkpoint(self):
        self.tl_encoder.load_checkpoint()
        self.tl_color.load_checkpoint()


class TrafficLightV3():
    def __init__(self, lr, image_latent_size, latent_size, fc_dims, weight_decay, device, checkpoint_dir, weights):
        super(TrafficLightV3, self).__init__()

        self.tl_encoder = TrafficLightEncoder(lr=lr, image_state_size=image_latent_size, latent_size=latent_size, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir)
        self.tl_color = TrafficLightColorWeights(lr=lr, latent_size=latent_size, fc_dims=fc_dims, weight_decay=weight_decay, device=device, checkpoint_dir=checkpoint_dir, weights=weights)

    def zero_grad(self):
        self.tl_encoder.optimizer.zero_grad(set_to_none=True)
        self.tl_color.optimizer.zero_grad(set_to_none=True)
    
    def step(self):
        self.tl_encoder.optimizer.step()
        self.tl_color.optimizer.step()
        
    
    def compute_loss(self, raw_state_batch, latent_image):
        
        tl_state_label = raw_state_batch['traffic_light_state'].reshape(-1)
        weights = torch.bincount(tl_state_label, minlength=3)
        weights = torch.max(weights) / weights
        weights = torch.nan_to_num(weights, posinf=1.0)
        
        loss_object = nn.CrossEntropyLoss(weight=weights)
        
        tl_latent = self.tl_encoder(latent_image)
        tl_color = self.tl_color(tl_latent) 
        loss = loss_object(tl_color, tl_state_label)
        
        return loss
        
    
    def train(self):
        self.tl_encoder.train()
        self.tl_color.train()
    
    def eval(self):
        self.tl_encoder.eval()
        self.tl_color.eval()
    
    def save_checkpoint(self):
        self.tl_encoder.save_checkpoint()
        self.tl_color.save_checkpoint()
        
        
    def load_checkpoint(self):
        self.tl_encoder.load_checkpoint()
        self.tl_color.load_checkpoint()


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