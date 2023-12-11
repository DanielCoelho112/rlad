import torch
import torch.nn as nn
import torch.optim as optimizer

from utilities.networks import weights_init

class TrafficLightDecoder():
    def __init__(self, traffic_light_config, image_latent_size, device, checkpoint_dir):
        super(TrafficLightDecoder, self).__init__()

        self.device = device 
        
        self.tl_encoder = TrafficLightEncoder(lr=traffic_light_config['lr'], image_state_size=image_latent_size, latent_size=traffic_light_config['tl_latent_size'], weight_decay=traffic_light_config['weight_decay'], device=device, checkpoint_dir=checkpoint_dir)
        self.tl_color = TrafficLightColorWeights(lr=traffic_light_config['lr'], latent_size=traffic_light_config['tl_latent_size'], fc_dims=traffic_light_config['tl_fc_dims'], weight_decay=traffic_light_config['weight_decay'], device=device, checkpoint_dir=checkpoint_dir)

    def update(self, raw_state_batch, image_encoder, image_encoder_optim):
    
        self.zero_grad(image_encoder_optim)
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) 
        
        loss = self.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss.backward()
        
        self.step(image_encoder_optim)
        
    def zero_grad(self, image_encoder_optim):
        image_encoder_optim.zero_grad(set_to_none=True)
        self.tl_encoder.optimizer.zero_grad(set_to_none=True)
        self.tl_color.optimizer.zero_grad(set_to_none=True)
    
    
    def step(self, image_encoder_optim):
        image_encoder_optim.step()
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
        

class TrafficLightColorWeights(nn.Module):
    def __init__(self, lr, latent_size, fc_dims, weight_decay, device, checkpoint_dir):
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
        
        self.loss = nn.CrossEntropyLoss()

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