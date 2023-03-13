import torch

from agents.models.traffic_light.traffic_light import TrafficLightV1, TrafficLightV2, TrafficLightV3
from agents.models.safe_danger.safe_danger import SafeDanger

class ClassificationBranch():
    def __init__(self, classification_branch_config, image_latent_size, device, checkpoint_dir):
        super(ClassificationBranch, self).__init__()

        self.device = device 
        
        
        self.traffic_light = TrafficLightV1(lr=classification_branch_config['lr'], image_latent_size=image_latent_size, latent_size=classification_branch_config['tl_latent_size'], fc_dims=classification_branch_config['tl_fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                            device=device, checkpoint_dir=checkpoint_dir)
        
        
    def update(self, raw_state_batch, image_encoder):
    
        self.zero_grad(image_encoder)
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        loss = self.traffic_light.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss.backward()
        
        self.step(image_encoder)
        
    def zero_grad(self, image_encoder):
        image_encoder.optimizer.zero_grad(set_to_none=True)
        self.traffic_light.zero_grad()
    
    def step(self, image_encoder):
        image_encoder.optimizer.step()
        self.traffic_light.step()
        
    def train(self):
        self.traffic_light.train()

    def eval(self):
        self.traffic_light.eval()

    def save_checkpoint(self):
        self.traffic_light.save_checkpoint()
        
    def load_checkpoint(self):
        self.traffic_light.load_checkpoint()

class ClassificationBranchV2():
    def __init__(self, classification_branch_config, image_latent_size, device, checkpoint_dir):
        super(ClassificationBranchV2, self).__init__()

        self.device = device 
        
        
        self.traffic_light = TrafficLightV2(lr=classification_branch_config['lr'], image_latent_size=image_latent_size, latent_size=classification_branch_config['tl_latent_size'], fc_dims=classification_branch_config['tl_fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                            device=device, checkpoint_dir=checkpoint_dir, weights=classification_branch_config['weights'])
        
        
    def update(self, raw_state_batch, image_encoder):
    
        self.zero_grad(image_encoder)
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        loss = self.traffic_light.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss.backward()
        
        self.step(image_encoder)
        
    def zero_grad(self, image_encoder):
        image_encoder.optimizer.zero_grad(set_to_none=True)
        self.traffic_light.zero_grad()
    
    def step(self, image_encoder):
        image_encoder.optimizer.step()
        self.traffic_light.step()
        
    def train(self):
        self.traffic_light.train()

    def eval(self):
        self.traffic_light.eval()

    def save_checkpoint(self):
        self.traffic_light.save_checkpoint()
        
    def load_checkpoint(self):
        self.traffic_light.load_checkpoint()

class ClassificationBranchV2_lr3():
    def __init__(self, classification_branch_config, image_latent_size, image_encoder, device, checkpoint_dir):
        super(ClassificationBranchV2, self).__init__()

        self.device = device 
        
        
        self.traffic_light = TrafficLightV2(lr=classification_branch_config['lr'], image_latent_size=image_latent_size, latent_size=classification_branch_config['tl_latent_size'], fc_dims=classification_branch_config['tl_fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                            device=device, checkpoint_dir=checkpoint_dir, weights=classification_branch_config['weights'])
        
        
        
        self.checkpoint_image_encoder_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_tl.pt"
        self.image_encoder_optimizer =  torch.optim.Adam(
            image_encoder.parameters(), lr=classification_branch_config['lr'], weight_decay=classification_branch_config['weight_decay'])
        
        self.to(self.device)

        
        
    def update(self, raw_state_batch, image_encoder):
    
        self.zero_grad()
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        loss = self.traffic_light.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss.backward()
        
        self.step()
        
    def zero_grad(self):
        self.image_encoder_optimizer.zero_grad(set_to_none=True)
        self.traffic_light.zero_grad()
    
    def step(self):
        self.image_encoder_optimizer.step()
        self.traffic_light.step()
        
    def train(self):
        self.traffic_light.train()

    def eval(self):
        self.traffic_light.eval()

    def save_checkpoint(self):
        self.traffic_light.save_checkpoint()
        torch.save(self.image_encoder_optimizer.state_dict(), self.checkpoint_image_encoder_optimizer)
        
    def load_checkpoint(self):
        self.traffic_light.load_checkpoint()
        self.image_encoder_optimizer.load_state_dict(torch.load(self.checkpoint_image_encoder_optimizer))


class ClassificationBranchV2_alix():
    def __init__(self, classification_branch_config, image_latent_size, device, checkpoint_dir):
        super(ClassificationBranchV2_alix, self).__init__()

        self.device = device 
        
        
        self.traffic_light = TrafficLightV2(lr=classification_branch_config['lr'], image_latent_size=image_latent_size, latent_size=classification_branch_config['tl_latent_size'], fc_dims=classification_branch_config['tl_fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                            device=device, checkpoint_dir=checkpoint_dir, weights=classification_branch_config['weights'])
        
        
    def update(self, raw_state_batch, image_encoder, image_encoder_optim):
    
        self.zero_grad(image_encoder_optim)
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        loss = self.traffic_light.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss.backward()
        
        self.step(image_encoder_optim)
        
    def zero_grad(self, image_encoder_optim):
        image_encoder_optim.zero_grad(set_to_none=True)
        self.traffic_light.zero_grad()
    
    def step(self, image_encoder_optim):
        image_encoder_optim.step()
        self.traffic_light.step()
        
    def train(self):
        self.traffic_light.train()

    def eval(self):
        self.traffic_light.eval()

    def save_checkpoint(self):
        self.traffic_light.save_checkpoint()
        
    def load_checkpoint(self):
        self.traffic_light.load_checkpoint()

class ClassificationBranchV3():
    def __init__(self, classification_branch_config, image_latent_size, device, checkpoint_dir):
        super(ClassificationBranchV3, self).__init__()

        self.device = device 
        
        
        self.traffic_light = TrafficLightV3(lr=classification_branch_config['lr'], image_latent_size=image_latent_size, latent_size=classification_branch_config['tl_latent_size'], fc_dims=classification_branch_config['tl_fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                            device=device, checkpoint_dir=checkpoint_dir, weights=classification_branch_config['weights'])
        
        
    def update(self, raw_state_batch, image_encoder):
    
        self.zero_grad(image_encoder)
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        loss = self.traffic_light.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss.backward()
        
        self.step(image_encoder)
        
    def zero_grad(self, image_encoder):
        image_encoder.optimizer.zero_grad(set_to_none=True)
        self.traffic_light.zero_grad()
    
    def step(self, image_encoder):
        image_encoder.optimizer.step()
        self.traffic_light.step()
        
    def train(self):
        self.traffic_light.train()

    def eval(self):
        self.traffic_light.eval()

    def save_checkpoint(self):
        self.traffic_light.save_checkpoint()
        
    def load_checkpoint(self):
        self.traffic_light.load_checkpoint()

        
class ClassificationBranchV4_alix():
    def __init__(self, classification_branch_config, image_latent_size, device, checkpoint_dir):
        super(ClassificationBranchV4_alix, self).__init__()

        self.device = device 
        self.weight_tl = classification_branch_config['weight_tl']
        self.weight_danger = classification_branch_config['weight_danger']
        
        
        self.traffic_light = TrafficLightV2(lr=classification_branch_config['lr'], image_latent_size=image_latent_size, latent_size=classification_branch_config['tl_latent_size'], fc_dims=classification_branch_config['tl_fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                            device=device, checkpoint_dir=checkpoint_dir, weights=classification_branch_config['weights'])
        
        # weight the losses from traffic_light and safe_danger
        # try 10, 1
        self.safe_danger = SafeDanger(l3=classification_branch_config['lr'], image_state_size=image_latent_size, latent_size=classification_branch_config['latent_size'], fc_dims=classification_branch_config['fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                      device=device, checkpoint_dir=checkpoint_dir)
        
        
    def update(self, raw_state_batch, image_encoder, image_encoder_optim):
    
        self.zero_grad(image_encoder_optim)
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        loss_tl = self.traffic_light.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        loss_danger = self.safe_danger.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss = self.weight_tl * loss_tl + self.weight_danger * loss_danger
        
        loss.backward()
        
        self.step(image_encoder_optim)
        
    def zero_grad(self, image_encoder_optim):
        image_encoder_optim.zero_grad(set_to_none=True)
        self.safe_danger.optimizer.zero_grad(set_to_none=True)
        self.traffic_light.zero_grad()
    
    def step(self, image_encoder_optim):
        image_encoder_optim.step()
        self.safe_danger.optimizer.step()
        self.traffic_light.step()
        
    def train(self):
        self.traffic_light.train()
        self.safe_danger.train()

    def eval(self):
        self.traffic_light.eval()
        self.safe_danger.eval()
        
    def save_checkpoint(self):
        self.traffic_light.save_checkpoint()
        self.safe_danger.save_checkpoint()
        
    def load_checkpoint(self):
        self.traffic_light.load_checkpoint()
        self.safe_danger.load_checkpoint()