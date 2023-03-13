import torch
import torch.optim as optimizer

from agents.models.traffic_light.traffic_light import TrafficLightV1
from agents.models.semantic_segmentation.semantic_segmentation import SemanticSegmentationV1

class ClassificationBranch():
    def __init__(self, classification_branch_config, image_latent_size, image_encoder, device, checkpoint_dir):
        super(ClassificationBranch, self).__init__()

        self.device = device 
        self.weight_semantic = classification_branch_config['weight_semantic']
        self.weight_traffic_light = classification_branch_config['weight_traffic_light']
        
        # segmentation
        self.semantic_segmentation = SemanticSegmentationV1(entrypoint=classification_branch_config['decoder_segmentation_entry_point'],
                                                            lr=classification_branch_config['lr'], n_class=classification_branch_config['n_class'],
                                                            image_encoder=image_encoder, image_latent_size=image_latent_size, weight_decay=classification_branch_config['weight_decay'],
                                                            device=device, checkpoint_dir=checkpoint_dir)
        
        
        self.traffic_light = TrafficLightV1(lr=classification_branch_config['lr'], image_latent_size=image_latent_size, latent_size=classification_branch_config['tl_latent_size'], fc_dims=classification_branch_config['tl_fc_dims'], weight_decay=classification_branch_config['weight_decay'],
                                            device=device, checkpoint_dir=checkpoint_dir)
        
        
        self.optimizer_encoder = optimizer.Adam(
            image_encoder.parameters(), lr=classification_branch_config['lr'], weight_decay=classification_branch_config['weight_decay'])    
        
        self.optimizer_encoder_checkpoint_file = f"{checkpoint_dir}/weights/optimizers/classification_branch_encoder.pt"
        
        
    def update(self, raw_state_batch, image_encoder):
    
        self.zero_grad()
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        semantic_loss = self.semantic_segmentation.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        tl_loss = self.traffic_light.compute_loss(raw_state_batch=raw_state_batch, latent_image=latent_image)
        
        loss = self.weight_semantic * semantic_loss + self.weight_traffic_light * tl_loss
        
        loss.backward()
        
        self.step()
        
    def zero_grad(self):
        self.semantic_segmentation.zero_grad()
        self.traffic_light.zero_grad()
        self.optimizer_encoder.zero_grad(set_to_none=True)
    
    def step(self):
        self.semantic_segmentation.step()
        self.traffic_light.step()
        self.optimizer_encoder.step()
        
    def train(self):
        self.semantic_segmentation.train()
        self.traffic_light.train()

    def eval(self):
        self.semantic_segmentation.eval()
        self.traffic_light.eval()

    def save_checkpoint(self):
        self.semantic_segmentation.save_checkpoint()
        self.traffic_light.save_checkpoint()
        torch.save(self.optimizer_encoder.state_dict(), self.optimizer_encoder_checkpoint_file)
        

    def load_checkpoint(self):
        self.semantic_segmentation.load_checkpoint()
        self.traffic_light.load_checkpoint()
        self.optimizer_encoder.load_state_dict(torch.load(self.optimizer_encoder_checkpoint_file))

