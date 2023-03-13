import torch 
import torch.nn as nn
import torch.optim as optimizer

from torchvision import models
from torchvision.models import MNASNet0_5_Weights, MNASNet0_75_Weights

from utilities.networks import get_n_params_network

class MNASNet0_5(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(MNASNet0_5, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/mnasnet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/mnasnet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/mnasnet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/mnasnet.pt"
        
        if pretrained:
            self.model = models.mnasnet0_5(weights=MNASNet0_5_Weights.IMAGENET1K_V1)
        else:
            self.model = models.mnasnet0_5()
        
        print(self.model)
            
        in_features_fc = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = MNASNet0_5_Weights.IMAGENET1K_V1.transforms()
        
        self.optimizer = optimizer.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)
    
    def forward(self, x):
       x = self.transforms(x)
       return self.model(x)   
                
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
        
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        
class MNASNet0_75(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(MNASNet0_75, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/mnasnet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/mnasnet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/mnasnet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/mnasnet.pt"
        
        if pretrained:
            self.model = models.mnasnet0_75(weights=MNASNet0_75_Weights.IMAGENET1K_V1)
        else:
            self.model = models.mnasnet0_75()
            
        
        in_features_fc = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = MNASNet0_75_Weights.IMAGENET1K_V1.transforms()
        
        self.optimizer = optimizer.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)
    
    def forward(self, x):
       x = self.transforms(x)
       return self.model(x)   
                
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
        
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        

if __name__ == '__main__':
    model = MNASNet0_5(latent_size=256, lr=1e-3, weight_decay=1e-2, device=torch.device('cuda:0'), checkpoint_dir='', pretrained=True, target=False)
    
    
    # import numpy as np
    
    # a = np.full((2, 3, 256, 256), 100, dtype=np.uint8)
    
    # aa = torch.from_numpy(a).to(torch.device('cuda:0'))
    
    # print(model(aa).shape)
    
    # print(get_n_params_network(model))
    
    
    
    
    
    
    