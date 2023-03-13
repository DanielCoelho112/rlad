import torch 
import torch.nn as nn
import torch.optim as optimizer

from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from utilities.networks import get_n_params_network

class EfficientNet(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(EfficientNet, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/efficientnet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/efficientnet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/efficientnet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/efficientnet.pt"
        
        if pretrained:
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.model = models.efficientnet_b0()
        
        print(self.model) 
            
        in_features_fc = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
        
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
        


# if __name__ == '__main__':
#     model = EfficientNet(latent_size=128, lr=1e-3, weight_decay=1e-2, device=torch.device('cuda:0'), checkpoint_dir='', pretrained=False, target=False)
    
    
#     import numpy as np
    
#     a = np.full((10, 3, 256, 256), 100, dtype=np.uint8)
    
#     aa = torch.from_numpy(a).to(torch.device('cuda:0'))
    
#     print(model(aa).shape)
    
#     print(get_n_params_network(model))
    
    
    