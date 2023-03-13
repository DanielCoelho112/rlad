import torch 
import torch.nn as nn
import torch.optim as optimizer

from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from utilities.networks import get_n_params_network

class MobileNetV3(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(MobileNetV3, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/mobilenet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/mobilenet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/mobilenet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/mobilenet.pt"
        
        if pretrained:
            self.model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            self.model = models.mobilenet_v3_small()
            
        in_features_fc = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
        
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
#     model = MobileNetV3(latent_size=256, lr=1e-3, weight_decay=1e-2, device=torch.device('cuda:2'), checkpoint_dir='', pretrained=True, target=False)
    
    
#     import numpy as np
    
#     a = np.full((10, 3, 256, 256), 100, dtype=np.uint8)
    
#     aa = torch.from_numpy(a).to(torch.device('cuda:2'))
    
#     print(model(aa).shape)
    
    
    
    
    