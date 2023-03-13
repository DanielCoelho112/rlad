import torch 
import torch.nn as nn
import torch.optim as optimizer

from torchvision import models
from torchvision.models import ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights

from utilities.networks import get_n_params_network

class ShuffleNet05(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(ShuffleNet05, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/shufflenet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/shufflenet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/shufflenet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/shufflenet.pt"
        
        if pretrained:
            self.model = models.shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        else:
            self.model = models.shufflenet_v2_x0_5()
            
            
        in_features_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1.transforms()
        
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
       
class ShuffleNet05F(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(ShuffleNet05F, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/shufflenet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/shufflenet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/shufflenet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/shufflenet.pt"
        
        if pretrained:
            self.model = models.shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        else:
            self.model = models.shufflenet_v2_x0_5()
        
        if pretrained:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False
            
            for param in self.model.conv5.parameters():
                param.requires_grad = True
        
        in_features_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1.transforms()
        
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
        
        
class ShuffleNet10(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(ShuffleNet10, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/shufflenet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/shufflenet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/shufflenet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/shufflenet.pt"
        
        if pretrained:
            self.model = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        else:
            self.model = models.shufflenet_v2_x1_0()
            
            
        in_features_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.transforms()
        
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

# import numpy as np

# if __name__ == '__main__':
#     model = ShuffleNet05F(latent_size=256, lr=1e-3, weight_decay=1e-2, device=torch.device('cuda:3'), checkpoint_dir='', pretrained=True, target=False)
    
#     # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     # params = sum([np.prod(p.size()) for p in model_parameters])
#     # print(params)
    
#     import numpy as np
    
#     a = np.full((10, 3, 256, 256), 100, dtype=np.int8)
    
#     aa = torch.from_numpy(a).to(torch.device('cuda:3'))
    
    
#     print(model(aa).size())
    
    
    # print(model.transforms(aa).size()) 
    
    