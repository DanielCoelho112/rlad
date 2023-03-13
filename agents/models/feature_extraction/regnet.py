import torch 
import torch.nn as nn
import torch.optim as optimizer

from torchvision import transforms, models
from torchvision.models import RegNet_Y_400MF_Weights, RegNet_Y_800MF_Weights, RegNet_Y_1_6GF_Weights, RegNet_Y_3_2GF_Weights

from utilities.networks import get_n_params_network

class RegNetY400MF(nn.Module): 
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(RegNetY400MF, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/regnet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/regnet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/regnet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/regnet.pt"
        
        if pretrained:
            self.model = models.regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        else:
            self.model = models.regnet_y_400mf()
            
        in_features_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = RegNet_Y_400MF_Weights.IMAGENET1K_V2.transforms()
        
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

class RegNetY800MF(nn.Module): 
    def __init__(self, lr, image_size, image_size_cropped, weight_decay, pretrained, latent_size, use_tahn, device, target, checkpoint_dir):
        super(RegNetY800MF, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/regnet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/regnet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/regnet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/regnet.pt"
        
        if pretrained:
            self.model = models.regnet_y_800mf(weights=RegNet_Y_800MF_Weights.IMAGENET1K_V2)
        else:
            self.model = models.regnet_y_800mf()
            
        in_features_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = RegNet_Y_800MF_Weights.IMAGENET1K_V2.transforms()
        
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


class RegNetY_3_2_GF(nn.Module): 
    def __init__(self, lr, image_size, image_size_cropped, weight_decay, pretrained, latent_size, use_tahn, device, target, checkpoint_dir):
        super(RegNetY_3_2_GF, self).__init__() 
        self.latent_size = latent_size
        self.device = device
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/regnet_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/regnet_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/regnet.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/regnet.pt"
        
        if pretrained:
            self.model = models.regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
        else:
            self.model = models.regnet_y_3_2gf()
            
        in_features_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features_fc, out_features=latent_size)
        
        self.transforms = RegNet_Y_3_2GF_Weights.IMAGENET1K_V2.transforms()
        
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
#     model = RegNetY400MF(latent_size=1000, lr=1e-4, weight_decay=1e-2, device=torch.device('cuda:0'), checkpoint_dir='', pretrained=True, target=False)
#     print(get_n_params_network(model))
    
#     image = cv2.imread("img.png")
    
#     print(image.shape)
#     print(image.dtype)
    
#     resized_image = cv2.resize(image, (232,232), interpolation=cv2.INTER_AREA)
        
#     image_permute = np.einsum(
#             'kij->jki', resized_image)
        
#     print(image_permute.shape)
        
        
#     image_torch = torch.from_numpy(image_permute)
    
#     print(image_torch.shape)
#     print(image_torch.dtype)
    
#     image_transf = model.transforms(image_torch)
    
#     print(image_transf.shape)
#     print(image_transf.dtype)
#     print(image_transf[0])
    
#     image_transf_np = image_transf.numpy()
    
#     image_transf_np_01 = (image_transf_np - np.min(image_transf_np))/np.ptp(image_transf_np)
    
#     image_transf_np_01_ = np.einsum(
#             'kij->ijk', image_transf_np_01)
    
#     cv2.imshow('test', image_transf_np_01_)
#     cv2.waitKey(0)
    
    
    #img = torch.rand(size=(400,3,224,224)).to('cuda:0')
    #latent_vector = model(img)
    #print(latent_vector.size())

        
