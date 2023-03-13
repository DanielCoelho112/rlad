import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init, get_n_params_network, SpatialAttn

# https://github.com/0aqz0/pytorch-attention-mechanism/blob/3625e7ad82ea5e7c01e1558e469883518b25ce31/models.py#L10
class EncoderV0(nn.Module):
    def __init__(self, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV0, self).__init__()

        self.device = device

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 32, 5, stride=2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        
        self.conv6 = nn.Sequential(nn.Conv2d(64, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.dense = nn.Conv2d(64, 64, kernel_size=4, padding=0, bias=True)
                
        
        
        # self.projector = ProjectorBlock(64, 128)
    
        self.attn1 = SpatialAttn(in_features=64, normalize_attn=True)
        self.attn2 = SpatialAttn(in_features=64, normalize_attn=True)
        self.attn3 = SpatialAttn(in_features=64, normalize_attn=True)
        
        
        
    
        self.linear = nn.Sequential(nn.Linear(64 * 3, latent_size, bias=True),
                                nn.LayerNorm(latent_size), nn.Tanh())    
    
    
    
                             
        self._initialize_weights()

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.conv1(x)
        x = self.conv2(x)
        l1 = self.conv3(x)
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)
        l2 = self.conv4(x)
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)
        l3 = self.conv5(x)
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)
        x = self.conv6(x)
        
        g = self.dense(x)
        
        g1 = self.attn1(l1, g)
        g2 = self.attn2(l2, g)
        g3 = self.attn3(l3, g)
        
        g = torch.cat((g1,g2,g3), dim=1)
        out = self.linear(g)
    
    
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))







# if __name__ == '__main__':
#     enc = EncoderV0(lr=0, weight_decay=0, latent_size=256,
#                     device=torch.device('cuda:0'), target=False, checkpoint_dir=None)

#     import numpy as np 
#     a = np.full((10, 3, 224, 224), 100, dtype=np.int8)

#     a_torch = torch.from_numpy(a).to(torch.device('cuda:0'))

#     print(enc(a_torch).size())
#     print(get_n_params_network(enc))
