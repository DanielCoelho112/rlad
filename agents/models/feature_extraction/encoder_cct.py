import torch
import torch.nn as nn
import torch.optim as optimizer

from vit_pytorch.cct import cct_2, cct_4, cct_6

from utilities.networks import weights_init, get_n_params_network


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

        
        self.model =  cct_2(
                            img_size = 224,
                            n_conv_layers = 2,
                            kernel_size = 7,
                            stride = 2,
                            padding = 3,
                            pooling_kernel_size = 3,
                            pooling_stride = 2,
                            pooling_padding = 1,
                            num_classes = 256,
                            positional_embedding = 'sine', # ['sine', 'learnable', 'none']
                            )
        
        
        self.model.classifier.fc = nn.Sequential(nn.Linear(128, latent_size),
                                                 nn.LayerNorm(latent_size), nn.Tanh())
                                   
        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0
        
        out = self.model(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        
class EncoderV1(nn.Module):
    def __init__(self, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV1, self).__init__()

        self.device = device

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.model =  cct_4(
                            img_size = 224,
                            n_conv_layers = 2,
                            kernel_size = 7,
                            stride = 2,
                            padding = 3,
                            pooling_kernel_size = 3,
                            pooling_stride = 2,
                            pooling_padding = 1,
                            num_classes = 256,
                            positional_embedding = 'sine', # ['sine', 'learnable', 'none']
                            )
        
        
        self.model.classifier.fc = nn.Sequential(nn.Linear(128, latent_size),
                                                 nn.LayerNorm(latent_size), nn.Tanh())
                                   
        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0
        
        out = self.model(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


# cct.classifier.fc = torch.nn.Linear(256, 256)

# if __name__ == '__main__':
    
#     cc = EncoderV0(lr=0.001,latent_size=39200, weight_decay=0, device=torch.device('cuda:0'), target=False, checkpoint_dir='')
    
#     print(cc)
    
#     print(get_n_params_network(cc))

# # print(cct)