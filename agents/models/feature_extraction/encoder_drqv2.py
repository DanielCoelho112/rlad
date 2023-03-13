import torch
import torch.nn as nn
import torch.optim as optimizer

from utilities.networks import weights_init, get_n_params_network


class Encoder(nn.Module):
    def __init__(self, lr, image_size, weight_decay, device, target, checkpoint_dir):
        super(Encoder, self).__init__()

        self.image_size = image_size
        self.device = device
        self.out_dim = 32 * 47 * 47

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())




        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        out = x.view(x.shape[0], -1)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))

class EncoderV1(nn.Module):
    def __init__(self, lr, weight_decay, device, target, checkpoint_dir):
        super(EncoderV1, self).__init__()

        self.device = device
        self.out_dim = 32 * 23 * 23

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
                                   



        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        out = x.view(x.shape[0], -1)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))

class EncoderV1_256(nn.Module):
    def __init__(self, lr, latent_size,  weight_decay, device, target, checkpoint_dir, pretrained=False, input_channels=3):
        super(EncoderV1_256, self).__init__()

        self.device = device
        self.out_dim = 32 * 23 * 23

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.linear = nn.Sequential(nn.Linear(self.out_dim, latent_size),
                                    nn.LayerNorm(latent_size), nn.Tanh())
                                   



        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


class EncoderV2(nn.Module):
    def __init__(self, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV2, self).__init__()

        self.device = device
        self.out_dim = 64 * 6 * 6

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 32, 5, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU())
        
        self.linear = nn.Sequential(nn.Linear(self.out_dim, latent_size),
                                    nn.LayerNorm(latent_size), nn.Tanh())
                                   



        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))

class EncoderV2_without256(nn.Module):
    def __init__(self, lr, weight_decay, device, target, checkpoint_dir, input_channels=3, latent_size=None, pretrained=False):
        super(EncoderV2_without256, self).__init__()

        self.device = device
        self.out_dim = 64 * 6 * 6

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 32, 5, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU())

        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        out = x.view(x.shape[0], -1)
        
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))
        
class EncoderV3(nn.Module):
    def __init__(self, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV3, self).__init__()

        self.device = device
        self.out_dim = 128 * 4 * 4

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 32, 3, stride=2),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True), 
                                     nn.Conv2d(32, 64, 3, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True), 
                                     nn.Conv2d(64, 64, 3, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True), 
                                     nn.Conv2d(64, 128, 3, stride=2),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True), 
                                     nn.Conv2d(128, 128, 3, stride=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, stride=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.linear = nn.Sequential(nn.Linear(self.out_dim, latent_size),
                                    nn.LayerNorm(latent_size), nn.Tanh())
                                   



        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


# image size is 120.
class EncoderV4(nn.Module): 
    def __init__(self, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV4, self).__init__()

        self.device = device
        self.out_dim = 32 * 4 * 4

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.linear = nn.Sequential(nn.Linear(self.out_dim, latent_size),
                                    nn.LayerNorm(latent_size), nn.Tanh())
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))

# bisimulation encoder
class EncoderV5(nn.Module):
    def __init__(self, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV5, self).__init__()

        self.device = device
        self.out_dim = 256 * 2 * 2

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 64, 5, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 128, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(128, 128, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(128, 128, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(128, 256, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(256, 256, 3, stride=2),
                                     nn.ReLU())
        
        self.linear = nn.Sequential(nn.Linear(self.out_dim, latent_size),
                                    nn.LayerNorm(latent_size))
                                   



        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.0
        
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))

class EncoderV6_256(nn.Module):
    def __init__(self, lr, weight_decay, latent_size, device, target, checkpoint_dir,input_channels=3, pretrained=False):
        super(EncoderV6_256, self).__init__()

        self.device = device
        self.out_dim = 64 * 5 * 5

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(input_channels, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 64, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU())

        
        self.linear = nn.Sequential(nn.Linear(self.out_dim, latent_size),
                                    nn.LayerNorm(latent_size), nn.Tanh())
                                   
        self.apply(weights_init)
        
        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)

        self.to(self.device)
    


    def forward(self, x):
        x = x / 255.0 - 0.5
        
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
            
        return out
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))






# if __name__ == '__main__':
#     enc = EncoderV7_256(lr=0, weight_decay=0, latent_size=256,
#                     device=torch.device('cuda:1'), target=False, checkpoint_dir=None)

#     import numpy as np 
#     a = np.full((256, 3, 256, 256), 100, dtype=np.int8)

#     a_torch = torch.from_numpy(a).to(torch.device('cuda:1'))

#     print(enc(a_torch).size())
#     print(get_n_params_network(enc))
