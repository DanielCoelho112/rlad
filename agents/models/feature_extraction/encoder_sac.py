import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np


from utilities.networks import weights_init, get_n_params_network



class EncoderSACAE(nn.Module):
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(EncoderSACAE, self).__init__()

        self.device = device
        self.out_dim = 32 * 35 * 35

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU()) 

        self.fc = nn.Linear(self.out_dim, latent_size)
        self.ln = nn.LayerNorm(latent_size)

        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x / 255.
        x = self.convnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.ln(x)
        out = torch.tanh(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
        


class EncoderDRQV2(nn.Module):
    def __init__(self, lr, weight_decay, pretrained, latent_size, device, target, checkpoint_dir):
        super(EncoderDRQV2, self).__init__()

        self.device = device
        self.out_dim = 32 * 35 * 35

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
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
        x = x / 255. - 0.5 
        x = self.convnet(x)
        out = x.view(x.shape[0], -1)
        
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))



if __name__ == '__main__':
    enc = EncoderSACAE(lr=0, weight_decay=0, pretrained=False,
                    latent_size=80, device=torch.device('cuda:1'), target=False, checkpoint_dir=None)


    a = np.full((64, 3, 84, 84), 100, dtype=np.int8)

    a_torch = torch.from_numpy(a).to(torch.device('cuda:1'))

    print(enc(a_torch).size())
    print(get_n_params_network(enc))
