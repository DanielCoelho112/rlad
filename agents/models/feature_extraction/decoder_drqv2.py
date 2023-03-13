import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init, get_n_params_network

class DecoderV1_256(nn.Module):
    def __init__(self, lr, latent_size,  weight_decay, device, checkpoint_dir):
        super(DecoderV1_256, self).__init__()

        self.device = device
        self.out_dim = 32 * 23 * 23


        self.checkpoint_file = f"{checkpoint_dir}/weights/image_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_decoder.pt"

        self.linear = nn.Sequential(nn.Linear(latent_size ,self.out_dim))

        self.convtranspose = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 32, 3, stride=2),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1))
        

        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.view(-1, 32, 23, 23)
        
        out = self.convtranspose(x)
  
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


class DecoderV6_256(nn.Module):
    def __init__(self, lr,  weight_decay, latent_size, device, checkpoint_dir):
        super(DecoderV6_256, self).__init__()

        self.device = device
        self.out_dim = 64 * 5 * 5

        self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        self.linear = nn.Linear(latent_size, self.out_dim)
        
        self.convtranspose = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(64, 64, 3, stride=1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(64, 64, 3, stride=1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(64, 64, 3, stride=1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(64, 64, 3, stride=1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(64, 32, 3, stride=2),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(32, 32, 3, stride=2),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(32, 32, 3, stride=2),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1))

    
        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.apply(weights_init)
        
        self.to(self.device)
    


    def forward(self, x):
        x = F.relu(self.linear(x)) 
        x = x.view(-1, 64, 5, 5)
        out = self.convtranspose(x)
            
        return out
    

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))



# if __name__ == '__main__':
#     enc = EncoderV1(lr=0, weight_decay=0, image_size=224,
#                     device=torch.device('cuda:0'), target=False, checkpoint_dir=None)

#     import numpy as np 
#     a = np.full((64, 3, 224, 224), 100, dtype=np.int8)

#     a_torch = torch.from_numpy(a).to(torch.device('cuda:0'))

#     print(enc(a_torch).size())
#     print(get_n_params_network(enc))
