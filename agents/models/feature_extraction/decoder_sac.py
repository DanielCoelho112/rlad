import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np


from utilities.networks import weights_init, get_n_params_network


class DecoderSACAE(nn.Module):
    def __init__(self, lr, weight_decay, latent_size, device, checkpoint_dir):
        super(DecoderSACAE, self).__init__()

        self.device = device
        self.out_dim = 32 * 35 * 35

       

        self.checkpoint_file = f"{checkpoint_dir}/weights/image_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_decoder.pt"

        
        self.linear = nn.Linear(latent_size, self.out_dim)
        
        self.convtranspose = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 32, 3, stride=1),
                                     nn.ReLU(), 
                                     nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1),
                                     nn.ReLU()) 

        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.view(-1, 32, 35, 35)
        
        out = self.convtranspose(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
        




# if __name__ == '__main__':
#     enc = DecoderSACAE(lr=0, weight_decay=0, latent_size=80, device=torch.device('cuda:1'), target=False, checkpoint_dir=None)


#     a = np.full((64, 80), 0.5, dtype=np.float32)

#     a_torch = torch.from_numpy(a).to(torch.device('cuda:1'))

#     print(enc(a_torch).size())
