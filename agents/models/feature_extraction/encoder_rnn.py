import torch
import torch.nn as nn
import torch.optim as optimizer

from utilities.networks import weights_init, get_n_params_network

class EncoderV0(nn.Module):
    def __init__(self, n_layers, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV0, self).__init__()

        self.device = device
        self.n_layers = n_layers
        self.latent_size = latent_size

        self.out_dim = 64 * 6 * 6
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5, stride=2),
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
        
        self.linear = nn.Linear(self.out_dim, latent_size)
        
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=latent_size, num_layers=1, bidirectional=False, batch_first=True)
        
    
        self.norm = nn.Sequential(nn.LayerNorm(latent_size), nn.Tanh())
                                   

        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        # x: BxTxCxHxW
        s = x.size()
        x = x.view(-1, *s[2:]) # x: (B*T)x(C)x(H)x(W))
        
        x = x / 255.0 - 0.5
        x = self.convnet(x).view(s[0]*s[1], -1)
        x = self.linear(x)
        x = x.view(s[0], s[1], -1)
        
        h0 = torch.zeros(self.n_layers, x.size(0), self.latent_size).to(self.device) 
        c0 = torch.zeros(self.n_layers, x.size(0), self.latent_size).to(self.device)
       
        x, (_, _) = self.lstm(x, (h0, c0))  # BxTxd'
        
        # getting last time from left to right
        x = x[:,-1,:]
        
        out = self.norm(x)
        

        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))

class EncoderV1(nn.Module):
    def __init__(self, n_layers, lr, latent_size,  weight_decay, device, target, checkpoint_dir, input_channels=3, pretrained=False):
        super(EncoderV1, self).__init__()

        self.device = device
        self.n_layers = n_layers
        self.latent_size = latent_size

        self.out_dim = 64 * 6 * 6
        
        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/image_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_encoder.pt"

        
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5, stride=2),
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
        
        self.linear = nn.Linear(self.out_dim, latent_size)
        
        self.gru = nn.GRU(input_size=latent_size, hidden_size=latent_size, num_layers=1, bidirectional=False, batch_first=True)
        
    
        self.norm = nn.Sequential(nn.LayerNorm(latent_size), nn.Tanh())
                                   

        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        # x: BxTxCxHxW
        s = x.size()
        x = x.view(-1, *s[2:]) # x: (B*T)x(C)x(H)x(W))
        
        x = x / 255.0 - 0.5
        x = self.convnet(x).view(s[0]*s[1], -1)
        x = self.linear(x)
        x = x.view(s[0], s[1], -1)
        
        h0 = torch.zeros(self.n_layers, x.size(0), self.latent_size).to(self.device) 
       
        x, _ = self.gru(x, h0)  # BxTxd'
        
        # getting last time from left to right
        x = x[:,-1,:]
        
        out = self.norm(x)
        

        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))

# if __name__ == '__main__':
#     enc = EncoderV1(lr=0, n_layers=1, weight_decay=0, latent_size=256,
#                     device=torch.device('cuda:2'), target=False, checkpoint_dir=None)

#     import numpy as np 
#     a = np.full((256, 3, 3, 224, 224), 100, dtype=np.int8)

#     a_torch = torch.from_numpy(a).to(torch.device('cuda:2'))

#     print(enc(a_torch).size())
#     print(get_n_params_network(enc))
    
