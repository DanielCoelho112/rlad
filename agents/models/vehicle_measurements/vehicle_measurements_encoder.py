import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init, get_n_params_network
from torchsummary import summary

class VehicleMeasurementsEncoder(nn.Module):
    def __init__(self, lr, num_inputs, fc_dims, out_dims, weight_decay, device, checkpoint_dir, target):
        super(VehicleMeasurementsEncoder, self).__init__()

        self.device = device
        self.out_dim = out_dims

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/vehicle_measurements_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/vehicle_measurements_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/vehicle_measurements_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/vehicle_measurements_encoder.pt"

        self.encoder = nn.Sequential(nn.Linear(num_inputs, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, out_dims))
        
        
        
        
        self.norm = nn.Sequential(nn.LayerNorm(out_dims), nn.Tanh())
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):

        x = self.encoder(x)
        
        out = self.norm(x)
        
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))


class VehicleMeasurementsEncoderV0(nn.Module):
    def __init__(self, lr, num_inputs, fc_dims, out_dims, weight_decay, device, checkpoint_dir, target):
        super(VehicleMeasurementsEncoderV0, self).__init__()

        self.device = device
        self.n_layers = 1
        self.out_dim = out_dims

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/vehicle_measurements_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/vehicle_measurements_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/vehicle_measurements_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/vehicle_measurements_encoder.pt"

        self.encoder = nn.Sequential(nn.Linear(num_inputs, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, out_dims))
        
        
        self.lstm = nn.LSTM(input_size=out_dims, hidden_size=out_dims, num_layers=self.n_layers, bidirectional=False, batch_first=True)
        
        
        self.norm = nn.Sequential(nn.LayerNorm(out_dims), nn.Tanh())
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        # x: BxTxL
        s = x.size()
        x = x.view(-1, s[-1]) # x: (B*T)x(C)x(H)x(W))
        x = self.encoder(x)
        x = x.view(s[0], s[1], -1)
    
        h0 = torch.zeros(self.n_layers, x.size(0), self.out_dim).to(self.device) 
        c0 = torch.zeros(self.n_layers, x.size(0), self.out_dim).to(self.device)
       
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

class VehicleMeasurementsEncoder_GRU(nn.Module):
    def __init__(self, lr, num_inputs, fc_dims, out_dims, weight_decay, device, checkpoint_dir, target):
        super(VehicleMeasurementsEncoder_GRU, self).__init__()

        self.device = device
        self.n_layers = 1
        self.out_dim = out_dims

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/vehicle_measurements_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/vehicle_measurements_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/vehicle_measurements_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/vehicle_measurements_encoder.pt"

        self.encoder = nn.Sequential(nn.Linear(num_inputs, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, out_dims))
        
        
        self.gru = nn.GRU(input_size=out_dims, hidden_size=out_dims, num_layers=self.n_layers, bidirectional=False, batch_first=True)
        
        
        self.norm = nn.Sequential(nn.LayerNorm(out_dims), nn.Tanh())
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        # x: BxTxL
        s = x.size()
        x = x.view(-1, s[-1]) # x: (B*T)x(C)x(H)x(W))
        x = self.encoder(x)
        x = x.view(s[0], s[1], -1)
    
        h0 = torch.zeros(self.n_layers, x.size(0), self.out_dim).to(self.device) 
       
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
#     enc = VehicleMeasurementsEncoderV1(lr=0, num_inputs=2, fc_dims=8, out_dims=16,weight_decay=0, device=torch.device('cuda:0'), target=False, checkpoint_dir=None)

#     import numpy as np 
#     a = np.full((10, 3 ,2), 0.5, dtype=np.float32)

#     a_torch = torch.from_numpy(a).to(torch.device('cuda:0'))

#     print(enc(a_torch).size())
#     print(get_n_params_network(enc))

    
#     # summary(enc, (3,2))