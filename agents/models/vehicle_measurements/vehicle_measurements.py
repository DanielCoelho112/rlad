import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init

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


