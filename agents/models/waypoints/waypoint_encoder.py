import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import weights_init, get_n_params_network
from torchsummary import summary


class WaypointLinearEncoder(nn.Module):
    def __init__(self, lr, num_waypoints, fc_dims, out_dims, weight_decay, device, checkpoint_dir, target):
        super(WaypointLinearEncoder, self).__init__()

        self.device = device
        self.mode = 'linear'

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/waypoint_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/waypoint_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/waypoint_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/waypoint_encoder.pt"

        input_layer =num_waypoints * 2
        
        self.encoder = nn.Sequential(nn.Linear(input_layer, fc_dims),
                                     nn.ReLU(), 
                                     nn.Linear(fc_dims, out_dims),
                                     nn.LayerNorm(out_dims))
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        out = self.encoder(x)
            
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))
        
class WaypointConvEncoder(nn.Module):
    def __init__(self, lr, num_waypoints, fc_dims, out_dims, weight_decay, device, checkpoint_dir, target):
        super(WaypointConvEncoder, self).__init__()

        self.num_waypoints = num_waypoints
        self.device = device
        self.mode = 'conv'

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/waypoint_encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/waypoint_encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/waypoint_encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/waypoint_encoder.pt"

        
        self.conv = nn.Conv1d(2, fc_dims, 2)
        
        n_element = self.get_output_conv()
        
        self.linear1 = nn.Linear(n_element, fc_dims)
        self.linear2 = nn.Linear(fc_dims, out_dims)
        self.layer_norm = nn.LayerNorm(out_dims)
        self.tanh = nn.Tanh()
                                   
        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)

        
    def get_output_conv(self):
        waypoints = torch.zeros(size=(1,self.num_waypoints, 2))
        waypoints = waypoints.permute(0, 2, 1)
        return self.conv(waypoints).nelement()
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.layer_norm(x)
        x = self.tanh(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer, map_location=self.device))




if __name__ == '__main__':
    enc = WaypointConvEncoder(lr=0, num_waypoints=10, fc_dims=20, out_dims=32,weight_decay=0, device=torch.device('cuda:0'), target=False, checkpoint_dir=None)

    import numpy as np 
    a = np.full((1,10,2), 0.5, dtype=np.float32)

    a_torch = torch.from_numpy(a).to(torch.device('cuda:0'))

    print(enc(a_torch).size())
    print(get_n_params_network(enc))

    
    # summary(enc, (3,2))