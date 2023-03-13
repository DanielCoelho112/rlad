import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F

from utilities.networks import get_n_params_network, create_resnet_basic_block


class DecoderSegmentedV0(nn.Module):
    def __init__(self, lr, latent_size, n_class, weight_decay, device, checkpoint_dir):
        super(DecoderSegmentedV0, self).__init__()

        self.device = device
        self.out_dim = 64 * 6 * 6


        self.checkpoint_file = f"{checkpoint_dir}/weights/image_segmented_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_segmented_decoder.pt"

        self.linear = nn.Sequential(nn.Linear(latent_size ,self.out_dim))

        self.convtranspose = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=1),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(64),
                                            nn.ConvTranspose2d(64, 64, 3, stride=2),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(64),
                                            nn.ConvTranspose2d(64, 64, 3, stride=2),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(64),
                                            nn.ConvTranspose2d(64, 32, 3, stride=2),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(32),
                                            nn.ConvTranspose2d(32, 32, 3, stride=2, output_padding=1),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(32))
                                            
        
        
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.view(-1, 64, 6, 6)
        
        x = self.convtranspose(x)
        out = self.classifier(x)
  
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


class DecoderSegmentedV0_without256(nn.Module):
    def __init__(self, lr, latent_size, n_class, weight_decay, device, checkpoint_dir):
        super(DecoderSegmentedV0_without256, self).__init__()

        self.device = device
        self.out_dim = 64 * 6 * 6


        self.checkpoint_file = f"{checkpoint_dir}/weights/image_segmented_decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/image_segmented_decoder.pt"


        self.convtranspose = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=1),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(64),
                                            nn.ConvTranspose2d(64, 64, 3, stride=2),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(64),
                                            nn.ConvTranspose2d(64, 64, 3, stride=2),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(64),
                                            nn.ConvTranspose2d(64, 32, 3, stride=2),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(32),
                                            nn.ConvTranspose2d(32, 32, 3, stride=2, output_padding=1),
                                            nn.ReLU(), 
                                            nn.BatchNorm2d(32))
                                            
        
        
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)


    def forward(self, x):
        x = x.view(-1, 64, 6, 6)
        
        x = self.convtranspose(x)
        out = self.classifier(x)
  
        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


# if __name__ == '__main__':
#     enc = DecoderSegmentedV0(lr=0, weight_decay=0, n_class=6, latent_size=256,
#                     device=torch.device('cuda:3'), checkpoint_dir=None)

#     import numpy as np 
#     a = np.full((256, 256), 0.5, dtype=np.float32)

#     a_torch = torch.from_numpy(a).to(torch.device('cuda:3'))

#     print(enc(a_torch).size())
#     print(get_n_params_network(enc))
