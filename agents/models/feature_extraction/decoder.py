import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np

from utilities.networks import weights_init

class DecoderV0(nn.Module):
    def __init__(self, lr, weight_decay, latent_size, device, checkpoint_dir):
        super(DecoderV0, self).__init__()

        self.latent_size = latent_size
        self.device = device

        self.checkpoint_file = f"{checkpoint_dir}/weights/decoder.pt"
        self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/decoder.pt"

        self.fc = nn.Linear(latent_size, 2304)

        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.conv2 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.conv3 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1)

        self.apply(weights_init)
        
        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)

    def forward(self, x):

        x = F.relu(self.fc(x))

        deconv = x.view(-1, 64, 6, 6)
        x1 = F.relu(self.conv1(deconv))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        obs = self.conv5(x4)

        return obs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


# def get_n_params(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp
# if __name__ == '__main__':
#     dec = DecoderV0(lr=0, weight_decay=0, pretrained=False,
#                     latent_size=256, device=None, checkpoint_dir=None)

#     # a = np.full((10,3,250,250), 100, dtype=np.int8)
#     a = np.full((10, 256), 0.5, dtype=np.float32)

#     a_torch = torch.from_numpy(a)

#     print(dec(a_torch).size())
