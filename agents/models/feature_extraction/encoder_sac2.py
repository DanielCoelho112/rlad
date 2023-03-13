import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np

from torchvision import transforms

from utilities.networks import weights_init, get_n_params_network



class EncoderV3(nn.Module):
    def __init__(self, lr, image_size, image_size_cropped, weight_decay, pretrained, latent_size, use_tanh, device, target, checkpoint_dir):
        super(EncoderV3, self).__init__()

        self.image_size = image_size
        self.image_size_cropped = image_size_cropped
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


        self.transforms = self.get_transforms()

        self.apply(weights_init)

        self.optimizer = optimizer.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.to(self.device)

    def get_transforms(self):
        trans = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size_cropped)])

        return trans

    def forward(self, x):
        x = x / 255. - 0.5
        x = x.to(torch.float32)
        x = self.transforms(x)
        
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
    enc = EncoderV3(lr=0, weight_decay=0, image_size=232, image_size_cropped=224, pretrained=False, use_tanh=True,
                    latent_size=256, device=torch.device('cuda:0'), target=False, checkpoint_dir=None)


    a = np.full((64, 3, 260, 260), 100, dtype=np.int8)

    a_torch = torch.from_numpy(a).to(torch.device('cuda:0'))

    print(enc(a_torch).size())
