import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import numpy as np

from torchvision import transforms

from utilities.networks import weights_init, get_n_params_network



class EncoderV0(nn.Module):
    def __init__(self, lr, image_size, image_size_cropped, weight_decay, pretrained, latent_size, use_tanh, device, target, checkpoint_dir):
        super(EncoderV0, self).__init__()

        self.image_size = image_size
        self.image_size_cropped = image_size_cropped
        self.latent_size = latent_size
        self.device = device
        self.use_tanh = use_tanh

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder.pt"

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2)
        self.conv5 = nn.Conv2d(32, 64, 3, stride=2)

        self.fc = nn.Linear(2304, latent_size)
        self.lnorm = nn.LayerNorm(latent_size)

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
        x = x / 255.
        x = x.to(torch.float32)
        x = self.transforms(x)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        h = x5.view(x5.size(0), -1)

        out_linear = self.fc(h)
        out = self.lnorm(out_linear)
        
        if self.use_tanh:
            out = torch.tanh(out)

        return out

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        torch.save(self.optimizer.state_dict(), self.checkpoint_optimizer)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        self.optimizer.load_state_dict(torch.load(self.checkpoint_optimizer))


class EncoderV1(nn.Module):
    def __init__(self, lr, image_size, image_size_cropped, weight_decay, pretrained, latent_size, use_tanh, device, target, checkpoint_dir):
        super(EncoderV1, self).__init__()

        self.image_size = image_size
        self.image_size_cropped = image_size_cropped
        self.latent_size = latent_size
        self.device = device
        self.use_tanh = use_tanh

        if target:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder_target.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder_target.pt"
        else:
            self.checkpoint_file = f"{checkpoint_dir}/weights/encoder.pt"
            self.checkpoint_optimizer = f"{checkpoint_dir}/weights/optimizers/encoder.pt"

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc = nn.Linear(2304, latent_size)
        self.lnorm = nn.LayerNorm(latent_size)

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
        x = x / 255.
        x = x.to(torch.float32)
        x = self.transforms(x)

        x1 =self.conv1(x)
        x2 =self.conv2(x1)
        x3 =self.conv3(x2)
        x4 =self.conv4(x3)
        x5 =self.conv5(x4)

        h = x5.view(x5.size(0), -1)

        out_linear = self.fc(h)
        out = self.lnorm(out_linear)
        
        if self.use_tanh:
            out = torch.tanh(out)

        return out

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
if __name__ == '__main__':
    enc = EncoderV0(lr=0, weight_decay=0, image_size=232, image_size_cropped=224, pretrained=False, use_tanh=True,
                    latent_size=256, device=None, target=False, checkpoint_dir=None)

    print(get_n_params_network(enc))


    a = np.full((10, 3, 250, 250), 100, dtype=np.int8)

    a_torch = torch.from_numpy(a)

    print(enc(a_torch).size())
