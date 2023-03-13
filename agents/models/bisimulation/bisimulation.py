import random
import torch
import torch.nn as nn


class DeterministicTransitionModel(nn.Module):

    def __init__(self, state_size, device, checkpoint_dir, n_actions=2, layer_width=512):
        super().__init__()
        
        self.checkpoint_file = f"{checkpoint_dir}/weights/transition_model.pt"
        
        self.fc = nn. Linear(state_size + n_actions, layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, state_size)
        self.to(device)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class RewardDecoder(nn.Module):
    def __init__(self, state_size, device, checkpoint_dir):
        super().__init__()
        self.checkpoint_file = f"{checkpoint_dir}/weights/reward_decoder.pt"
        self.decoder = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1))
        self.to(device)
        
    def forward(self, x):
        return self.decoder(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    