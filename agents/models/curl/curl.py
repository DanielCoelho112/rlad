import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.networks import update_target_network

class CURL(nn.Module):
    def __init__(self, lr, latent_dim, projection_dim, bilinear_product, device):
        super(CURL, self).__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        self.use_bilinear_product = bilinear_product
        
        
        self.use_projection = latent_dim != projection_dim
        
        self.W = nn.Parameter(torch.rand(projection_dim, projection_dim))
        
                    
        if not self.use_projection:
            
            self.projection = None
            self.projection_target = None 
            self.optimizer = torch.optim.Adam([self.W], lr=lr)
        
        else:
            self.projection = nn.Sequential(
                nn.Linear(latent_dim, projection_dim))
            
            self.projection_target = nn.Sequential(
                nn.Linear(latent_dim, projection_dim))
    
            update_target_network(target=self.projection_target, source=self.projection, tau=1)
            
            params = [self.W] + list(self.projection.parameters())
            
            self.optimizer = torch.optim.Adam(params, lr=lr)
            
        
        self.loss = nn.CrossEntropyLoss()
        
        self.to(self.device)
    
    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        
        if self.use_projection:
            z_a = self.projection(z_a)
            with torch.no_grad():
                z_pos = self.projection_target(z_pos)
        
        if self.use_bilinear_product:
            Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B).
            logits = torch.matmul(z_a, Wz)  # (B,B).
        else:
            # cosine similarity.
            max_z_pos, _ = torch.max(z_pos, dim=1)
            max_z_a, _ = torch.max(z_a, dim=1)
            logits = torch.matmul(z_pos, z_a.T) / (max_z_pos * max_z_a)
            
        logits = logits - torch.max(logits, 1)[0][:, None] # substracting max for stability.
        return logits