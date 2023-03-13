import torch
import torch.nn as nn
import torch.optim as optimizer

from importlib import import_module

class SemanticSegmentationV1():
    def __init__(self, entrypoint, image_encoder, lr, n_class, image_latent_size, weight_decay, device, checkpoint_dir):
        super(SemanticSegmentationV1, self).__init__()

        self.device = device 
        
        module_str, class_str = entrypoint.split(
            ':')
        _Class = getattr(import_module(module_str), class_str)
        self.decoder = _Class(lr=lr, weight_decay=weight_decay, latent_size=image_latent_size, device=device, n_class=n_class, checkpoint_dir=checkpoint_dir)
        
        self.loss = nn.CrossEntropyLoss()   
    
    def zero_grad(self):
        self.decoder.optimizer.zero_grad(set_to_none=True)
    
    def step(self):
        self.decoder.optimizer.step()
        
    def compute_loss(self, raw_state_batch, latent_image):
        labels = raw_state_batch['semantic_image'].squeeze(1) # 256x224x224
        labels = labels.type(torch.LongTensor).to(self.device)
        
        segmented_image = self.decoder(latent_image) # 256x6x224x224
        
        loss = self.loss(segmented_image, labels)

        return loss
        
    def train(self):
        self.decoder.train()

    def eval(self):
        self.decoder.eval()

    def save_checkpoint(self):
        self.decoder.save_checkpoint()
        

    def load_checkpoint(self):
        self.decoder.load_checkpoint()



class SemanticSegmentationV0():
    def __init__(self, entrypoint, image_encoder, lr, n_class, image_latent_size, weight_decay, device, checkpoint_dir):
        super(SemanticSegmentationV0, self).__init__()

        self.device = device 
        
        module_str, class_str = entrypoint.split(
            ':')
        _Class = getattr(import_module(module_str), class_str)
        self.decoder = _Class(lr=lr, weight_decay=weight_decay, latent_size=image_latent_size, device=device, n_class=n_class, checkpoint_dir=checkpoint_dir)
        
        self.optimizer_encoder = optimizer.Adam(
            image_encoder.parameters(), lr=lr, weight_decay=weight_decay)    
        
        self.optimizer_encoder_checkpoint_file = f"{checkpoint_dir}/weights/optimizers/semantic_encoder.pt"
        
        self.loss = nn.CrossEntropyLoss()   
    
    def update(self, raw_state_batch, image_encoder):
        labels = raw_state_batch['semantic_image'].squeeze(1) # 256x224x224
        labels = labels.type(torch.LongTensor).to(self.device)
        
        
        latent_image = image_encoder(raw_state_batch['central_rgb']) #256x256
        
        segmented_image = self.decoder(latent_image) # 256x6x224x224
        
        self.optimizer_encoder.zero_grad(set_to_none=True)
        self.decoder.optimizer.zero_grad(set_to_none=True)
        
        loss = self.loss(segmented_image, labels)
        
        loss.backward()

        self.optimizer_encoder.step()
        self.decoder.optimizer.step()
        
    def train(self):
        self.decoder.train()

    def eval(self):
        self.decoder.eval()

    def save_checkpoint(self):
        self.decoder.save_checkpoint()
        torch.save(self.optimizer_encoder.state_dict(), self.optimizer_encoder_checkpoint_file)
        

    def load_checkpoint(self):
        self.decoder.load_checkpoint()
        self.optimizer_encoder.load_state_dict(torch.load(self.optimizer_encoder_checkpoint_file))


# if __name__ == '__main__':
#     # enc = WaypointLinearEncoder(lr=0, num_waypoints=10, fc_dims=20, out_dims=32,weight_decay=0, device=torch.device('cuda:0'), target=False, checkpoint_dir=None)

#     # import numpy as np 
#     # a = np.full((1, 20), 0.5, dtype=np.float32)

#     # a_torch = torch.from_numpy(a).to(torch.device('cuda:0'))

#     # print(enc(a_torch).size())
#     # print(get_n_params_network(enc))
    
    # light_state = torch.randint(low=0, high=10, size=(10,4))
    # mask = torch.randint(low=0, high=2, size=(10,)).to(torch.bool)
    # print(mask)
    
    # print(light_state)
    
    # new = light_state[mask]
    
    # print(new)
    
    
#     light_state_copy = light_state.clone()
    
#     print(light_state)
#     print(light_state)

#     idx_on = light_state > 0 
    
#     print(idx_on.nelement())
    
#     # print(idx_on)
#     # summary(enc, (3,2))
    
#     light_state[idx_on] = 1
    
#     print(light_state)
    
#     new_ligh_state = light_state_copy[idx_on].reshape(-1, 1)
    
#     n_tl_on = new_ligh_state.nelement()
    
#     print(n_tl_on)
    
#     exit()
    
#     print(new_ligh_state.size())
#     print(new_ligh_state)
#     print(new_ligh_state -1)