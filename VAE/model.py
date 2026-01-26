import torch
import torch.nn as nn
import torch.nn.functional as F

# input img -> hidden dim -> mean, std -> parameterization -> decoder -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20): # latent space dim
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x): # q_phi(z/x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h) # z ~ N(mu, sigma)
        return mu, sigma
    
    def decode(self, z): # q_theta(x/z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma* epsilon # reparameterization
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma
       