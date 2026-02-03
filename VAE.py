import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 784
h_dim = 200
z_dim = 20
epochs = 10
batch_size = 32
lr = 1e-3

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
       
dataset = datasets.MNIST(root = 'dataset/', train = True, transform=transforms.ToTensor(), download= True)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle= True)
model = VariationalAutoEncoder(input_dim, h_dim, z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss(reduction='sum') # matches ELBO

for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader))
    for i,(x, _) in loop:
        # forward pass
        x = x.to(device).view(x.shape[0], input_dim) # [z, 1, 28, 28]->[z, 784]
        x_reconstructed, mu, sigma = model(x)
        
        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) # KL(q(z/x)||N(0,I))
        ### pushes latent distribution towards standard gaussian
        
        #backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    
model = model.to(device)

def inference(digit, num_ex = 1):
    images = []
    i = 0
    for x, y in dataset:
        if y == i:
            images.append(x)
            i += 1
        if i == 10:
            break
    
    encoding_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encoding_digit.append((mu, sigma))
    
    mu, sigma = encoding_digit[digit]
    for ex in range(num_ex):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma* epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f'{digit}.png')
        
for i in range(10):
    inference(i, num_ex= 1)