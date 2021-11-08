import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x,label):
        x = torch.cat([x, label], dim=1)
        #print('dis',x)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            #nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x,label):
        # print(x)
        # print(label)
        x = torch.cat([x, label], dim=1)
        return self.gen(x)


class Posterior(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            #nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x,label):
        # print(x)
        # print(label)
        x = torch.cat([x, label], dim=1)
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 32
image_dim = 3
batch_size = 32
num_epochs = 150

disc = Discriminator(image_dim+1).to(device)
gen = Generator(z_dim+1, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
expert_states_concat_01 = pd.read_csv('traj.csv')
expert_states_concat_02 = pd.read_csv('traj_up_left.csv')
expert_states_concat = pd.concat([expert_states_concat_02,expert_states_concat_01])

a = torch.zeros(360)
b = torch.ones(360)
labels  = torch.cat([a, b], dim=0)
# expert_states_concat = pd.read_csv('traj_up_left.csv')

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for i in range(200):
        print('count',i)
        random_idx = np.random.randint(0, expert_states_concat.shape[0], 1 * batch_size * 1)
        real = expert_states_concat.values[random_idx, :]
        label = labels[random_idx].reshape([batch_size,1])
        real = torch.FloatTensor(real).to(device)

        # real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        # print(noise.shape)
        # print(label.shape)
        fake = gen(noise,label)
        disc_real = disc(real,label).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake,label).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake,label).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        with torch.no_grad():
            print('c=1')
            label_t = torch.ones(32).reshape([batch_size,1])
            fake = gen(fixed_noise,label_t)
            print(fake)

            print('c=0')
            label_t = torch.zeros(32).reshape([batch_size,1])
            fake = gen(fixed_noise,label_t)
            print(fake)
    