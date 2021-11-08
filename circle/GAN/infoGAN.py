import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import random

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = torch.cat([x, label], dim=1)
        #print('dis',x)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            # nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x,label):
        # print(x)
        # print(label)
        x = torch.cat([x, label], dim=1)
        return self.gen(x)


class Posterior(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        #x = torch.cat([x, label], dim=1)
        #print('dis',x)
        return self.disc(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 #3e-4
lr_c = 3e-4
lr_dis = 3e-4 #3e-4
z_dim = 16
image_dim = 3
batch_size = 32
num_epochs = 150

disc = Discriminator(image_dim).to(device)
pos = Posterior(image_dim).to(device)
gen = Generator(z_dim+1, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
expert_states_concat_01 = pd.read_csv('traj_1.csv')
expert_states_concat_02 = pd.read_csv('traj_2.csv')
expert_states_concat = pd.concat([expert_states_concat_02,expert_states_concat_01])

a = torch.zeros(360)
b = torch.ones(360)
labels  = torch.cat([a, b], dim=0)
# expert_states_concat = pd.read_csv('traj_up_left.csv')

opt_disc = optim.Adam(disc.parameters(), lr=lr_dis)
opt_post = optim.Adam(pos.parameters(), lr=lr_c)
opt_gen = optim.Adam([{'params':gen.parameters()}, {'params':pos.parameters()}], lr=lr)
#opt_gen = optim.Adam(gen.parameters(), lr=lr)

# criterion_disc = nn.MSELoss()
criterion_disc = nn.BCEWithLogitsLoss()
criterion_post = nn.BCELoss()

writer_dis = SummaryWriter(f"logs/disc")
writer_gen = SummaryWriter(f"logs/gen")
writer_pos = SummaryWriter(f"logs/pos")

for epoch in range(num_epochs):
    for i in range(200):
        print('count',i)
        random_idx = np.random.randint(0, expert_states_concat.shape[0], 1 * batch_size * 1)
        real = expert_states_concat.values[random_idx, :]
        #label = labels[random_idx].reshape([batch_size,1])
        real = torch.FloatTensor(real).to(device)

        # real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        # print(noise.shape)
        # print(label.shape)
        #fake = gen(noise,label)
        #random_label = torch.randint(0, 2, (batch_size,)).reshape([batch_size,1])
        random_label_pos_0 = torch.randint(0, 2, (batch_size,))
        random_label_pos = random_label_pos_0.reshape([batch_size,1])
        fake = gen(noise,random_label_pos)

        disc_real = disc(real).view(-1)
        lossD_real = criterion_disc(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion_disc(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        writer_dis.add_scalar("discrim_loss", lossD, epoch*i+i)
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Posterior
        # lambda_cat = 1
        # opt_post.zero_grad()
        # # random_label_pos_0 = torch.randint(0, 2, (batch_size,))
        # # random_label_pos = random_label_pos_0.reshape([batch_size,1])
        gt_labels = Variable(torch.LongTensor(random_label_pos_0), requires_grad=False)

        # fake_pos = gen(noise,random_label_pos)
        # fake = gen(noise,random_label_pos)
        # pred_label = pos(fake.detach()).view(-1)
        # # print(pred_label)
        # # print(random_label_pos_0)
        # info_loss = lambda_cat * criterion_post(pred_label, gt_labels.float())
        # writer_pos.add_scalar("info_loss", info_loss, epoch*i+i)

        # info_loss.backward()
        # opt_post.step()

        lambda_cat = 1
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        c_label = pos(fake).view(-1)
        print('c_labe',c_label)
        info_loss_gen = lambda_cat * criterion_post(c_label, gt_labels.float())
        lossG = criterion_disc(output, torch.ones_like(output))
        lossG_total= lossG - info_loss_gen
        writer_pos.add_scalar("info_loss_gen", info_loss_gen, epoch*i+i)
        writer_gen.add_scalar("lossG", lossG, epoch*i+i)
        writer_gen.add_scalar("lossG_total", lossG_total, epoch*i+i)
        gen.zero_grad()
        lossG_total.backward()
        opt_gen.step()


        with torch.no_grad():
            label_t = torch.zeros(32).reshape([batch_size,1])
            fake = gen(fixed_noise,label_t)
            print('zero',fake)
            
            label_t = torch.ones(32).reshape([batch_size,1])
            fake_1 = gen(fixed_noise,label_t)
            print('one',fake_1)
  