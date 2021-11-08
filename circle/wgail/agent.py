import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from collections import deque
from env import EnvCircle
from model import ActorCritic, Discriminator
from torch.distributions import Categorical
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Agent:
    def __init__(self):
        self.lr_actor = 0.001
        self.lr_discrim = 0.003
        self.gamma = 0.99
        self.actor = ActorCritic(2,4, 64)
        self.discriminator = Discriminator(3,32)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_discrim = optim.RMSprop(self.discriminator.parameters(), lr=self.lr_discrim)
        # self.discrim_criterion = nn.BCELoss()
        self.eps = np.finfo(np.float32).eps.item()
        self.n_ppo_update = 0

    def get_state(self):
        pass

    def get_action(self,state):
        state0 = torch.tensor(state,dtype=torch.float)
        probs, state_value = self.actor(state0)
        m = Categorical(probs)
        action = m.sample()
        # print('action',action.item())
        logAction = m.log_prob(action).reshape(-1)
        return action.item(),logAction,state_value,m

    def get_reward(self, state, action):
        state_action = torch.FloatTensor(np.concatenate([state, [action]])).to(device)
        return -np.log(self.discriminator(state_action).cpu().data.numpy())

    def compute_gae(self,next_value, rewards, masks, values):
        masks = [torch.from_numpy(item).float() for item in masks]
        gamma=0.99
        tau=0.95
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self,mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]   

    def ppo_update(self,ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for i in range(ppo_epochs):
            self.n_ppo_update +=i
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                
                dist, value = self.actor(state)
                dist = Categorical(dist)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
                writer.add_scalar("Critic", critic_loss, self.n_ppo_update)
                writer.add_scalar("Actor", actor_loss,self.n_ppo_update)
                writer.add_scalar("Total Loss", loss,self.n_ppo_update)

                self.optimizer_actor.zero_grad()
                loss.backward()
                self.optimizer_actor.step()


def train():
    num_steps = 360
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    env = EnvCircle()

    i_update = 0

    # expert_states_concat = pd.read_csv('traj.csv')
    expert_states_concat = pd.read_csv('traj_up_left.csv')
   
    while i_update<500:
        i_update+=1
        env.update_num(i_update)
        print('i_update',i_update)
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0
        env.reset()
        state = [0,0]

        for t in range(num_steps):
            final_move,log_prob_act,state_value,dist = agent.get_action(state)
            reward = agent.get_reward(state,final_move)
            state_new, _, done, _ = env.step(final_move)

            entropy += dist.entropy().mean()

            log_probs.append(log_prob_act.unsqueeze(0))
            values.append(state_value.unsqueeze(0))
            rewards.append(torch.FloatTensor(reward).to(device))
            masks.append(np.array([(1 - done)]))

            state = torch.FloatTensor(state)
            states.append(state.unsqueeze(0))
            actions.append(torch.FloatTensor(np.array([final_move])).unsqueeze(0))
            state = state_new
            
            if done:
                env.reset()

        _,_,next_value,_= agent.get_action(state)
        returns = agent.compute_gae(next_value, rewards, masks, values)

   
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values).detach()
        
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        log_probs = torch.cat(log_probs).detach()
        advantage = returns - values
    
        if i_update % 5 == 0:
            agent.ppo_update(4, 64, states, actions, log_probs, returns, advantage)

        expert_state_action = expert_states_concat.values[np.random.randint(0, expert_states_concat.shape[0], 2 * num_steps * 1), :]
        expert_state_action = torch.FloatTensor(expert_state_action).to(device)
        state_action = torch.cat([states, actions], 1)

        fake = agent.discriminator(state_action)
        real = agent.discriminator(expert_state_action)
        agent.optimizer_discrim.zero_grad()
        # discrim_loss = agent.discrim_criterion(fake, torch.ones((states.shape[0], 1)).to(device)) + \
        #         agent.discrim_criterion(real, torch.zeros((expert_state_action.size(0), 1)).to(device))
        discrim_loss = -torch.mean(real) + torch.mean(fake)

        writer.add_scalar("discrim_loss", discrim_loss, i_update)

        discrim_loss.backward()
        agent.optimizer_discrim.step()

        # weight clipping
        for p in agent.discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        torch.save(agent.actor.state_dict(), './model-GAIL')


if __name__ == '__main__':
    train()