from gym import make
#import pybullet_envs
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
from collections import deque
from torch.distributions.normal import Normal



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Original paper
# https://arxiv.org/pdf/1801.01290.pdf



# Use Xavier initialization for the weights and initializes the biases to zero for linear layers.
# It sets the weights to values drawn from a Gaussian distribution with mean 0 and variance
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class ValueNetwork(nn.Module): # state-Value network
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # Optional


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# stocastic policy
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, high, low):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_size)
        self.log_std = nn.Linear(hidden_dim, action_size)
        
        self.high = torch.tensor(high).to(device)
        self.low = torch.tensor(low).to(device)
        
        self.apply(weights_init_) # Optional
        
        # Action rescaling
        self.action_scale = torch.FloatTensor(
                (high - low) / 2.).to(device)
        self.action_bias = torch.FloatTensor(
                (high + low) / 2.).to(device)
    
    def forward(self, state):
        log_std_min=-20
        log_std_max=2
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        m = self.mean(x)
        s = self.log_std(x)
        s = torch.clamp(s, min = log_std_min, max = log_std_max)
        return m, s
    
    def sample(self, state):
        noise=1e-6
        m, s = self.forward(state) 
        std = s.exp()
        normal = Normal(m, std)
        
        
        ## Reparameterization (https://spinningup.openai.com/en/latest/algorithms/sac.html)
        # There are two sample functions in normal distributions one gives you normal sample ( .sample() ),
        # other one gives you a sample + some noise ( .rsample() )
        a = normal.rsample() # This is for the reparamitization
        tanh = torch.tanh(a)
        action = tanh * self.action_scale + self.action_bias
        
        logp = normal.log_prob(a)
        # Comes from the appendix C of the original paper for scaling of the action:
        logp =logp-torch.log(self.action_scale * (1 - tanh.pow(2)) + noise)
        logp = logp.sum(1, keepdim=True)
        
        return action, logp


# Action-Value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # Critic-1: Q1 
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Critic-2: Q2 
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # Optional


    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(state_action))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(state_action))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2
    
# Buffer
class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class Sac_agent:
    def __init__(self, state_size, action_size, hidden_dim, high, low, buffer_size, batch_size,
                 gamma, tau,num_updates, update_rate, alpha):
        
         # Actor Network 
        self.actor = Actor(state_size, action_size,hidden_dim, high, low).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        

        # Critic Network and Target Network
        self.critic = Critic(state_size, action_size, hidden_dim).to(device)   
        
        self.critic_target = Critic(state_size, action_size, hidden_dim).to(device)        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
        # copy weights
        self.hard_update(self.critic_target, self.critic)
        
        # Value network and Target Network
        self.value = ValueNetwork(state_size, hidden_dim).to(self.device)
        self.value_optim =optim.Adam(self.value.parameters(), lr=1e-4)
        self.target_value = ValueNetwork(state_size, hidden_dim).to(self.device)
        
        # copy weights
        self.hard_update(self.target_value, self.value)
        
        self.state_size = state_size
        self.action_size = action_size
        
        # According to Gaussion policy (stochastic),
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2, -11 for Reacher-v2)
        self.target_entropy = -float(self.action_size)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        
        self.memory = ReplayMemory(buffer_size, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.update_rate = update_rate
        self.alpha = alpha
        
        self.iters = 0
        
     
        
    def hard_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)
            
    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            
    def learn(self, batch):
        for _ in range(self.num_updates):                
            state, action, reward, next_state, mask = batch

            state = torch.tensor(state).to(device).float()
            next_state = torch.tensor(next_state).to(device).float()
            reward = torch.tensor(reward).to(device).float().unsqueeze(1)
            action = torch.tensor(action).to(device).float()
            mask = torch.tensor(mask).to(device).int().unsqueeze(1)
            
               
            value=self.value(state)
            value_target=self.target_value(next_state)


            # compute target action
            with torch.no_grad():
                act_next, logp_next = self.actor.sample(next_state)
                
                # compute targets
                Q_target1, Q_target2 = self.critic_target(next_state, act_next) 
                min_Q = torch.min(Q_target1, Q_target2)
                Q_target = reward + self.gamma*mask*(min_Q - self.alpha*logp_next) # Eq.8 of the original paper

            # update value
            self.value_optim.zero_grad()
            value_difference = min_Q - logp_next
            value_loss = 0.5 * F.mse_loss(value, value_difference)
            # gradient steps (KALDIM)
            value_loss.backward(retain_graph=True)
            self.value_optim.step()
            
            # update critic       
            critic_1, critic_2 = self.critic(state, action)
            critic_loss1 = F.mse_loss(critic_1, Q_target)
            critic_loss2 = F.mse_loss(critic_2, Q_target)       

            # update actor 
            pi, log_pi = self.actor.sample(state)
            Q1_pi, Q2_pi = self.critic(state, pi)
            min_Q_pi = torch.min(Q1_pi, Q2_pi)
            actor_loss = (self.alpha*log_pi - min_Q_pi).mean()
            
            #gradient steps
            self.critic_optimizer.zero_grad()
            critic_loss1.backward()          
            self.critic_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss2.backward()          
            self.critic_optimizer.step()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()            
            self.actor_optimizer.step()
            
            # update alpha
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            # update critic_targe
            self.soft_update(self.critic_target, self.critic)
        
    def act(self, state):
        state =  torch.tensor(state).unsqueeze(0).to(device).float()
        action, logp = self.actor.sample(state)
        return action.cpu().data.numpy()[0]
    
    def step(self, state, action, reward, next_state, mask):
        self.iters += 1
        self.memory.push((state, action, reward, next_state, mask))
        if ( len(self.memory) >= self.memory.batch_size ) and ( self.iters % self.update_rate == 0 ):
            self.learn(self.memory.sample())
        
    def save(self):
        torch.save(self.actor.state_dict(), "ant_actor.pkl")
        torch.save(self.critic.state_dict(), "ant_critic.pkl")
        
    def test(self):
        new_env = make("AntBulletEnv-v0")
        new_env.seed(9)
        reward = []
        for i in range(10):
            state = new_env.reset()
            local_reward = 0
            done = False
            while not done:
                action = self.act(state)
                state, r, done, _ = new_env.step(action)
                local_reward += r
            reward.append(local_reward)
        return reward
    


def sac(episodes):
    agent = Sac_agent(state_size = state_size, action_size = action_size, hidden_dim = 256, high = high, low = low, 
                  buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, gamma = GAMMA, tau = TAU, 
                  num_updates = NUM_UPDATES, update_rate = UPDATE_RATE, alpha = ENTROPY_COEFFICIENT)
    time_start = time.time()
    reward_list = []
    avg_score_deque = deque(maxlen = 100)
    avg_scores_list = []
    mean_reward = -20000
    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps+=1    
            if i < 10:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            
            agent.step(state, action, reward, next_state, mask)
            total_reward += reward
            state = next_state
            
        reward_list.append(total_reward)
        avg_score_deque.append(total_reward)
        mean = np.mean(avg_score_deque)
        avg_scores_list.append(mean)
        
        if total_reward > 2000:
            r = agent.test()
            local_mean = np.mean(r)
            print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}, max reward: {np.max(r)}, average reward on test: {local_mean}")
            if local_mean > mean_reward:
                mean_reward = local_mean
                agent.save()
                print("Saved")
        elif (i+1) % 1 == 0:
            s =  (int)(time.time() - time_start)            
            print("Ep.: {}, Ep.Steps: {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}".\
            format(i+1, episode_steps, total_reward, mean, \
                  s//3600, s%3600//60, s%60))
            
    return reward_list, avg_scores_list



# Environment
env = make("Pendulum-v0")
np.random.seed(0)
env.seed(0)

action_size = env.action_space.shape[0]
print(f'size of each action = {action_size}')
state_size = env.observation_space.shape[0]
print(f'size of state = {state_size}')
low = env.action_space.low
high = env.action_space.high
print(f'low of each action = {low}')
print(f'high of each action = {high}')


BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99            
TAU = 0.005               
EPISODES = 1500
NUM_UPDATES = 1
UPDATE_RATE = 1
ENTROPY_COEFFICIENT = 0.2


# Traning agent

reward, avg_reward = sac(EPISODES)
best_actor = Actor(state_size, action_size, hidden_dim = 256, high = high, low = low)
best_actor.load_state_dict(torch.load("ant_actor.pkl"))        
best_actor.to(device) 


new_env = make("Pendulum-v0")
#new_env.seed(9)
reward_test = []
for i in range(100):
    state = new_env.reset()
    local_reward = 0
    done = False
    while not done:
        state =  torch.tensor(state).to(device).float()
        action,logp = best_actor(state)        
        action = action.cpu().data.numpy()
        state, r, done, _ = new_env.step(action)
        local_reward += r
    reward_test.append(local_reward)


import plotly.graph_objects as go
x = np.array(range(len(reward_test)))
m = np.mean(reward_test)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=reward_test, name='test reward',
                                 line=dict(color="green", width=1)))

fig.add_trace(go.Scatter(x=x, y=[m]*len(reward_test), name='average reward',
                                 line=dict(color="red", width=1)))
    
fig.update_layout(title="SAC",
                           xaxis_title= "test",
                           yaxis_title= "reward")
fig.show()

print("average reward:", m)


