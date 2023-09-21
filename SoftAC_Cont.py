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
import matplotlib as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# Original paper
# https://arxiv.org/pdf/1801.01290.pdf

#  For entropy regularization
# https://docs.cleanrl.dev/rl-algorithms/sac/#explanation-of-the-logged-metrics



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
        self.action_scale = torch.FloatTensor((high - low) / 2.).to(device)
        self.action_bias = torch.FloatTensor((high + low) / 2.).to(device)
    
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

        self.apply(weights_init_) # OpReplayMemorytional


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
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.memory_capacity

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class Sac_agent:
    def __init__(self, state_size, action_size, hidden_dim, high, low, memory_capacity, batch_size,
                 gamma, tau,num_updates, policy_freq, alpha):
        
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
        self.value = ValueNetwork(state_size, hidden_dim).to(device)
        self.value_optim =optim.Adam(self.value.parameters(), lr=1e-4)
        self.target_value = ValueNetwork(state_size, hidden_dim).to(device)
        
        # copy weights
        self.hard_update(self.target_value, self.value)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = ReplayMemory(memory_capacity, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.iters = 0
        self.policy_freq=policy_freq
        
        ## For Dynamic Adjustment of the Parameter alpha (entropy coefficient) according to Gaussion policy (stochastic):
        self.target_entropy = -float(self.action_size) # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2, -11 for Reacher-v2)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = alpha
        
        
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
                         
            value_current=self.value(state)
            value_next=self.target_value(next_state)
            act_next, logp_next = self.actor.sample(next_state)
                
            ## Compute targets
            Q_target_main = reward + self.gamma*mask*value_next # Eq.8 of the original paper

            ## Update Value Network
            Q_target1, Q_target2 = self.critic_target(next_state, act_next) 
            min_Q = torch.min(Q_target1, Q_target2)
            value_difference = min_Q - logp_next # substract min Q value from the policy's log probability of slelecting that action
            value_loss = 0.5 * F.mse_loss(value_current, value_difference) # Eq.5 from the paper
            # Gradient steps 
            self.value_optim.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_optim.step()
            
            ## Update Critic Network       
            critic_1, critic_2 = self.critic(state, action)
            critic_loss1 = 0.5*F.mse_loss(critic_1, Q_target_main) # Eq. 7 of the original paper
            critic_loss2 = 0.5* F.mse_loss(critic_2, Q_target_main) # Eq. 7 of the original paper
            total_critic_loss=critic_loss1+ critic_loss2 
            # Gradient steps
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward() 
            self.critic_optimizer.step() 

            ## Update Actor Network with Entropy Regularized (look at the link for entropy regularization)
            act_pi, log_pi = self.actor.sample(state) # Reparameterize sampling
            Q1_pi, Q2_pi = self.critic(state, act_pi)
            min_Q_pi = torch.min(Q1_pi, Q2_pi)
            actor_loss =-(min_Q_pi-self.alpha*log_pi ).mean() # For minimization
            # Gradient steps
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            ## Dynamic adjustment of the Entropy Parameter alpha (look at the link for entropy regularization)
            alpha_loss = (-self.log_alpha * (log_pi.detach()) - self.log_alpha* self.target_entropy).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
            ## Soft Update Target Networks using Polyak Averaging
            if (self.iters % self.policy_freq == 0):         
                self.soft_update(self.critic_target, self.critic)
                self.soft_update(self.target_value, self.value)
        
    def act(self, state):
        state =  torch.tensor(state).unsqueeze(0).to(device).float()
        action, logp = self.actor.sample(state)
        return action.cpu().data.numpy()[0]
    
    def step(self):
        self.learn(self.memory.sample())
        
    def save(self):
        torch.save(self.actor.state_dict(), "pen_actor.pkl")
        torch.save(self.critic.state_dict(), "pen_critic.pkl")
        

def sac(episodes):
    agent = Sac_agent(state_size = state_size, action_size = action_size, hidden_dim = hidden_dim, high = high, low = low, 
                  memory_capacity = memory_capacity, batch_size = batch_size, gamma = gamma, tau = tau, 
                  num_updates = num_updates, policy_freq =policy_freq, alpha = entropy_coef)
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
            agent.iters=episode_steps
            if i < 10: # To increase exploration
                action = env.action_space.sample() # to sample the random actions by randomly
            else:
                action = agent.act(state) # to sample the actions by Gaussian 
            next_state, reward, done, _ = env.step(action)
            
             # Ignore the "done" signal if it comes from hitting the time horizon.
            if episode_steps == env._max_episode_steps: # if the current episode has reached its maximum allowed steps
                mask = 1
            else:
                mask = float(not done)
            
            if (len(agent.memory) >= agent.memory.batch_size): 
                agent.step()
            total_reward += reward
            state = next_state
            print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}")
            agent.memory.push((state, action, reward, next_state, mask))
            env.render()
        
        reward_list.append(total_reward)
        avg_score_deque.append(total_reward)
        mean = np.mean(avg_score_deque)
        avg_scores_list.append(mean)
        
    plt.plot(episode_steps,reward_list)
    agent.save()
    print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}, max reward: {np.max(reward_list)}")
    
                
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


batch_size=20 # size that will be sampled from the replay memory that has maximum of "memory_capacity"
memory_capacity = 200 # 2000, maximum size of the memory
gamma = 0.99            
tau = 0.005               
num_of_train_episodes = 1500
num_updates = 1 # how many times you want to update the networks in each episode
policy_freq= 2 # lower value more probability to soft update,  policy frequency for soft update of the target network borrowed by TD3 algorithm
entropy_coef = 0.2 # For entropy regularization
num_of_test_episodes=200
hidden_dim=256

# Traning agent
reward, avg_reward = sac(num_of_train_episodes)



# Testing
new_env = make("Pendulum-v0")
best_actor = Actor(state_size, action_size, hidden_dim = hidden_dim, high = high, low = low)
best_actor.load_state_dict(torch.load("pen_actor.pkl"))        
best_actor.to(device) 
reward_test = []
for i in range(num_of_test_episodes):
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


