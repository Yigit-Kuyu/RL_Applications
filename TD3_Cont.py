import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# Deterministic policy
class ActorNet(nn.Module):
    def __init__(self, state_num, action_num, min_action, max_action):
        super(ActorNet, self).__init__()
        self.input = nn.Linear(state_num, 256)
        self.fc = nn.Linear(256, 512)
        self.output = nn.Linear(512, action_num)
        
        # Get the action interval for clipping
        self.min_action = min_action
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        x = F.sigmoid(self.output(x))
        action = (self.max_action - self.min_action) * x + self.min_action
        
        return action
    


class CriticNet(nn.Module):
    def __init__(self, state_num, action_num):
        super(CriticNet, self).__init__()
        # Critic network 1
        self.input1 = nn.Linear(state_num + action_num, 256)
        self.fc1 = nn.Linear(256, 512)
        self.output1 = nn.Linear(512, 1)
        
        # Critic network 2
        self.input2 = nn.Linear(state_num + action_num, 256)
        self.fc2 = nn.Linear(256, 512)
        self.output2 = nn.Linear(512, 1)
        
    def forward(self, x, u):
        # Critic network 1
        x1 = torch.cat([x, u], 1)
        x1 = F.relu(self.input1(x1))
        x1 = F.relu(self.fc1(x1))
        value1 = self.output1(x1)
        
        # Critic network 2
        x2 = torch.cat([x, u], 1)
        x2 = F.relu(self.input2(x2))
        x2 = F.relu(self.fc2(x2))
        value2 = self.output2(x2)
        
        return value1, value2
    
    def network1(self, x, u):
        # Critic network 1
        x1 = torch.cat([x, u], 1)
        x1 = F.relu(self.input1(x1))
        x1 = F.relu(self.fc1(x1))
        value = self.output1(x1)
        
        return value
    
class TD3():
    def __init__(self, env, memory_size=10000000, batch_size=64, gamma=0.95, learning_rate=1e-3, eps_min=0.05, eps_period=10000, tau=0.01, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        super(TD3, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.shape[0]
        self.action_max = float(env.action_space.high[0])
        self.action_min = float(env.action_space.low[0])
                
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor
        self.actor_net = ActorNet(self.state_num, self.action_num, self.action_min, self.action_max).to(self.device)
        self.actor_opt = optim.Adam(self.actor_net.parameters(), lr=learning_rate)
        
        # Target Actor
        self.actor_target_net = ActorNet(self.state_num, self.action_num, self.action_min, self.action_max).to(self.device)
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        
        # Critic
        self.critic_net = CriticNet(self.state_num, self.action_num).to(self.device)
        self.critic_opt = optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        
        # Critic Target
        self.critic_target_net = CriticNet(self.state_num, self.action_num).to(self.device)
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        
        # Learning setting
        self.gamma = gamma
        self.tau = tau
        
        # Noise setting
        self.epsilon = 1
        self.eps_min = eps_min
        self.eps_period = eps_period
        
        # Policy noise and update
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.update_count = 0

    # Get the action
    def get_action(self, state, exploration=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor_net(state).cpu().detach().numpy().flatten()
        
        if exploration:
            # Get noise (gaussian distribution with epsilon greedy)
            action_mean = (self.action_max + self.action_min) / 2
            action_std = (self.action_max - self.action_min) / 2
            action_noise = np.random.normal(action_mean, action_std, 1)[0]
            action_noise *= self.epsilon
            self.epsilon = self.epsilon - (1 - self.eps_min) / self.eps_period if self.epsilon > self.eps_min else self.eps_min
            
            # Final action
            action = action + action_noise
            action = np.clip(action, self.action_min, self.action_max)
            return action
        
        else:
            return action
    
    # Learn the policy
    def learn(self):
        # Replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
    
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)
        
        
        ## Training Actor (Calculation of current Q)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        # Current Q values
        q1_current, q2_current = self.critic_net(states, actions)
        
        
        # Get the action noise for the target policy smoothing.
        # will be added to next action.
        noise = torch.randn_like(actions) * self.policy_noise
        noise = noise.clamp(min=-self.noise_clip, max=self.noise_clip) # make it to a certain range to prevent overly large perturbations
        
        
        ## Training target Critic (Calculation of next Q)
        next_states = torch.FloatTensor(next_states).to(self.device)
        # Target policy smoothing (Get the next action with noise)
        next_actions = self.actor_target_net(next_states) + noise
        next_actions = next_actions.clamp(min=self.action_min, max=self.action_max) # to fall within a valid action range.
        # Double Q learning with clip (Get the target q values through clipped double q)
        q1_next, q2_next = self.critic_target_net(next_states, next_actions)
        
        ## Calculation of expected Q (y value)
        q_next = torch.min(q1_next, q2_next) # clip the double Q learning and take the smallest one
        q_expected = (rewards + self.gamma * q_next * (1-dones))

        
        # Calculate the critic loss which is sum of mean squared error 
        # between Q value of target Critic and the the two Critics. 
        # Main loss or critic loss (mean-squared Bellman error-MSBE, Temporal difference)
        critic_loss = F.mse_loss(q1_current, q_expected.detach()) + F.mse_loss(q2_current, q_expected.detach())
        
        ## Update the critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Delayed policy update (update less frequently and mainly used for stability)
        self.update_count += 1
        # Actor and Critic networks are updated periodically after certain number of iterations using Polyak averaging.
        if self.update_count == self.policy_freq:
            self.update_count = 0
            
            
            # Calculate the actor loss 
            actor_loss = - self.critic_net.network1(states, self.actor_net(states)).mean()
            
            ## Update actor
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # soft update target networks using Polyak averaging
            # For actor target:
            for target_param, param in zip(self.actor_target_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
            # For critic target-1 and critic target-2
            for target_param, param in zip(self.critic_target_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


def main():
    env = gym.make("Pendulum-v0")
    batch_size=64
    agent = TD3(env, memory_size=100000, batch_size=batch_size, gamma=0.95, learning_rate=1e-3, eps_min=0.00001, eps_period=100000, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2)
    ep_rewards = deque(maxlen=1)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.get_action(state, True) # get noisy action
            next_state, reward , done, _ = env.step(action)
            ep_reward += reward

            agent.replay_buffer.add(state, action, reward, next_state, done)
            if len(agent.replay_buffer) > batch_size:
                agent.learn()
            
            if done:
                ep_rewards.append(ep_reward)
                if i % 1 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state

if __name__ == '__main__':
    main()