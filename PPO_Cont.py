#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import torch
from torch.distributions import Normal

from collections import deque
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import os



# About problem:
# https://www.gymlibrary.dev/environments/classic_control/pendulum/



# stochastic policy
class actor_builder(torch.nn.Module): # has two outputs, mu and log_sigma
    def __init__(self, innershape, outershape, actor_space):
        super(actor_builder, self).__init__()
        self.input_shape = innershape
        self.output_shape = outershape
        self.action_bound = actor_space[1]
        
        # Common layer same as critic
        self.common = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True)
        )
        
        # Problem dim=1
        self.mu_out = torch.nn.Linear(128, 1) # Predict the mean (mu) of the action distribution
        self.mu_tanh = torch.nn.Tanh() # tanh activation function, bound mu within the range [-1,1] 
        
        # The use of the Softplus activation function for log_sigma is related to modeling a Gaussian (normal) distribution. 
        # In Gaussian distributions, the standard deviation (sigma) must be positive because it determines the spread or dispersion of data. 
        # The Softplus activation ensures that log_sigma remains positive, which is equivalent to ensuring that sigma is positive. 
        # This is crucial for representing valid standard deviations in a probability distribution.
        self.sigma_out = torch.nn.Linear(128, 1) # Predict log standard deviation (log_sigma) of the action distribution
        self.sigma_tanh = torch.nn.Softplus() # softplus activation function, to ensure predicted standart deviation (sigma) is positive

    
    #  Since PPO uses stochastic policy, compute the mean and variance of each variable for action selection
    def forward(self, obs_ac):
        common = self.common(obs_ac)
        mean = self.mu_out(common)
        mean = self.mu_tanh(mean) * self.action_bound # to rescale for the valid action range of the envieonment
        log_sigma = self.sigma_out(common)
        log_sigma = self.sigma_tanh(log_sigma)

        return mean, log_sigma


class critic_builder(torch.nn.Module): # has one ouput: value
    def __init__(self, innershape):
        super(critic_builder, self).__init__()
        self.input_shape = innershape
        
        # Common layer same as actor
        self.common = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True)
        )
        self.value = torch.nn.Linear(128, 1)
  
    def forward(self, obs_ac):
        out = self.common(obs_ac)
        out = self.value(out)
        return out


class PPO:
    def __init__(self, inputshape, outputshape, action_space):
        self.action_space = action_space
        self.input_shape = inputshape
        self.output_shape = outputshape
        self._init(self.input_shape, self.output_shape, self.action_space)
        self.lr_actor = 0.0001
        self.lr_critic = 0.0001
        self.batch_size = 64 # number of samples that are randomly selected from the total amount of stored data
        self.decay_index = 0.95
        # self.sigma = 0.5
        # self.sigma_actor = np.full((T_len, 1), 0.5, dtype='float32')
        
        # A value of 0.2 will clip the ratio between 0.8 and 1.2
        self.epilson = 0.2 # is used to clip the ratio between the new policy and old policy
        
        self.c_loss = torch.nn.MSELoss()
        self.c_opt = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_critic) # critic optimizer
        self.a_opt = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_actor) # actor optimizer
        self.update_actor_epoch = 5
        self.update_critic_epoch = self.update_actor_epoch
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

    def _init(self, inner, outter, actionspace):
        self.pi = actor_builder(inner, outter, actionspace)
        self.piold = actor_builder(inner, outter, actionspace)
        self.v = critic_builder(inner)
        self.memory = deque(maxlen=T_len)

    def get_action(self, obs_):
        obs_ = torch.Tensor(copy.deepcopy(obs_))
        mean, sigma = self.pi(obs_)
        

        # You can change normal (Gaussian) distribution to discrete one for discrete problem
        dist = Normal(mean.cpu().detach(), sigma.cpu().detach()) # Use normal distribution to help Pytorch to sample actions
        prob_ = dist.sample()
        log_prob_ = dist.log_prob(prob_)
        return prob_, log_prob_

    def state_store_memory(self, s, a, r, logprob_):
        self.memory.append((s, a, r, logprob_))

    
    def compute_gae(self, next_state_frame, values, reward_,masks, gamma=0.99, lam=0.95): # Do it
        returns=[]
        gae=0
        state_frame = torch.Tensor(next_state_frame)
        #next_value= ppo.v(state_frame).detach().numpy() # numeric
        next_value= ppo.v(state_frame) # tensor
        values = values + [next_value]
        for step in  reversed(range(len(reward_))):
            delta = reward_[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lam * masks[step] * gae
            returns.insert(0,gae+values[step])
        return returns
    
  
    # Calculate the reward decay, according to the Markov process, push forward from the last reward
    def decayed_reward(self, singal_state_frame, reward_):
        decayed_rd = []
        state_frame = torch.Tensor(singal_state_frame)
        value_target = ppo.v(state_frame).detach().numpy()
        for rd_ in reward_[::-1]:
            value_target = rd_ + value_target * self.decay_index
            decayed_rd.append(value_target)
        decayed_rd.reverse()
        return decayed_rd

    
    # Calculate the advantage value for actor update
    # Use the advantage function instead of the expected reward because it reduces the variance of the estimation
    def advantage_calcu(self, decay_reward, state_t1,returns,values):   
        ''''
        # Optional advantage calculation with discounted reward
        state_t1 = torch.Tensor(state_t1)
        critic_value_ = self.v(state_t1)
        d_reward = torch.Tensor(decay_reward)
        advantage = d_reward - critic_value_
        return advantage
        '''
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values).detach()
        print(f'returns:  {returns.size()[0]} values:  {values.size()[0]} ')
        advantage=returns - values
        return advantage

    
    ## Once a batch of experience has been used to do gradient update, the experience is then discarded
    ## and then the policy moves on
    
    # Calculate Q(s, a) and V(s) for critic update
    def critic_update(self, state_t1, d_reward_,returns,values,with_batch,rand_ids):
        if with_batch==True:
            returns_batch   = torch.cat(returns).detach().requires_grad_(True)
            values_batch    = torch.cat(values).detach().requires_grad_(True)
            
            #rand_ids = torch.LongTensor(rand_ids)
            #returns_tensor = torch.cat(returns)  # Convert the list of tensors to a single tensor
            #values_tensor = torch.cat(values)  
            #returns_batch = returns_tensor.index_select(0, rand_ids)
            #values_batch = values_tensor.index_select(0, rand_ids) 
            
            #returns_batch = returns[rand_ids]
            #values_batch = values[rand_ids]
            
            critic_loss = (returns_batch- values_batch).pow(2).mean()
            self.history_critic = critic_loss.detach().item()
            self.c_opt.zero_grad()
            critic_loss.backward(retain_graph=True)
            #critic_loss.backward()
            self.c_opt.step()
            
            
        else:   # with discounted reward
            q_value = torch.Tensor(d_reward_).squeeze(-1)
            target_value = self.v(state_t1).squeeze(-1)
            critic_loss = self.c_loss(target_value, q_value) # DEGISECEK, gae GELECEK
            self.history_critic = critic_loss.detach().item()
            self.c_opt.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.c_opt.step()

    
    
    def actor_update(self, state_, action_, advantage,mini_batch_size, with_batch=True):
        if with_batch==True:
            batch_size = state_.size(0)
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            action_ = torch.FloatTensor(action_)
            action_=action_[rand_ids]
            self.a_opt.zero_grad()
            state_=state_[rand_ids]
            pi_mean, pi_sigma = self.pi(state_)
            pi_mean_old, pi_sigma_old = self.piold(state_)
            advantage=advantage[rand_ids]

            pi_dist = Normal(pi_mean, pi_sigma)
            pi_dist_old = Normal(pi_mean_old, pi_sigma_old)

            # log likelihoods of particular actions
            logprob_new = pi_dist.log_prob(action_.reshape(-1, 1)) # new policy action probability
            logprob_old = pi_dist_old.log_prob(action_.reshape(-1, 1)) # old policy action probability

            # To ensure that the policy update is not too drastic from the old policy to the new policy. 
            # We want to prevent policy updates that are too aggressive, which could lead to instability in training. I
            ratio = torch.exp(logprob_new - logprob_old)
            
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage

            actor_loss = torch.min(torch.cat((surrogate1, surrogate2), dim=1), dim=1)[0]
            actor_loss = -torch.mean(actor_loss)
            self.history_actor = actor_loss.detach().item()

            actor_loss.backward(retain_graph=True)
            self.a_opt.step()

            return rand_ids 
        else:
            action_ = torch.FloatTensor(action_)
            self.a_opt.zero_grad()
            pi_mean, pi_sigma = self.pi(state_)
            pi_mean_old, pi_sigma_old = self.piold(state_)

            pi_dist = Normal(pi_mean, pi_sigma)
            pi_dist_old = Normal(pi_mean_old, pi_sigma_old)

            logprob_new = pi_dist.log_prob(action_.reshape(-1, 1))
            logprob_old = pi_dist_old.log_prob(action_.reshape(-1, 1))

            ratio = torch.exp(logprob_new - logprob_old)
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage

            actor_loss = torch.min(torch.cat((surrogate1, surrogate2), dim=1), dim=1)[0]
            actor_loss = -torch.mean(actor_loss)
            self.history_actor = actor_loss.detach().item()

            actor_loss.backward(retain_graph=True)
            self.a_opt.step()
            rand_ids=0
            return rand_ids

    def update(self, state_all, action_, discount_reward_,returns,values,mini_batch_size):
        self.hard_update(self.pi, self.piold)
        state_ = torch.Tensor(state_all)
        act = action_
        d_reward = np.concatenate(discount_reward_).reshape(-1, 1)
        adv = self.advantage_calcu(d_reward, state_,returns,values)
        with_batch=True
        if with_batch:
            # Each iteration, construct surrogete loss on "max_step_episode_len" timesteps
            # and optimize it with minibatch SGD (can use ADAM) for K epochs.
            for i in range(self.update_actor_epoch):
                rand_ids =self.actor_update(state_, act, adv, mini_batch_size,with_batch=with_batch)
                print(f'epochs: {self.ep}, timestep: {self.t}, actor_loss: {self.history_actor}')
                self.critic_update(state_, d_reward, returns,values,with_batch=with_batch, rand_ids=rand_ids)
                print(f'epochs: {self.ep}, timestep: {self.t}, critic_loss: {self.history_critic}')
        else:
            for i in range(self.update_actor_epoch):
                rand_ids =self.actor_update(state_, act, adv, mini_batch_size,with_batch=with_batch)
                print(f'epochs: {self.ep}, timestep: {self.t}, actor_loss: {self.history_actor}')
            for i in range(self.update_critic_epoch):
                self.critic_update(state_, d_reward,returns,values,with_batch=with_batch, rand_ids=0)
                print(f'epochs: {self.ep}, timestep: {self.t}, critic_loss: {self.history_critic}')
        
            

    @staticmethod
    def hard_update(model, target_model):
        weight_model = copy.deepcopy(model.state_dict())
        target_model.load_state_dict(weight_model)


if __name__ == '__main__':
    T_len = 64 # Test len
    max_step_episode_len=3200
    epochs=200
    
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    # env.seed(1)
    
    action_shape = [env.action_space.low.item(), env.action_space.high.item()]
    state_shape = np.array(env.observation_space.shape)        

    ppo = PPO(state_shape.item(), 1, action_shape)

    count = 0
    ep_history = []
    values=[]
    masks=[]
    rewards=[]
    for epoch in range(epochs):
        obs = env.reset()
        obs = obs.reshape(1, 3)
        ep_rh = 0
        ppo.ep += 1
        
        
        # get the reward, value..etc during each time step in the current episode
        for t in range(max_step_episode_len): # each time step generates action, observation and reward
            env.render()
            
            # Policy (action) is samples from gaussion distribution to get continuous output value 
            action, logprob = ppo.get_action(obs)
            state_frame = torch.Tensor(obs)
            #value_target = ppo.v(state_frame).detach().numpy() # numeric
            value_target = ppo.v(state_frame) # tensor
            # each state has a list
            obs_t1, reward, done, _ = env.step(action.detach().numpy().reshape(1, 1)) # obs_t1: next state
            obs_t1 = obs_t1.reshape(1, 3)
            reward = (reward + 16) / 16 #  scaling the rewards to a range of roughly -1 to 0 (optional)
            ppo.state_store_memory(obs, action.detach().numpy().reshape(1, 1), reward, logprob)
            values.append(value_target)
            masks.append(done)
            obs = obs_t1
            ep_rh += reward

            if (t+1) % T_len == 0 or t == max_step_episode_len- 1:
                s_t, a, rd, _ = zip(*ppo.memory)
                s_t = np.concatenate(s_t).squeeze()
                a = np.concatenate(a).squeeze()
                rd = np.concatenate(rd)
                
                if len(values)==128:
                   print('stop')

                discount_reward = ppo.decayed_reward(obs_t1, rd)  # Value geri donduruyor
                returns=ppo.compute_gae(obs_t1, values,rd,masks, gamma=0.99, lam=0.95) # Do it
                
                ppo.update(s_t, a, discount_reward,returns,values,mini_batch_size=5)
                ppo.memory.clear()
                values=[]
                masks=[]

            ppo.t += 1
        ep_history.append(ep_rh)
        print(f'epochs: {ppo.ep}, ep_reward: {ep_rh}')

    ep_history = np.array(ep_history)
    plt.plot(np.arange(ep_history.shape[0]), ep_history)
    plt.show()
    env.close()