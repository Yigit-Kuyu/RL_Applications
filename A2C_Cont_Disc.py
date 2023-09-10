import gym
import torch
import numpy as np
from torch.distributions.categorical import Categorical
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Memory:
    def __init__(self):
        self.rewards = []
        self.action_prob = []
        self.state_values = []
        self.entropy = []

    # compute the discounted rewards
    def calculate_data_dr(self, gamma):
        disc_rewards = []
        R = 0
        # Update the value of the last state by using the last reward,
        # the value of the second last step with two rewards so on up to the first state
        for reward in self.rewards[::-1]:
            R = reward + gamma*R
            disc_rewards.insert(0, R)

        
        disc_rewards = torch.Tensor(disc_rewards) # transform to tensor 
        min_reward = disc_rewards.min()
        max_reward = disc_rewards.max()
        disc_rewards = (disc_rewards - min_reward) / (max_reward - min_reward) # normalizes the rewards to the range [0, 1] 
        

        return torch.stack(self.action_prob), torch.stack(self.state_values), disc_rewards.to(device), torch.stack(self.entropy)

    
    ## http://incompleteideas.net/book/RLbook2020.pdf , section 10.3, pp. 249
    # compute the average rewards
    def calculate_data_ar(self):
        
        average_reward=sum(self.rewards) / len(self.rewards)

        average_reward_value_function = [r - average_reward for r in self.rewards]
        average_reward_value_function=torch.Tensor(average_reward_value_function)
        return torch.stack(self.action_prob), torch.stack(self.state_values), average_reward_value_function.to(device), torch.stack(self.entropy)


    
    def update(self, reward, entropy, log_prob, state_value):
        self.entropy.append(entropy)
        self.action_prob.append(log_prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)

    def reset(self):
        del self.rewards[:]
        del self.action_prob[:]
        del self.state_values[:]
        del self.entropy[:]



# stochastic policy
class ActorCriticContinuous(nn.Module):
    
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticContinuous, self).__init__()

        # For critic
        self.fc_1_critic = nn.Linear(state_dim, hidden_dim)
        self.fc_2_critic = nn.Linear(hidden_dim, int(hidden_dim/2))

        # For actor
        self.fc_1_actor = nn.Linear(state_dim, hidden_dim)
        self.fc_2_actor = nn.Linear(hidden_dim, int(hidden_dim/2))

        # critic head: output state value
        self.critic_head = nn.Linear(int(hidden_dim/2), 1) # problem dim=1

        # actor head: output mean and std
        self.actor_head_mean = nn.Linear(int(hidden_dim/2), action_dim)
        self.actor_head_sigma = nn.Linear(int(hidden_dim / 2), action_dim)

    
    # The critic network in A2C is responsible for estimating the state value function, not the action-value function (Q-value).
    # This is because the A2C algorithm uses the advantage function to update the policy and the value function.
    def forward_critic(self, inp):
        x = F.leaky_relu(self.fc_1_critic(inp))
        x = F.leaky_relu(self.fc_2_critic(x))

        # how good is the current state?
        state_value = self.critic_head(x) # state-value

        
        return state_value
    
    def forward_actor(self,inp):
        x = F.leaky_relu(self.fc_1_actor(inp))
        x = F.leaky_relu(self.fc_2_actor(x))

        action_mean = (self.actor_head_mean(x))
        action_sigma = F.softplus(self.actor_head_sigma(x))

        return action_mean, action_sigma


class ActorCriticDiscrete(nn.Module):
   
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticDiscrete, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, int(hidden_dim/2))

        # critic head
        self.critic_head = nn.Linear(int(hidden_dim/2), 1)

        # actor head
        self.actor_head = nn.Linear(int(hidden_dim/2), action_dim)

    def forward(self, inp):
        x = F.leaky_relu(self.fc_1(inp))
        x = F.leaky_relu(self.fc_2(x))

        # how good is the current state?
        state_value = self.critic_head(x)

        # actor's probability to take each action
        action_prob = F.softmax(self.actor_head(x), dim=-1)

        return action_prob, state_value



# No replay buffer
class A2C:

    def __init__(self, parameters, learning_rate):
        
        self.lr = learning_rate
        self.optimize = torch.optim.Adam(parameters, learning_rate)


    # Takes a Memory object (presumably containing experiences), a discount factor (gamma), and an epsilon value (eps) as inputs.
    # It computes the loss and performs one step of optimization on the actor-critic model using the advantage actor-critic algorithm.
    def train(self,memory, gamma, eps):
        dis_count_reward_option=0
        if dis_count_reward_option==1: # For discounted reward
            action_prob, values, disc_rewards, entropy = memory.calculate_data_dr(gamma)
            # Calculate advantage function as value function (values) and the return of the current N-step trajectoy (disc_reward)
            advantage = disc_rewards.detach() - values
        else: # For average reward
            action_prob, values, average_reward_value, entropy =memory.calculate_data_ar() 
            advantage = average_reward_value.detach() - values
        # Actor loss (first aim to maximize discounted reward (also advantage due to close relationship with discounted reward) so minimize negative one)
        # The policy loss can vary greatly from episode to episode. If you use gradient descent to update the network parameters, 
        # each sample's loss would contribute to the parameter updates, but with potentially different scales.
        # Using .mean() scales the loss so that it represents the average loss over the entire batch.  This is a kind of normalization.
        policy_loss = (-action_prob*advantage.detach()).mean() - eps*entropy.mean() # Add entropy for exploration
        # Critic-loss
        value_loss = 0.5 * advantage.pow(2).mean() #MSE
        ## Total-loss
        loss = policy_loss + value_loss 

        self.optimize.zero_grad()
        loss.backward()
        self.optimize.step()

    
        

# It processes the state through the model to obtain an action.
# For continuous action spaces, it calculates both the mean and standard deviation of the action distribution.
# It then samples an action from a multivariate normal distribution defined by these parameters.
def select_action(model, state, mode):
    state = torch.Tensor(state).to(device)
    if mode == "continuous":
        V_value=model.forward_critic(state)
        mean, sigma = model.forward_actor(state)
        s = torch.distributions.MultivariateNormal(mean, torch.diag(sigma))
    else:
        probs, state_value = model(state)
        s = Categorical(probs)

    action = s.sample()
    entropy = s.entropy()
    log_prob_action=s.log_prob(action) #  the log probability of the sampled action 

    return action.cpu().numpy(), entropy,log_prob_action, V_value

# This function evaluates the performance of the actor-critic model in the environment by running repeats number of episodes.
# It returns the average performance over these episodes.
def evaluate(actor_critic, env, repeats, mode):
    actor_critic.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                if mode == "continuous":
                    mean, sigma = actor_critic.forward_actor(state)
                    m = torch.distributions.Normal(mean, sigma)
                else:
                    probs, _ = actor_critic(state)
                    m = Categorical(probs)

            action = m.sample()
            state, reward, done, _ = env.step(action.cpu().numpy())
            perform += reward
    actor_critic.train()
    return perform/repeats


def main(gamma=0.99, lr=5e-3, num_episodes=400, eps=0.001, seed=42, lr_step=100, lr_gamma=0.9, measure_step=10, 
         measure_repeats=100, horizon=200, hidden_dim=64, env_name='Pendulum-v0', render=True): # 'CartPole-v1' for discrete
    
    env = gym.make(env_name)
    torch.manual_seed(seed)
    env.seed(seed)
    

    # check whether the environment has a continuous or discrete action space.
    if type(env.action_space) == gym.spaces.Discrete:
        action_mode = "discrete"
    elif type(env.action_space) == gym.spaces.Box:
        action_mode = "continuous"
    else:
        raise Exception("action space is not known")

    # Get number of actions for the discrete case and action dimension for the continuous case.
    if action_mode == "continuous":
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]

    if action_mode == "continuous":
        actor_critic = ActorCriticContinuous(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    else:
        actor_critic = ActorCriticDiscrete(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim).to(device)

    a2c=A2C(actor_critic.parameters(),0.0001)
    performance = []
    for episode in range(num_episodes):
        # reset memory
        memory = Memory()
        # display the episode_performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(actor_critic, env, measure_repeats, action_mode)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            

        state = env.reset()

        done = False
        count = 0
        # The agent performs N steps (or less if the episode stops earlier) in the environment before each update
        while not done and count < horizon:
            count += 1
            action, entropy, log_prob, state_value = select_action(actor_critic, state, action_mode)
            state, reward, done, _ = env.step(action)
            env.render()
            # save the information
            memory.update(reward, entropy, log_prob, state_value)


        # Update process
        a2c.train(memory, gamma, eps) # At each update, the agent collected up to N (horizon) states and rewards
        

    return actor_critic, performance


if __name__ == '__main__':
    main()
