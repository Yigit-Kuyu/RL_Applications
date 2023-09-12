import torch
import torch.nn.functional as F
import torch.optim as optim
import gym
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Modified by Open.ai Chatgpt

# About Actor-Critic
# https://huggingface.co/blog/deep-rl-a2c


env = gym.make('CartPole-v1')


# Define the Actor-Critic network
class ActorCriticNetwork:
    def __init__(self, state_size, action_size, hidden_size): # action_size=2, hidden_size=64
        self.actor_fc1 = torch.nn.Linear(state_size, hidden_size)
        self.actor_fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.actor_fc3 = torch.nn.Linear(hidden_size, action_size)
        self.actor_activation = torch.nn.LeakyReLU()

        self.critic_fc1 = torch.nn.Linear(state_size, hidden_size)
        self.critic_fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.critic_fc3 = torch.nn.Linear(hidden_size, 1)
        self.critic_activation = torch.nn.LeakyReLU()

        self.optimizer = optim.Adam(self.parameters_yck(), lr=lr)

    # The basic idea behind actor-critic methods is that there are two deep neural networks. 
    
    # The actor network approximates the agent’s policy: a probability distribution that tells us 
    # the probability of selecting a (continuous) action given some state of the environment. 
    
    # The critic network approximates the value function: the agent’s estimate of future 
    # rewards that follow the current state. 
    
    def forward(self, x):
        # The actor is responsible for selecting actions based on the current state and
        # uses the critic's estimates of the Q-values to update its policy by choosing the action with the highest expected Q-value.
        # The critic estimates the Q-values for the state-action pairs.
        x1 = self.actor_activation(self.actor_fc1(x))
        x2 = self.actor_activation(self.actor_fc2(x1))
        actor_x = self.actor_fc3(x2) # For given state, returns two values for two possible actions

        x1 = self.critic_activation(self.critic_fc1(x))
        x2 = F.relu(self.critic_fc2(x1))
        critic_x = self.critic_fc3(x2)  # For given state, returns one Q value

        return actor_x, critic_x

    def parameters_yck(self):
        params = list(self.actor_fc1.parameters()) + list(self.actor_fc2.parameters()) + list(self.actor_fc3.parameters())
        params += list(self.critic_fc1.parameters()) + list(self.critic_fc2.parameters()) + list(self.critic_fc3.parameters())
        return params

# the epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
        if torch.rand(1).item() < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                actor_values, _ = actor_critic_network.forward(torch.tensor(state).float().unsqueeze(0))
                return actor_values.argmax(dim=1).item()


def plot_res(values, title=''):
    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    # Set hyperparameters
    lr = 0.001
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.001
    epsilon_decay = 0.995
    num_episodes = 1000


    # Initialize the Actor-Critic network and optimizer
    actor_critic_network = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.n, 64)

    final = []
    title = 'actor-critic'
    # Train the Actor-Critic network
    for episode in range(num_episodes):
        current_state = env.reset()
        done = False
        total_reward = 0
        epsilon = max(epsilon_min, epsilon_decay * epsilon) # epsilon is updated with an epsilon decay policy.
        while not done:
            action = epsilon_greedy_policy(current_state, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # estimate the actor values (Q-values for all possible actions) for the current state,
            # estimate critic value for the current state ( Q-values for the state-action pairs)
            actor_values, critic_value = actor_critic_network.forward(torch.tensor(current_state).float().unsqueeze(0))
            # estimate the actor values of next state
            next_actor_values, _ = actor_critic_network.forward(torch.tensor(next_state).float().unsqueeze(0))
            # The TD( Temporal Difference) target is calculated as the sum of the reward with selecting the action with the highest value
            
            td_target = reward + gamma * next_actor_values.max(dim=1)[0] * (1 - done) 
            # the actor always chooses the action with the highest policy value (next_actor_values)
            
            # The TD error depends on next actor values for current state
            td_error = td_target - critic_value
            # obtain a probability distribution over the two possible actions.
            actor_probabilities = F.softmax(actor_values, dim=-1)
            target_action = torch.tensor([action], dtype=torch.int64)
            target_probabilities = F.one_hot(target_action, num_classes=2).float()
            # actor_loss: between the actor output for the current state (actor_values) and the selected action (action).
            # measure of how well the actor network is selecting actions that lead to high rewards:
            actor_loss = F.mse_loss(actor_probabilities, target_probabilities)
            # measure of how well the critic network is estimating the value function:
            critic_loss = F.mse_loss(critic_value, td_target.detach())
            loss = actor_loss + critic_loss
            actor_critic_network.optimizer.zero_grad()
            loss.backward()
            actor_critic_network.optimizer.step()
            state = next_state
        final.append(total_reward)
        print(f"Episode {episode + 1}: Total reward = {total_reward}")
    plot_res(final, title)
