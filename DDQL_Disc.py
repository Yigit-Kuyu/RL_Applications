import gym
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import wrappers
import random
from torch.autograd import Variable
import time
import copy



# About DDQL:
# https://towardsdatascience.com/double-deep-q-networks-905dd8325412


def plot_res(values, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
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


def q_learning(env, Build_DoubleDQN, episodes, gamma=0.9,
               epsilon=0.3, eps_decay=0.99,
               replay=True, replay_size=20,
               title='DQL', double=True,
               n_update=10, soft=True, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    #DQN_replay_instance = DQN_replay()
    final = []
    memory = []
    episode_i = 0
    sum_total_replay_time = 0
    for episode in range(episodes):
        episode_i += 1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                Build_DoubleDQN.target_update()
        if double and soft:  # Double Q learning and  soft update
            Build_DoubleDQN.target_update()

        # Reset state
        state = env.reset()
        done = False
        total = 0

        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = Build_DoubleDQN.predict(state) # Q values for all actions
                action = torch.argmax(q_values).item() 

            # Render the game screen and update it with each step
            env.render(mode='human')
            # If you want to see a screenshot of the game as an image,
            # rather than as a pop-up window, you should set the mode argument of the render function to rgb_array:
            # env_screen = env.render(mode='rgb_array')
            # env.close()
            # plt.imshow(env_screen)

            # Take action and add reward to total
            # Apply the action to the environment
            next_state, reward, done, _ = env.step(action)
            

            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))

            
            q_values = Build_DoubleDQN.predict(state).tolist()
            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    Build_DoubleDQN.update(state, q_values)
                break

            if replay: # Replay memory var
                t0 = time.time()
                # Update network weights using replay memory
                Build_DoubleDQN.replay(memory, replay_size, gamma)
                t1 = time.time()
                sum_total_replay_time += (t1 - t0)
            else:
                # Update network weights using the last step only
                #  the neural network is used as a function approximator to estimate the Q-values for different state-action pairs.

                
                q_values_next = BuildNN.predict(next_state)
                kk=torch.max(q_values_next).item()
                # Update the current state-action pair of Q values
                q_values[action] = reward + gamma * kk # Q(s,a) = r + γ * max(Q(s',a'))
               
                BuildNN.update(state, q_values) # tranning loop

            state = next_state

        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        """
        plt.imshow(env.render(mode='rgb_array'))  # visualize game after each episode
        plt.axis('off')
        plt.show()
        """
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            if replay:
                print("Average replay time:", sum_total_replay_time / episode_i)
    avg_reward = sum(final) / episodes
    print("Average reward:", avg_reward)
    plot_res(final, title)

    return final



class Build_DoubleDQN():
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim*2, action_dim)
                    )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
            self.target = copy.deepcopy(self.model)

    def update(self, current_state, current_state_q_values):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(current_state)) # produces a set of predicted Q-values for all possible actions in the current state.
        # Temporal difference:
        loss = self.criterion(y_pred, Variable(torch.Tensor(current_state_q_values))) # calculates the error between the predicted Q-values and the target Q-values
        self.model.zero_grad() # sets the gradients of all parameters in the neural network to zero
        loss.backward() #  The gradients of the loss function with respect to the weights of the neural network are calculated using backpropagation.
        self.optimizer.step() # updates the parameters of the neural network using the calculated gradients.
        # "self.optimizer" is responsible for updating the model parameters based on the gradients with respect to loss function
        # step() in "self.optimizer.step()"  is called on the optimizer object to apply these updates to the neural network.

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def replay(self, memory, size, gamma=0.9):

        if len(memory) >= size+4:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            self.update(states, targets) # targets: Q values of the states

    def target_predict(self, s):
            ''' Use target network to make predicitons.'''
            with torch.no_grad():
                return self.target(torch.Tensor(s))

    def target_update(self, TAU=0.1):
            ''' Update target network with the model weights.'''
            model_params = self.model.named_parameters()
            target_params = self.target.named_parameters()

            updated_params = dict(target_params)

            for model_name, model_param in model_params:
                if model_name in target_params:
                    # Update parameter (soft update)
                    updated_params[model_name].data.copy_((TAU) * model_param.data + (1 - TAU) * target_params[model_param].data)

            self.target.load_state_dict(updated_params)


            self.target.load_state_dict(self.model.state_dict())

# Demonstration
env = gym.envs.make("CartPole-v1")
#env= wrappers.Monitor(env, 'random_files',force=True) # Visualize initial environment
num_episodes = 150

# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001


doubleQ_with_replay= Build_DoubleDQN(n_state, n_action, n_hidden, lr)

double_Q = q_learning(env, doubleQ_with_replay, num_episodes, gamma=.9, epsilon=0.2, replay=True, title='DoubleDQL with Replay')