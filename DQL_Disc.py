import gym
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import wrappers
import random
from torch.autograd import Variable
from IPython import display



# About Deep and Double Q learning:
# https://builtin.com/artificial-intelligence/double-deep-q-learning

# Difference between double and deep q learning:
#https://datascience.stackexchange.com/questions/38632/what-is-difference-between-the-ddqn-and-dqn#:~:text=From%20what%20I%20understand%2C%20the,values%20over%20all%20possible%20actions.



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


def q_learning(env, model, episodes, gamma=0.9,
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20,
               title='DQL', double=False,
               n_update=10, soft=False, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = []
    episode_i = 0
    sum_total_replay_time = 0
    for episode in range(episodes):
        episode_i += 1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()

        # Reset state
        state = env.reset()
        done = False
        total = 0

        while not done: # iterade until the episode is done
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state) # q value for each state 
                action = torch.argmax(q_values).item() # take max index

            # Render the game screen and update it with each step
            #env.render(mode='human')
            
            plt.figure(3)
            plt.clf()
            env_screen = env.render(mode='rgb_array')
            #env.close()
            plt.imshow(env_screen)
            plt.title(f'Episode no:  {episode}')
            display.clear_output(wait=True)
            #display.display(plt.gcf())
            plt.draw()
            plt.pause(0.00001)
            #plt.show()
            
            # If you want to see a screenshot of the game as an image,
            # rather than as a pop-up window, you should set the mode argument of the render function to rgb_array:
            # env_screen = env.render(mode='rgb_array')
            # env.close()
            # plt.imshow(env_screen)
            # plt.title(f'Episode no:  %episode')
            
            # Take action and add reward to total
            # Apply the action to the environment
            next_state, reward, done, _ = env.step(action) # Determine next state and reward based on action

            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))

            
            q_values = model.predict(state).tolist() # tensor to list

            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    model.update(state, q_values)
                break

            if replay: # Replay memory var ise
                t0 = time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
                t1 = time.time()
                sum_total_replay_time += (t1 - t0)
            else:
                # Update network weights using the last step only
                #  the neural network is used as a function approximator to estimate the Q-values for different state-action pairs.

                q_values_next = model.predict(next_state) # Predict the q value for the next state
                kk=torch.max(q_values_next).item()
                # Update the current state-action pair of Q values
                q_values[action] = reward + gamma * kk # Q(s,a) = r + γ * max(Q(s',a'))
                
                model.update(state, q_values) # tranning loop

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


# Deep Q-Learning is a specific RL algorithm that utilizes neural networks to approximate Q-values.
class BuildNN():
    ''' Deep Q Nural Network class. '''
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
    def update(self, current_state, current_state_q_values):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(current_state)) # produces a set of predicted Q-values for all possible actions in the current state.
        # Temporal difference:
        loss = self.criterion(y_pred, Variable(torch.Tensor(current_state_q_values))) # calculates the error between the predicted Q-values and the target Q-values
        self.model.zero_grad() # sets the gradients of all parameters in the neural network to zero
        loss.backward() # (Direkt sonuca etki ediyor) The gradients of the loss function with respect to the weights of the neural network are calculated using backpropagation.
        self.optimizer.step() # updates the parameters of the neural network using the calculated gradients.
        # "self.optimizer" is responsible for updating the model parameters based on the gradients with respect to loss function
        # step() in "self.optimizer.step()"  is called on the optimizer object to apply these updates to the neural network.
    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))




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

# Get DQN results
simple_dqn = BuildNN(n_state, n_action, n_hidden, lr)
simple = q_learning(env, simple_dqn, num_episodes , gamma=.9, epsilon=0.3)
print('stop')



