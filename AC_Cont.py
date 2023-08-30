import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import Input
import tensorflow_probability as tfp
import time

import matplotlib as plt


# About the problem:
# https://www.gymlibrary.dev/environments/classic_control/pendulum/

LEARNING_RATE = 0.001
LEARNING_RATE_ENHANCE = 0.01
gamma = 0.7

episodes = 1000
step_max = 200


class ac_Net:
    def __init__(self, in_num: int, out_num: int):
        self.in_num = in_num
        self.out_num = out_num
        self.epsilon = 1e-07
        self.lr_a = LEARNING_RATE
        self.lr_c = LEARNING_RATE_ENHANCE
        self.gamma = gamma
        self.action_model = self.action_layer_build()
        self.critic_model = self.critic_layer_build()
        self.action_prob_history = []
        self.critic_prob_history = []
        self.reward_history = []
        self.td_error_history = []

    def action_layer_build(self):
        input = Input(shape=(3,))
        fc1= layers.Dense(units=128, activation='relu')(input)
        fc2 = layers.Dense(units=32, activation='relu')(fc1)
        # Model the policy as a Gaussian distribution. Instead of outputting a single action value directly, 
        # the policy network outputs the mean (mu) and standard deviation (sigma) of a Gaussian distribution. 
        actor_mu = layers.Dense(units=self.out_num, activation='sigmoid')(fc2)
        actor_sigma = layers.Dense(units=self.out_num, activation='sigmoid')(fc2) # output_dim=1

        model = models.Model(inputs=input, outputs=[actor_mu, actor_sigma])
        return model

    def critic_layer_build(self):
        input_ = Input(shape=(3,))
        common_ = layers.Dense(units=128, activation='relu')(input_)
        common_ = layers.Dense(units=32, activation='relu')(common_)
        critic_value = layers.Dense(units=self.out_num, activation='sigmoid')(common_)

        model = models.Model(inputs=input_, outputs=critic_value)
        return model

    # def action(self, s):
    #     mu, sigma, _ = self.model(s)
    #     mu = np.squeeze(mu)
    #     sigma = np.squeeze(sigma)
    #     normal_dist = np.random.normal(loc=mu, scale=sigma, size=sample_range)
    #     normal_action = np.clip(normal_dist, env.action_space.low, env.action_space.high)
    #     action = np.random.choice(normal_action)
    #     self.action_prob_history.append([mu, sigma, action])
    #     return action

    # gradient tape record critic td_error
    def loss_critic(self, s, r, s_t1):
        cv = self.critic_model(s)
        cv_t1 = self.critic_model(s_t1)
        # td_error= (r+gamma*V(s')-V(s))
        td_error = (cv_t1 * self.gamma + r) - cv
        self.td_error_history.append(td_error)
        return td_error

    # gradient tape record actor loss
    def loss_actor(self, normal_dist, td):
        with tf.GradientTape() as tape:
            # mu, sigma, _ = self.model.model.output
            # pdf = 1 / np.sqrt(2. * np.pi * sigma) * np.exp(- np.square(action - mu) / (2 * np.square(sigma)))

            log_prob = np.log(pdf + self.epsilon)
            actor_loss = -(log_prob * td)
        return actor_loss

def plot_results(scores, avg_scores, actor_losses, critic_losses):
    """Plot the training progress of the actor-critic algorithm."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot the raw scores
    axs[0, 0].plot(scores)
    axs[0, 0].set_title('Raw scores')

    # Plot the rolling average of the scores
    axs[0, 1].plot(avg_scores)
    axs[0, 1].set_title('Rolling average scores')

    # Plot the actor loss over time
    axs[1, 0].plot(actor_losses)
    axs[1, 0].set_title('Actor loss')

    # Plot the critic loss over time
    axs[1, 1].plot(critic_losses)
    axs[1, 1].set_title('Critic loss')

    # Add axis labels and titles
    for ax in axs.flat:
        ax.set(xlabel='Episode', ylabel='Value')
    fig.suptitle('Training Progress of Actor-Critic with only bootstrapping', fontsize=16)

    fontsize = 12
    for ax in axs.flat:
        ax.tick_params(labelsize=fontsize)
        ax.set_title(ax.get_title(), fontsize=fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    fig.tight_layout()

    # Show the plot
    plt.show()



def AC_loop(episodes, max_step):
    env = gym.make('Pendulum-v0')
    env.seed(1)
    env = env.unwrapped # remove all the wrapper in gym library and return raw/core environment

    action_shape = env.action_space.shape[0]
    state_shape = env.observation_space.shape[0]
    action_range = env.action_space.high[0]

    agent = ac_Net(state_shape, action_shape)
    optimizer_actor = keras.optimizers.Adam(learning_rate=agent.lr_a)
    optimizer_critic = keras.optimizers.Adam(learning_rate=agent.lr_c)
    # Available loss function in Keras: https://keras.io/api/losses/
    loss = keras.losses.Huber()
    
    for epoch in range(episodes):
        obs = env.reset()
        obs = tf.reshape(obs, (1, env.observation_space.shape[0])) # observation space=3
        count = 0
        ep_hs = []
        done=False
        actor_loses=[]
        critic_loses=[]
        avg_scores=[]
        score=0
        while not done:
            env.render()
            # persistent=True: tape does not get deleted after a single gradient computation
            with tf.GradientTape(persistent=True) as tape:
                mu, sigma = agent.action_model(obs)
                mu = tf.squeeze(mu)
                sigma = tf.squeeze(sigma)
                normal_dist = tfp.distributions.Normal(mu, sigma)
                action = tf.clip_by_value(normal_dist.sample(1), -action_range, action_range) # to prevent the agent from taking actions that are outside the range of the possible actions
                agent.action_prob_history.append(action)

                obs_t1, reward, done, info = env.step(action) # obs_t1=next state
                reward = reward / 10
                score+=reward
                obs_t1 = tf.reshape(obs_t1, (1, env.observation_space.shape[0]))

                td_error = agent.loss_critic(obs, reward, obs_t1)
                
                agent.critic_prob_history.append(td_error)
                log_prob = normal_dist.log_prob(action)
                loss_action = -log_prob * td_error # actor loss
                loss_critic = loss(td_error, 0) # critic loss


            # When training a machine learning model, the goal is often to minimize a loss function by adjusting the model's parameters. 
            # Gradient descent algorithms, such as stochastic gradient descent (SGD), 
            #  use the gradients of the loss function with respect to the model's parameters to determine the direction and magnitude of parameter updates.    
                    
            # calculate the gradient of loss function with respect to trainable weights
            grad_action = tape.gradient(loss_action, agent.action_model.trainable_weights) 
            grad_critic = tape.gradient(loss_critic, agent.critic_model.trainable_weights)
            
            # Once you have the gradients, you can pass them to the optimizer's apply_gradients() method along with the corresponding variables. 
            # It updates the model's parameters according to the gradient information and the learning rate.
            # This step essentially moves the model's parameters towards a better configuration in terms of minimizing the loss function.
            optimizer_actor.apply_gradients(zip(grad_action, agent.action_model.trainable_weights))
            optimizer_critic.apply_gradients(zip(grad_critic, agent.critic_model.trainable_weights))
            del tape # release the resources used by the tape

            agent.reward_history.append(reward)
            obs = obs_t1
            count += 1
            ep_hs.append(reward)

            #if count > max_step:
            if done:
                actor_loses.append(loss_action)
                critic_loses.append(loss_critic)
                ep_sum = sum(ep_hs)
                avg_scores.append(np.mean(ep_hs[-50:]))
                #print(f'epoch:{epoch}, reward: {ep_sum}')
                #break
        print('Episode {}\tEpisode Score: {:.2f}'.format(epoch, ep_hs[epoch]))
    
    plot_results(ep_hs, avg_scores, actor_loses, critic_loses)
    #agent.action_model.save('action_model.h5')
    #agent.critic_model.save('critic_model.h5')


if __name__ == '__main__':
    AC_loop(episodes, step_max)



