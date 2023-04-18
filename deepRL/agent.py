# Choose Actions
# Learn from Experience
import torch as T
from deep_Q_network import LinearDeepNetwork
import numpy as np
import gym

from utils import plot_learning_curve
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# TODO: Q-Learning Tabular Agent
# TODO: Q-Learning Deep Learning Agent

class Agent():
    def __init__(self, input_dims, n_actions,
                 lr=0.0001, gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [_ for _ in range(self.n_actions)]

        self.Q = LinearDeepNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Exploit

            # Encode Observation-State within the Q-Network
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)

            # Propagate Forward to obtain Actions
            actions = self.Q.forward(state)

            # Obtain Action
            action = T.argmax(actions).item()

        else:
            # Explore
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        # Linear Decay
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        # Why Zero-Grad ?
        self.Q.optimizer.zero_grad()
        # Numpy to Tensors
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        # Target Value

        # Propagate Forward with Current State, to obtain Current Q-Action Value
        q_pred = self.Q.forward(states)[actions]
        #print(q_pred)

        # Propagate Forward with Next-State, to obtain Next Q-Action Value
        q_next = self.Q.forward(state_).max()

        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()

        self.Q.optimizer.step()
        self.decrement_epsilon()

def create_animation(fig, ax, x_axis, scores, eps_history, filename, writer='ffmpeg', fps=30):
    def animate(i):
        ax.clear()
        ax.plot(x_axis[:i + 1], scores[:i + 1], label="scores")
        ax.plot(x_axis[:i + 1], eps_history[:i + 1])
        ax.set_title("Learning Curve DQN")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Learning")

    ani = animation.FuncAnimation(fig, animate, frames=len(x_axis), repeat=True)
    #ani.save(filename, writer=writer, fps=fps)

    plt.show()

    return ani

from utils import create_plot
""" Train Agent """
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_games = 100
    scores = []
    eps_history = []

    agent = Agent(input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)

    for episode in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            # Select Action, given Observation in Environment
            action = agent.choose_action(obs)

            # Take Action and Stepping-through (Transitioning) the Environment
            obs_, reward, done, info = env.step(action)
            score += reward

            # Update Q-Estimates through Learning Rule
            agent.learn(obs, action, reward, obs_)

            obs = obs_

        scores.append(score)
        eps_history.append(agent.epsilon)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print("Episode: ", episode, f"Score {score}", f"Avg Score {avg_score}")

    filename = "cartpole_naive_dqn.png"
    x = [_ + 1 for _ in range(n_games)]

    # print(len(x))
    # print(len(scores))

    plot_learning_curve(x, scores, eps_history, filename)

    fig, ax = create_plot(x, scores, eps_history)
    #ani = create_animation(fig, ax, x, scores, eps_history, filename)


