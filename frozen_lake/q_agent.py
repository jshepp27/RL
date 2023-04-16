import gym
import numpy as np
from pprint import pprint

ALPHA = 0.001
ACTIONS = 0
STATES = 0
EPSILON = 0.01
EPSILON_DECAY = 0.001

GAMMA = 0.9
MAX_STEPS = 1000
EPISODES = 100000

""" Construct Agent """
class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_dec

        self.Q = {}

        self.init_Q()

    # Initialise Q-Table with Zero State-Action Values, given no gained Experience
    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0

    # Select Action over Exploration-Exploration Scheme
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            # Explore
            action = np.random.choice([_ for _ in range(self.n_actions)])
        else:
            # Array of Possible Actions, given State
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            #print("Available: ", actions)

            action = np.argmax(actions) # Note: Argmax Breaks Ties with Lower-Index
            #print("Max: ", actions)

        return action

    # Decay rate of Exploration over Time-Steps
    def decay_epsilon(self):
        # Options: Logrithmic, Linear
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.eps_min \
            else self.eps_min

    # Update State-Action Values given Experience
    def learn(self, state, action, reward, state_):
        # Array of Possible Actions, given Next-State
        actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
        #print(actions)

        # Return Max Valued Action, given Next-State
        a_max = np.argmax(actions)
        #print(a_max)

        # Update Current-State, Current-Action Values given:
        # Differential Maximised Next-State-Action Value and Current State-Action Value
        self.Q[(state, action)] += self.lr * (reward + self.gamma * self.Q[(state_, a_max)] - \
                                                 self.Q[(state, action)])

        # Decreasing Exploration as Inherent to the Agent's Learning Process
        self.decay_epsilon()
