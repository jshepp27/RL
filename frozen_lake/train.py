import gym
import matplotlib.pyplot as plt
import numpy as np
from q_agent import Agent
import pickle

ALPHA = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9999995

GAMMA = 0.9
MAX_STEPS = 1000
EPISODES = 500000

""" Train Agent """
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    agent = Agent(lr=ALPHA, gamma=GAMMA, n_actions=4, n_states=16, eps_start=EPSILON_START,
                  eps_end=EPSILON_END, eps_dec=EPSILON_DECAY)

    scores = []
    win_pct_list = []

    for episode in range(EPISODES):
        done = False
        # Initial Observation-State: Observation == External Representation; State == Internal Representation
        observation = env.reset()
        score = 0

        # Play Episode
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            # Update State-Action Q-Values
            agent.learn(observation, action, reward, observation_)

            score += reward
            observation = observation_

        scores.append(score)

        if episode % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if episode % 1000 == 0:
                print('episode ', episode, 'win pct %.2f' % win_pct,
                      'epsilon %.2f' % agent.epsilon)

    plt.plot(win_pct_list)
    plt.show()
    plt.savefig("q_agent.png")

    # Persist Agent
    with open('trained_q_agent', 'wb') as f:
        pickle.dump(agent, f)
