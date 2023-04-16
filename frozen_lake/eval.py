from IPython.display import clear_output
from time import sleep
import pickle
import gym
import os

# PARAMS
EPISODES = 10
action_map = {0: "Left", 1: "Down", 2: "Right", 3: "Up", }

def replay_frames(frames, actions):

    for i, frame in enumerate(frames):
        os.system("clear")
        clear_output(wait=True)
        print(action_map[actions[i]])
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {action_map[frame['action']]}")
        print(f"Reward: {frame['reward']}")
        sleep(2)

with open('trained_q_agent', 'rb') as f:
    trained_agent = pickle.load(f)

""" Evaluate Agent """
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    agent = trained_agent

    scores = []
    win_pct_list = []

    for episode in range(EPISODES):
        done = False
        # Initial Observation-State: Observation == External Representation; State == Internal Representation
        observation = env.reset()
        score = 0
        frames = []
        actions = []

        # Play Episode
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            # Update State-Action Q-Values
            #agent.learn(observation, action, reward, observation_)

            score += reward

            actions.append(action)
            frames.append({
                "frame": env.render(mode="ansi"),
                "state": observation_,
                "action": action,
                "reward": reward
            })

            observation = observation_

        replay_frames(frames, actions)
        scores.append(score)


