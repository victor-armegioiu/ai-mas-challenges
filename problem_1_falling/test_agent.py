from utils import read_cfg
from falling_objects_env import FallingObjects, PLAYER_KEYS, ACTIONS
from argparse import ArgumentParser
from demo_agent import DemoAgent
from dqn_agent import DDQNAgent
import importlib
import numpy as np
import cv2 as cv2

BATCH_SIZE = 32

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        '-c', '--config-file', default='configs/default.yaml', type=str,  dest='config_file',
        help='Default configuration file'
    )
    arg_parser.add_argument(
        '-a', '--agent', default='demo_agent+DemoAgent', type=str,  dest='agent',
        help='The agent to test in format <module_name>+<class_name>'
    )

    args = arg_parser.parse_args()
    config_file = args.config_file
    cfg = read_cfg(config_file)
    test_agent_name = args.agent.split("+")
    test_steps = cfg.test_steps
    test_agent = getattr(importlib.import_module(test_agent_name[0]), test_agent_name[1])

    print(f"Testing agent {test_agent_name[1]}")

    env = FallingObjects(cfg)

    #agent = test_agent(max(ACTIONS.keys()))

    # Dueling Deep Q-Learning Agent
    agent = DDQNAgent()
    all_r = 0
    obs = env.reset()

    # In lieu of having a state comprised of a single observation, we stack the last 3 images
    # at any given time in order to create a state, as suggested in DeepMind's DQN paper;
    # we do this in order to preserve the movement of the falling objects.
    s1, _, r1, _ = env.step(0)
    s2, _, r2, _ = env.step(0)
    s3, _, r3, _ = env.step(0)

    all_r += (r1 + r2 + r3)
    curr_obs = [s1, s2, s3]

    # Lambda function to reshape, convert to grayscale and stack the images in our observation list.
    make_obs = lambda obs_list : np.stack((cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY).reshape((1, 86, 86)) for obs in obs_list), axis=3)

    for i in range(test_steps):
    	# curr_obs is a list of the last 3 frames
        action = agent.act(make_obs(curr_obs))

        next_frame, r, done, _ = env.step(action)
        all_r += r

        print('STEP', i, ':', action, '->', all_r)

        # next_obs takes the last 2 entries in our initial observation and adds the current frame
        next_obs = curr_obs[1:] + [next_frame]

        # We cache the experiences in our replay buffer for further usage in our training steps.
        agent.remember(make_obs(curr_obs), action, r, make_obs(next_obs), done)

        curr_obs = next_obs
        agent.replay(min(len(agent.memory), BATCH_SIZE))


    print(f"Reward for {test_steps} steps: {all_r} ")
