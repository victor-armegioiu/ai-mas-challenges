from utils import read_cfg
from falling_objects_env import FallingObjects, PLAYER_KEYS, ACTIONS
from argparse import ArgumentParser
from demo_agent import DemoAgent
from dqn_agent import DQNAgent
import importlib

BATCH_SIZE = 50

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
    agent = DQNAgent()
    all_r = 0
    obs = env.reset()

    for i in range(test_steps):
        action = agent.act(obs)

        next_obs, r, done, _ = env.step(action)
        agent.remember(obs, action, r, next_obs, done)
        obs = next_obs

        all_r += r
        agent.replay(min(BATCH_SIZE, len(agent.memory)))

    print(f"Reward for {test_steps} steps: {all_r} ")
