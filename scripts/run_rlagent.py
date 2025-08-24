import argparse
import json, csv
from tarware.rl_agent import RLAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='tarware-tiny-3agvs-2pickers-globalobs-v1')
    parser.add_argument('--algo', choices=['DQN','PPO'], default='DQN')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--output', default='rl_metrics.json')
    parser.add_argument('--csv', default='rl_metrics.csv')
    args = parser.parse_args()

    agent = RLAgent(args.env, algo=args.algo)
    agent.train(timesteps=args.timesteps)
    results = agent.evaluate(episodes=args.episodes, max_steps=args.max_steps)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.csv, 'w') as f:
        f.write("episode, reward\n")
        for i, r in enumerate(results):
            f.write(f"{i},{r}\n")

if __name__ == '__main__':
    main()
