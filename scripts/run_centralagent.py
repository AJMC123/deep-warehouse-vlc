import argparse
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

# our libs
from tarware.central_agent import CentralAgent
from tarware.vlc_grid import VLCGrid

def make_env(env_id: str, max_steps: int | None, render: bool):
    # DO NOT pass render_mode; this env doesn't accept it
    kwargs = dict(max_steps=max_steps) if max_steps is not None else {}
    env = gym.make(env_id, **kwargs)
    return env

def reset_env(env, seed=None):
    """Gymnasium 0.29 returns (obs, info). Older custom envs sometimes return only obs."""
    res = env.reset(seed=seed)
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        obs, info = res
    else:
        obs, info = res, {}
    return obs, info

def step_env(env, actions):
    """Always handle (obs, rewards, terminateds, truncateds, info)."""
    res = env.step(actions)
    if len(res) == 5:
        obs, rewards, terminateds, truncateds, info = res
    elif len(res) == 4:
        obs, rewards, dones, info = res
        terminateds = dones
        truncateds = dones
    else:
        raise RuntimeError("Unexpected env.step() return format.")
    return obs, rewards, terminateds, truncateds, info

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def write_json(path, payload):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def write_csv(path, rows, fieldnames):
    ensure_dir(path)
    first = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if first:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="Gym id, e.g. tarware-tiny-3agvs-2pickers-globalobs-v1")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--output", default="logs/central_metrics.json")
    ap.add_argument("--csv", default="logs/central_metrics.csv")
    args = ap.parse_args()

    env = make_env(args.env, args.max_steps, args.render)
    # use the naked env for attributes (masks, maps, etc.)
    env_unwrapped = env.unwrapped

    vlc = VLCGrid()
    planner = CentralAgent(vlc)

    all_ep_rows = []
    for ep in range(args.episodes):
        planner.reset_metrics()
        obs, _ = reset_env(env, seed=args.seed)

        done = False
        total_reward = 0.0
        steps = 0

        # render only *after* reset per Gymnasium's order enforcer
        if args.render:
            env_unwrapped.render()

        while not done:
            # valid-action mask from the environment (shape: [num_agents, action_size])
            masks = env_unwrapped.compute_valid_action_masks()
            # plan macro actions (list[int], one per agent)
            actions = planner.plan(obs, masks, env_unwrapped)

            obs, rewards, terminateds, truncateds, info = step_env(env, actions)

            # rewards is a list (per agent) → sum for episode return
            step_return = float(np.sum(rewards))
            total_reward += step_return
            steps += 1

            # accumulate metrics inside the planner
            planner.update_metrics(info, step_return)

            if args.render:
                env_unwrapped.render()

            done = bool(np.all(terminateds)) or bool(np.all(truncateds))
            if args.max_steps is not None and steps >= args.max_steps:
                break

        # collect one row per episode
        m = planner.get_metrics()
        row = {
            "episode": ep,
            "steps": steps,
            "total_reward": round(float(total_reward), 6),
            "shelf_deliveries": int(m.get("shelf_deliveries", 0)),
            "clashes": int(m.get("clashes", 0)),
            "stucks": int(m.get("stucks", 0)),
            "agvs_distance_travelled": int(m.get("agvs_distance_travelled", 0)),
            "pickers_distance_travelled": int(m.get("pickers_distance_travelled", 0)),
            "agvs_idle_time": int(m.get("agvs_idle_time", 0)),
            "pickers_idle_time": int(m.get("pickers_idle_time", 0)),
        }
        all_ep_rows.append(row)

    # write files
    write_json(args.output, all_ep_rows)
    write_csv(
        args.csv,
        all_ep_rows,
        fieldnames=list(all_ep_rows[0].keys()),
    )

    env.close()

if __name__ == "__main__":
    main()
