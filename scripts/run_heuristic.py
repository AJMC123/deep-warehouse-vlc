import argparse
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

# our heuristic runner
from tarware.heuristic import heuristic_episode

def make_env(env_id: str, max_steps: int | None, render: bool):
    # DO NOT pass render_mode; this env doesn't accept it
    kwargs = dict(max_steps=max_steps) if max_steps is not None else {}
    env = gym.make(env_id, **kwargs)
    return env

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

def aggregate_infos(infos_list):
    """infos_list: list of `info` dicts returned each step; sum the counters."""
    agg = {
        "shelf_deliveries": 0,
        "clashes": 0,
        "stucks": 0,
        "agvs_distance_travelled": 0,
        "pickers_distance_travelled": 0,
        "agvs_idle_time": 0,
        "pickers_idle_time": 0,
    }
    for inf in infos_list:
        for k in agg.keys():
            if k in inf:
                # some infos may be scalar or list; make them ints
                val = inf[k]
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = int(np.sum(val))
                agg[k] += int(val)
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="Gym id, e.g. tarware-tiny-3agvs-2pickers-globalobs-v1")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--output", default="logs/heuristic_metrics.json")
    ap.add_argument("--csv", default="logs/heuristic_metrics.csv")
    args = ap.parse_args()

    env = make_env(args.env, args.max_steps, args.render)

    rows = []
    for ep in range(args.episodes):
        # IMPORTANT: pass the naked env to the heuristic to avoid OrderEnforcing attr errors
        infos, global_return, per_agent_returns = heuristic_episode(
            env.unwrapped,
            render=args.render,
            seed=args.seed,
        )

        agg = aggregate_infos(infos)
        steps = len(infos)

        row = {
            "episode": ep,
            "steps": steps,
            "total_reward": round(float(global_return), 6),
            "shelf_deliveries": int(agg["shelf_deliveries"]),
            "clashes": int(agg["clashes"]),
            "stucks": int(agg["stucks"]),
            "agvs_distance_travelled": int(agg["agvs_distance_travelled"]),
            "pickers_distance_travelled": int(agg["pickers_distance_travelled"]),
            "agvs_idle_time": int(agg["agvs_idle_time"]),
            "pickers_idle_time": int(agg["pickers_idle_time"]),
        }
        rows.append(row)

    write_json(args.output, rows)
    write_csv(args.csv, rows, fieldnames=list(rows[0].keys()))

    env.close()

if __name__ == "__main__":
    main()
