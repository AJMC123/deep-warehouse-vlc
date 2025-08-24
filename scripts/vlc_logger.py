import argparse
import csv
import json
import os
from typing import Dict, Any, List, Tuple

import gymnasium as gym
import numpy as np

from tarware.central_agent import CentralAgent
from tarware.vlc_grid import VLCGrid


def _reset_compat(env, seed=None):
    res = env.reset(seed=seed)
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        return res
    return res, {}


def _step_compat(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        return out  # obs, rewards, terminateds, truncateds, info
    if isinstance(out, tuple) and len(out) == 4:
        obs, rewards, dones, info = out
        return obs, rewards, dones, dones, info
    raise RuntimeError(f"Formato inesperado em env.step(): {type(out)} len={len(out) if isinstance(out, tuple) else 'n/a'}")


def _is_goal(unwrapped, x: int, y: int) -> bool:
    # goals são lista de (y, x) no env; aqui recebemos (x, y)
    return (x, y) in [(gx, gy) for (gy, gx) in unwrapped.goals]


def _collect_vlc_step_metrics(unwrapped) -> Dict[str, Any]:
    """
    Retorna métricas VLC para o step atual:
      - flags por picker: on_highway / at_goal
      - totais: pickers_on_highway, pickers_off_highway, agvs_on_highway, agvs_off_highway
      - violation: True se algum picker estiver fora de highway e fora de goal
    """
    highways = unwrapped.highways  # matriz [y, x] em {0,1}
    num_agvs = unwrapped.num_agvs
    num_agents = unwrapped.num_agents

    picker_flags = []
    for ag in unwrapped.agents[num_agvs:]:
        on_hw = bool(highways[ag.y, ag.x] == 1)
        at_goal = _is_goal(unwrapped, ag.x, ag.y)
        picker_flags.append({
            "id": ag.id,
            "x": ag.x,
            "y": ag.y,
            "on_highway": on_hw,
            "at_goal": at_goal
        })

    # Contagens agregadas para pickers
    pickers_on_highway = sum(1 for f in picker_flags if f["on_highway"] or f["at_goal"])
    pickers_off_highway = len(picker_flags) - pickers_on_highway
    picker_violation = pickers_off_highway > 0

    # AGVs: só para referência de uso de VLC (não é proibição)
    agv_flags = []
    for ag in unwrapped.agents[:num_agvs]:
        on_hw = bool(highways[ag.y, ag.x] == 1)
        agv_flags.append({
            "id": ag.id,
            "x": ag.x,
            "y": ag.y,
            "on_highway": on_hw
        })
    agvs_on_highway = sum(1 for f in agv_flags if f["on_highway"])
    agvs_off_highway = len(agv_flags) - agvs_on_highway

    return {
        "picker_flags": picker_flags,
        "agv_flags": agv_flags,
        "pickers_on_highway": int(pickers_on_highway),
        "pickers_off_highway": int(pickers_off_highway),
        "agvs_on_highway": int(agvs_on_highway),
        "agvs_off_highway": int(agvs_off_highway),
        "picker_violation": bool(picker_violation),
    }


def main():
    parser = argparse.ArgumentParser(description="VLC usage logger (sem alterar código do env).")
    parser.add_argument("--env", default="tarware-tiny-3agvs-2pickers-globalobs-v1")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps_csv", default="logs/vlc_usage_steps.csv")
    parser.add_argument("--summary_json", default="logs/vlc_usage_summary.json")
    parser.add_argument("--every", type=int, default=50, help="Print parcial a cada N steps")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.steps_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    # cria env (sem mexer no código)
    env = gym.make(args.env, disable_env_checker=True)
    unwrapped = env.unwrapped

    # planeador central que já tens
    agent = CentralAgent(vlc=VLCGrid())

    # CSV por step
    step_fields = [
        "episode", "step",
        "pickers_on_highway", "pickers_off_highway", "picker_violation",
        "agvs_on_highway", "agvs_off_highway",
        # flags por picker (até 16 pickers para não explodir; ajusta se precisares)
    ]
    max_pickers_to_log = max(2, unwrapped.num_pickers)  # tenta abranger todos
    for i in range(max_pickers_to_log):
        step_fields += [f"picker{i}_x", f"picker{i}_y", f"picker{i}_on_highway", f"picker{i}_at_goal"]

    with open(args.steps_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=step_fields)
        writer.writeheader()

        all_summaries: List[Dict[str, Any]] = []

        for ep in range(args.episodes):
            obs, info = _reset_compat(env, seed=args.seed + ep)
            agent.reset_metrics()

            episode_viols = 0
            episode_steps = 0
            episode_pickers_on_total = 0
            episode_pickers_off_total = 0
            episode_agvs_on_total = 0
            episode_agvs_off_total = 0

            for t in range(args.max_steps):
                episode_steps = t + 1

                masks = unwrapped.compute_valid_action_masks()
                macro_actions = agent.plan(obs, masks, unwrapped)

                # step
                obs, rewards, term, trunc, info = _step_compat(env, macro_actions)

                # métricas VLC deste step
                vlc = _collect_vlc_step_metrics(unwrapped)
                episode_pickers_on_total += vlc["pickers_on_highway"]
                episode_pickers_off_total += vlc["pickers_off_highway"]
                episode_agvs_on_total += vlc["agvs_on_highway"]
                episode_agvs_off_total += vlc["agvs_off_highway"]
                episode_viols += int(vlc["picker_violation"])

                # linha CSV
                row = {
                    "episode": ep + 1,
                    "step": episode_steps,
                    "pickers_on_highway": vlc["pickers_on_highway"],
                    "pickers_off_highway": vlc["pickers_off_highway"],
                    "picker_violation": int(vlc["picker_violation"]),
                    "agvs_on_highway": vlc["agvs_on_highway"],
                    "agvs_off_highway": vlc["agvs_off_highway"],
                }
                # flags por picker
                for i in range(max_pickers_to_log):
                    if i < len(vlc["picker_flags"]):
                        pf = vlc["picker_flags"][i]
                        row[f"picker{i}_x"] = pf["x"]
                        row[f"picker{i}_y"] = pf["y"]
                        row[f"picker{i}_on_highway"] = int(pf["on_highway"])
                        row[f"picker{i}_at_goal"] = int(pf["at_goal"])
                    else:
                        row[f"picker{i}_x"] = ""
                        row[f"picker{i}_y"] = ""
                        row[f"picker{i}_on_highway"] = ""
                        row[f"picker{i}_at_goal"] = ""
                writer.writerow(row)

                # prints ocasionais
                if episode_steps % max(1, args.every) == 0:
                    print(f"[EP {ep+1}] step={episode_steps} | pickers_on={vlc['pickers_on_highway']} "
                          f"off={vlc['pickers_off_highway']} viol={int(vlc['picker_violation'])}")

                if args.render:
                    env.render()

                done = (isinstance(term, (list, np.ndarray)) and all(term)) or (isinstance(trunc, (list, np.ndarray)) and all(trunc))
                if done:
                    print(f"[EP {ep+1}] terminado aos {episode_steps} steps.")
                    break

            # resumo por episódio
            num_pickers = unwrapped.num_pickers
            ep_summary = {
                "episode": ep + 1,
                "steps": episode_steps,
                "picker_violation_steps": episode_viols,
                "picker_violation_rate": (episode_viols / float(episode_steps)) if episode_steps > 0 else 0.0,
                "avg_pickers_on_highway_per_step": episode_pickers_on_total / float(episode_steps),
                "avg_pickers_off_highway_per_step": episode_pickers_off_total / float(episode_steps),
                "avg_agvs_on_highway_per_step": episode_agvs_on_total / float(episode_steps),
                "avg_agvs_off_highway_per_step": episode_agvs_off_total / float(episode_steps),
                "num_pickers": num_pickers,
            }
            all_summaries.append(ep_summary)
            print("[Resumo EP]:", json.dumps(ep_summary, indent=2))

    # guardar JSON resumo (lista de episódios)
    with open(args.summary_json, "w", encoding="utf-8") as fj:
        json.dump(all_summaries, fj, indent=2)
    print("\nVLC usage logs salvos em:")
    print("  CSV  (steps):", args.steps_csv)
    print("  JSON (epis.):", args.summary_json)


if __name__ == "__main__":
    main()
