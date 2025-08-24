import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_env(env_id, render=False):
    # evita passar render_mode a envs que não suportam
    env = gym.make(env_id, disable_env_checker=True)
    return env

def reset_env(env, seed=None):
    """Compatível com Gymnasium: reset -> (obs, info)"""
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}

def step_env(env, action):
    """
    Compatível com Gymnasium: step -> (obs, reward, terminated, truncated, info)
    Alguns envs antigos devolvem listas; normalizamos para floats/bools.
    """
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    elif len(out) == 4:
        # fallback muito antigo
        obs, reward, done, info = out
        return obs, reward, bool(done), info
    else:
        raise RuntimeError(f"Formato inesperado de step: len={len(out)}")

def maybe_render(env, do_render):
    if not do_render:
        return
    try:
        env.unwrapped.render()
    except Exception:
        # Se não tiver pyglet instalado, falha silenciosa (ou instala pyglet para janela)
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        help="ID do ambiente Gym (ex: tarware-tiny-3agvs-2pickers-globalobs-v1)")
    parser.add_argument("--model", type=str, required=True,
                        help="Caminho para o .zip do modelo PPO (ex: logs/ppo/best_model.zip)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Ações determinísticas")
    args = parser.parse_args()

    print(f"[EVAL] Carregar modelo: {args.model}")
    model = PPO.load(args.model, device="cpu")

    env = make_env(args.env, render=args.render)

    all_rewards = []

    for ep in range(1, args.episodes + 1):
        print(f"\n===== EVAL EP {ep}/{args.episodes} =====")
        obs, _ = reset_env(env, seed=args.seed)
        ep_reward = 0.0
        steps = 0

        maybe_render(env, args.render)

        while steps < args.max_steps:
            # SB3 aceita obs no formato que o env fornece (tuple/np arrays). Não force reshape aqui.
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = step_env(env, action)

            # reward pode ser lista/np.array -> somamos tudo para métrica global
            if isinstance(reward, (list, tuple, np.ndarray)):
                ep_reward += float(np.sum(reward))
            else:
                ep_reward += float(reward)

            steps += 1
            maybe_render(env, args.render)

            if done:
                break

        print(f"[EVAL] Episódio {ep}: reward_total={ep_reward:.3f} | steps={steps}")
        all_rewards.append(ep_reward)

        # pequena pausa para deixar janela atualizar
        if args.render:
            time.sleep(0.1)

    if all_rewards:
        print("\n--- RESUMO ---")
        print(f"Episódios: {len(all_rewards)}")
        print(f"Reward média: {np.mean(all_rewards):.3f}")
        print(f"Reward std   : {np.std(all_rewards):.3f}")
        print(f"Reward min/max: {np.min(all_rewards):.3f} / {np.max(all_rewards):.3f}")

    env.close()


if __name__ == "__main__":
    main()
