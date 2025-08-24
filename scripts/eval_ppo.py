# scripts/eval_ppo.py
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as SpaceDict
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
import tarware  # garante o registo das envs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--render", action="store_true")
    # como agregar recompensas de vários agentes: sum | mean | first
    p.add_argument("--reward_reduce", type=str, default="sum", choices=["sum", "mean", "first"])
    return p.parse_args()

def make_env(env_id: str, want_render: bool):
    manual_render = False
    if want_render:
        try:
            env = gym.make(env_id, render_mode="human")
            if getattr(env, "render_mode", None) != "human":
                manual_render = True
        except TypeError:
            env = gym.make(env_id)
            manual_render = True
    else:
        env = gym.make(env_id)
    return env, manual_render

class MultiAgentToSingle(gym.Wrapper):
    """
    Compatibiliza env multi-agente que devolve listas/arrays por agente
    para formato single-agente:
      - obs: concatena/flatten para 1D
      - reward: reduz com sum/mean/first
      - terminated/truncated: any() sobre a lista
    """
    def __init__(self, env, reward_reduce="sum"):
        super().__init__(env)
        assert reward_reduce in ("sum", "mean", "first")
        self.reward_reduce = reward_reduce

        # Se observation_space for Box multi-dim, vamos achatar a seguir com FlattenObservation
        if isinstance(self.observation_space, SpaceDict):
            pass  # FlattenObservation tratará
        elif isinstance(self.observation_space, Box):
            # se já for 1D tudo bem; se não, FlattenObservation trata
            pass

    def _reduce_reward(self, r):
        if isinstance(r, (list, tuple, np.ndarray)):
            arr = np.asarray(r, dtype=np.float32)
            if self.reward_reduce == "sum":
                return float(np.sum(arr))
            elif self.reward_reduce == "mean":
                return float(np.mean(arr))
            else:
                return float(arr[0])
        return float(r)

    def _bool_any(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return bool(np.any(x))
        return bool(x)

    def _flatten_obs(self, obs):
        # obs pode ser lista/tuple por agente, ou matriz (n_agents, feat)
        if isinstance(obs, (list, tuple)):
            try:
                obs = np.asarray(obs)
            except Exception:
                # fallback: flatten elemento a elemento
                obs = np.concatenate([np.asarray(o).reshape(-1) for o in obs], axis=0)
        arr = np.asarray(obs)
        return arr.reshape(-1)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        obs = self._flatten_obs(obs)
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated = done
            truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False

        obs = self._flatten_obs(obs)
        reward = self._reduce_reward(reward)
        terminated = self._bool_any(terminated)
        truncated = self._bool_any(truncated)
        return obs, reward, terminated, truncated, info

def wrap_flatten(env: gym.Env) -> gym.Env:
    """Garante flatten final (Box multi-dim ou Dict)."""
    if isinstance(env.observation_space, (SpaceDict,)):
        env = FlattenObservation(env)
    elif isinstance(env.observation_space, Box) and len(env.observation_space.shape) > 1:
        env = FlattenObservation(env)
    return env

def reset_compat(env, seed=None):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    # obs já deverá vir 1D pelo wrapper; garantir:
    obs = np.asarray(obs).reshape(-1)
    return obs, info

def main():
    args = parse_args()

    model = PPO.load(args.model)

    env, manual_render = make_env(args.env, args.render)
    env = MultiAgentToSingle(env, reward_reduce=args.reward_reduce)
    env = wrap_flatten(env)  # redundante, mas seguro

    # seed
    try:
        env.reset(seed=args.seed)
    except TypeError:
        try:
            env.seed(args.seed)
        except Exception:
            pass

    ep_rewards, ep_lengths = [], []

    for ep in range(args.episodes):
        obs, info = reset_compat(env, seed=args.seed)
        terminated = truncated = False
        total_r = 0.0
        steps = 0

        while not (terminated or truncated):
            if args.render and manual_render:
                try:
                    env.render()
                except Exception:
                    pass

            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)  # wrapper já compatibiliza
            total_r += float(r)
            steps += 1
            if steps >= args.max_steps:
                truncated = True

        ep_rewards.append(total_r)
        ep_lengths.append(steps)

        extra = []
        if isinstance(info, dict):
            for key in ["deliveries", "collisions", "stucks", "pickrate", "global_reward"]:
                if key in info:
                    extra.append(f"{key}={info[key]}")
        extra_str = (" | " + ", ".join(extra)) if extra else ""
        print(f"[Eval] Episode {ep+1}/{args.episodes}: Return={total_r:.3f}, Steps={steps}{extra_str}")

    print("\n=== Summary ===")
    print(f"Episodes: {args.episodes}")
    print(f"Avg Return: {np.mean(ep_rewards):.3f}  (± {np.std(ep_rewards):.3f})")
    print(f"Avg Steps:  {np.mean(ep_lengths):.1f}")
    env.close()

if __name__ == "__main__":
    main()
