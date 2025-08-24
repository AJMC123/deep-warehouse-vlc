import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from tarware.wrappers import TupleToMultiDiscrete
from scripts.callbacks import build_callbacks


def make_env(env_id: str, seed: int):
    def _thunk():
        env = gym.make(env_id, disable_env_checker=True)
        env = TupleToMultiDiscrete(env)
        env.reset(seed=seed)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tarware-tiny-3agvs-2pickers-globalobs-v1")
    parser.add_argument("--steps", type=int, default=200_000, help="Total timesteps de treino")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", default="logs/ppo", help="Pasta de logs, checkpoints e best_model.zip")
    parser.add_argument("--eval_freq", type=int, default=10_000, help="Avaliar a cada N steps")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Episódios por avaliação")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000, help="Checkpoint a cada N steps")
    parser.add_argument("--tensorboard", action="store_true", help="Ativar TensorBoard logging")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    set_random_seed(args.seed)

    # VecEnv para PPO
    env = DummyVecEnv([make_env(args.env, args.seed)])
    eval_env = DummyVecEnv([make_env(args.env, args.seed + 100)])

    # Callbacks (eval + checkpoints + opcional early-stop)
    callbacks = build_callbacks(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_path=args.logdir,
        checkpoint_freq=args.checkpoint_freq,
        patience_evals=0,
    )

    

    # Política MLP padrão; podes ajustar net_arch se quiseres
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.logdir if args.tensorboard else None,
        n_steps=2048,          # reduz se memória for curta (ex. 1024)
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    print(f"[PPO] Treino a iniciar por {args.steps} timesteps…")
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    final_path = os.path.join(args.logdir, "ppo_central_final.zip")
    model.save(final_path)
    print(f"[PPO] Treino concluído. Modelo final guardado em: {final_path}")
    print(f"[PPO] Melhor modelo (com EvalCallback): {os.path.join(args.logdir, 'best_model.zip') if os.path.exists(os.path.join(args.logdir,'best_model.zip')) else 'ainda não gerado'}")


if __name__ == "__main__":
    main()
