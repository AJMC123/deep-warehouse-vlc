# scripts/callbacks.py
import os
from typing import Optional

from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)

def build_callbacks(
    eval_env,
    eval_freq: int = 10_000,
    eval_episodes: int = 5,
    save_path: str = "logs/ppo",
    checkpoint_freq: int = 50_000,
    patience_evals: Optional[int] = 10,
) -> CallbackList:
    """
    Devolve uma CallbackList com:
      - EvalCallback: avalia e guarda best_model.zip
      - CheckpointCallback: checkpoints regulares
      - (opcional) Early stop se não houver melhoria por N avaliações
    """
    os.makedirs(save_path, exist_ok=True)

    early_stop = None
    if patience_evals is not None and patience_evals > 0:
        # Nota: versões SB3 antigas não têm min_delta
        early_stop = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=patience_evals,
            min_evals=1,
            verbose=1,
        )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=early_stop,  # ok: é um Callback
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    callbacks = [eval_cb, ckpt_cb] if early_stop is None else [eval_cb, ckpt_cb, early_stop]
    return CallbackList(callbacks)
