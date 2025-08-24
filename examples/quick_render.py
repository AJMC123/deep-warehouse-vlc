from time import sleep

# If the env is exported here; if not, adjust import to whatever the original exposes.
from tarware.warehouse import WarehouseEnv

# IMPORTANT for Gymnasium 0.29+: create with render_mode="human"
env = WarehouseEnv(render_mode="human")

# Optional: Some repos forget this; if your class doesn't define it, set it:
if not hasattr(env, "metadata"):
    env.metadata = {"render_modes": ["human"], "render_fps": 10}
elif "render_fps" not in env.metadata:
    env.metadata["render_fps"] = 10

obs, info = env.reset()
for _ in range(300):
    # take a random action per agent if multi-agent, otherwise sample once
    try:
        action = env.action_space.sample()
    except Exception:
        # fallback if it's a list of spaces
        action = [sp.sample() for sp in env.action_space]

    obs, reward, terminated, truncated, info = env.step(action)
    # render is driven by env (don’t pass custom kwargs)
    env.render()

    if isinstance(terminated, (list, tuple)) or isinstance(truncated, (list, tuple)):
        done = any(terminated) or any(truncated)
    else:
        done = bool(terminated) or bool(truncated)

    if done:
        obs, info = env.reset()

env.close()
