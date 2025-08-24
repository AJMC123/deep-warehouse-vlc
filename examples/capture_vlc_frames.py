# scripts/capture_vlc_frames.py
import os
import imageio
import gymnasium as gym
import tarware  # ensures all tarware envs are registered

ENV_ID = "tarware-tiny-3agvs-2pickers-globalobs-v1"
OUTDIR = "plots_thesis"
N_FRAMES = 12

os.makedirs(OUTDIR, exist_ok=True)

# IMPORTANT: disable the env checker so we can pass mode to render
env = gym.make(ENV_ID, disable_env_checker=True)

# Your env.reset() returns only obs (tuple for each agent). Handle both cases just in case.
ret = env.reset()
obs = ret[0] if isinstance(ret, tuple) and len(ret) == 2 else ret

for i in range(N_FRAMES):
    # Bypass wrappers and call your env's render signature directly
    frame = env.unwrapped.render(mode="rgb_array")
    path = os.path.join(OUTDIR, f"vlc_frame_{i+1:02d}.png")
    imageio.imwrite(path, frame)
    print(f"[OK] saved {path}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if any(terminated) or any(truncated):
        ret = env.reset()
        obs = ret[0] if isinstance(ret, tuple) and len(ret) == 2 else ret

env.close()
print(f"[DONE] {N_FRAMES} frames saved to {OUTDIR}")
