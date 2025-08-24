# scripts/capture_vlc_rgmb_letters.py
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import tarware  # registers tarware-* envs

ENV_ID   = "tarware-tiny-3agvs-2pickers-globalobs-v1"
OUTDIR   = "plots_thesis"
N_FRAMES = 4
SEED     = 42

os.makedirs(OUTDIR, exist_ok=True)
env = gym.make(ENV_ID, disable_env_checker=True)
env.reset(seed=SEED)

# RG B M mapping
ORDER  = ["R", "G", "B", "M"]
COLORS = {
    "R": (0.90, 0.25, 0.25),
    "G": (0.25, 0.75, 0.35),
    "B": (0.25, 0.45, 0.90),
    "M": (0.75, 0.35, 0.85),
}
ALPHA_BG = 0.18  # translucent tint behind the letter

def save_frame(idx):
    w = env.unwrapped
    H, W = w.grid_size

    # base image: white
    img = np.ones((H, W, 3), dtype=float)

    # draw shelves (same purple as renderer vibe)
    SHELVES = w.grid[w.grid.shape[0]-4]  # CollisionLayers.SHELVES
    img[SHELVES > 0] = (0.33, 0.25, 0.55)

    # draw goals (dark bays)
    for (x, y) in w.goals:
        img[y, x] = (0.25, 0.27, 0.29)

    fig = plt.figure(figsize=(6.0, 6.0))
    ax  = plt.gca()
    ax.imshow(img, interpolation="nearest", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Warehouse layout with VLC RGMB cell overlay (letters)")

    # overlay RGMB: compute pattern on highway cells only
    hw = w.highways.astype(bool)
    yy, xx = np.indices((H, W))
    ring = (xx + yy) % 4  # 0..3 -> R,G,B,M

    # draw a faint colored square + centered letter on each highway tile
    for k, key in enumerate(ORDER):
        mask = np.where(hw & (ring == k))
        for y, x in zip(mask[0], mask[1]):
            # translucent background tint
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1,
                                       facecolor=COLORS[key], edgecolor='none',
                                       alpha=ALPHA_BG))
            # bold letter
            ax.text(x, y, key,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=COLORS[key])

    # plot AGVs (circle outline) and Pickers (diamond outline)
    agv_layer = w.grid[-3]
    pik_layer = w.grid[-2]
    ys, xs = np.where(agv_layer > 0)
    ax.scatter(xs, ys, s=160, marker="o", facecolors="none", edgecolor="k", linewidth=0.9, label="AGV")
    ys, xs = np.where(pik_layer > 0)
    ax.scatter(xs, ys, s=200, marker="D", facecolors="none", edgecolor="k", linewidth=0.9, label="Picker")

    # legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=(0.33,0.25,0.55), edgecolor='none', label="Rack shelf"),
        Patch(facecolor=(0.25,0.27,0.29), edgecolor='none', label="Goal bay"),
        Line2D([0],[0], marker='o', color='k', markerfacecolor='none', linewidth=0, label='AGV'),
        Line2D([0],[0], marker='D', color='k', markerfacecolor='none', linewidth=0, label='Picker'),
    ] + [Patch(facecolor=COLORS[k], edgecolor='none', alpha=ALPHA_BG, label=f"VLC cell: {k}") for k in ORDER]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8)

    plt.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, f"vlc_rgmb_letters_{idx:02d}.png")
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[OK] {out}")

for i in range(1, N_FRAMES+1):
    save_frame(i)
    # advance the env so agents move between frames
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if any(terminated) or any(truncated):
        env.reset(seed=SEED)

env.close()
print("[DONE]")
