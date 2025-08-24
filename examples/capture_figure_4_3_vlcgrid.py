import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tarware.warehouse import Warehouse
from tarware.definitions import RewardType

CELL_SIZE = 20

def draw_vlc_id_overlay(env, vlc_ids):
    height, width = env.grid_size
    fig, ax = plt.subplots(figsize=(width * 0.3, height * 0.3))
    ax.set_xlim(0, width * CELL_SIZE)
    ax.set_ylim(0, height * CELL_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")

    for y in range(height):
        for x in range(width):
            rect = patches.Rectangle((x * CELL_SIZE, (height - y - 1) * CELL_SIZE),
                                     CELL_SIZE, CELL_SIZE,
                                     linewidth=0.5, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)

            if (x, y) in vlc_ids:
                text = str(vlc_ids[(x, y)])
                ax.text(x * CELL_SIZE + CELL_SIZE / 2,
                        (height - y - 1) * CELL_SIZE + CELL_SIZE / 2,
                        text, color='black', ha='center', va='center', fontsize=8)

    fig.savefig("figure_4_3.png", bbox_inches="tight", dpi=300)
    print("✅ Saved as figure_4_3.png")


def main():
    env = Warehouse(
        shelf_columns=7,
        column_height=10,
        shelf_rows=2,
        num_agvs=10,
        num_pickers=4,
        request_queue_size=8,
        max_inactivity_steps=200,
        max_steps=2000,
        reward_type=RewardType.GLOBAL,
        observation_type="global",
        normalised_coordinates=False
    )

    env.reset()

    # Generate dummy VLC grid IDs across highways
    vlc_ids = {}
    id_counter = 1
    for y in range(env.grid_size[0]):
        for x in range(env.grid_size[1]):
            if env.highways[y, x]:
                vlc_ids[(x, y)] = id_counter
                id_counter += 1

    draw_vlc_id_overlay(env, vlc_ids)


if __name__ == "__main__":
    main()
