# scripts/capture_figure_4_2.py
import os, sys
from pathlib import Path

# garante que o pacote tarware é encontrado quando corres "python scripts/..."
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # pip install pillow
import numpy as np

from tarware.warehouse import Warehouse
from tarware.definitions import RewardType
from tarware.central_agent import CentralAgent
from tarware.vlc_grid import VLCGrid



def make_tiny_env():
    # tarware-tiny-3agvs-2pickers-globalobs-v1 (equivalente)
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        num_agvs=3,
        num_pickers=2,
        request_queue_size=20,
        max_inactivity_steps=None,
        max_steps=500,
        reward_type=RewardType.INDIVIDUAL,
        observation_type="global",
        normalised_coordinates=False,
    )
    return env


def save_frame(frame: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out_path)
    print(f"[OK] Snapshot gravado em: {out_path}")


def main():
    env = make_tiny_env()
    vlc = VLCGrid()
    planner = CentralAgent(vlc)

    # reset
    obs = env.reset(seed=42)
    planner.reset_metrics()

    snapshot_saved = False
    out_file = ROOT / "logs" / "figures" / "figure_4_2.png"

    # corre alguns steps até apanhar “boa ação”
    # critério: quando houver pelo menos um AGV ocupado e uma shelf em request_queue
    for step in range(1, 400 + 1):
        # máscaras de ações válidas vindas do próprio env
        masks = env.compute_valid_action_masks(
            pickers_to_agvs=True, block_conflicting_actions=True
        )

        # planeamento central → macro_actions (um id por agente)
        macro_actions = planner.plan(obs, masks, env)

        # step
        obs, rewards, terminateds, truncateds, info = env.step(macro_actions)

        # guarda uma imagem quando o cenário está “interessante”
        # (AGVs ocupados e pelo menos 1 pedido ativo; normalmente acontece cedo)
        vehicles_busy = info.get("vehicles_busy", [])
        busy_agvs = any(vehicles_busy[: env.num_agvs]) if vehicles_busy else False
        has_requests = len(env.request_queue) > 0

        if (busy_agvs and has_requests) and not snapshot_saved:
            frame = env.render(mode="rgb_array")
            save_frame(frame, out_file)
            snapshot_saved = True

        # fallback: se não apanhou momento “interessante”, força no step 120
        if (step == 120) and not snapshot_saved:
            frame = env.render(mode="rgb_array")
            save_frame(frame, out_file)
            snapshot_saved = True

        # termina se episódio acabou
        if all(terminateds) or all(truncateds):
            break

    # se nunca renderizámos, ainda assim grava o último frame
    if not snapshot_saved:
        frame = env.render(mode="rgb_array")
        save_frame(frame, out_file)

    env.close()
    print("[DONE] Figure 4.2 pronta.")


if __name__ == "__main__":
    main()
