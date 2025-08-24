# scripts/render_vlc_layout.py
import sys
from pathlib import Path

# garantir import do pacote 'tarware' quando se corre "python scripts/..."
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tarware.warehouse import Warehouse
from tarware.definitions import RewardType

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def safe_font(px):
    """Tenta um TTF proporcional ao tamanho da célula; senão usa o default."""
    # tamanho do texto ~ 60% da célula
    size = max(10, int(px * 0.6))
    for name in ["arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def main(out_path="logs/figures/vlc_overlay_layout.png", seed=123):
    # 1) instanciar ambiente (sem render_mode; a tua env não usa kwargs de render)
    env = Warehouse(
        shelf_columns=3,          # "tiny": ímpar
        column_height=8,
        shelf_rows=1,
        num_agvs=3,
        num_pickers=2,
        request_queue_size=20,
        max_inactivity_steps=None,
        max_steps=1,
        reward_type=RewardType.INDIVIDUAL,
        observation_type="global",
        normalised_coordinates=False,
    )
    env.reset(seed=seed)

    # 2) obter o frame base (sem overlays custom) e metadados do viewer
    frame = env.render(mode="rgb_array")  # retorna np.uint8 [H,W,3]
    viewer = env.renderer
    cell_px = viewer.grid_size  # tamanho da célula em pixéis
    rows, cols = env.grid_size  # grelha lógica (linhas, colunas)

    # 3) preparar imagem PIL
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    font = safe_font(cell_px)

    # 4) Função util para centro de célula → coordenadas em pixéis
    #    Nota: o render inverte o eixo Y (0 em baixo), por isso fazemos flip
    def cell_center(x, y):
        ry = rows - y - 1
        cx = (cell_px + 1) * x + cell_px // 2 + 1
        cy = (cell_px + 1) * ry + cell_px // 2 + 1
        return cx, cy

    # 5) letras e cores (só TEXTO, sem quadrados)
    letters = ["R", "G", "B", "M"]
    colors = [(255, 0, 0, 255), (0, 170, 0, 255), (0, 120, 255, 255), (210, 0, 210, 255)]

    # 6) desenhar UMA VEZ em TODAS as células (highway + racks + goals)
    idx = 0
    for y in range(rows):
        for x in range(cols):
            cx, cy = cell_center(x, y)
            ch = letters[idx % len(letters)]
            col = colors[idx % len(colors)]
            # centraliza no centro da célula
            draw.text((cx, cy), ch, fill=col, font=font, anchor="mm")
            idx += 1

    # 7) guardar resultado
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    env.close()
    print(f"[OK] guardado: {out_path.resolve()}")


if __name__ == "__main__":
    main()
