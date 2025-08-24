import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_steps_csv(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def compute_percent_series(rows: List[dict]) -> Dict[int, List[Tuple[int, float]]]:
    """
    Devolve {episode: [(step, pct_on_highway), ...]}.
    pct = 100 * pickers_on_highway / (pickers_on_highway + pickers_off_highway)
    """
    by_ep: Dict[int, List[Tuple[int, float]]] = {}
    for r in rows:
        ep = safe_int(r.get("episode", 1), 1)
        step = safe_int(r.get("step", 0), 0)
        on_hw = safe_int(r.get("pickers_on_highway", 0), 0)
        off_hw = safe_int(r.get("pickers_off_highway", 0), 0)
        denom = on_hw + off_hw
        pct = 100.0 * on_hw / denom if denom > 0 else 0.0
        by_ep.setdefault(ep, []).append((step, pct))
    # ordena por step
    for ep, series in by_ep.items():
        series.sort(key=lambda t: t[0])
    return by_ep


def moving_average(xs: List[float], k: int) -> List[float]:
    if k <= 1 or k > len(xs):
        return xs
    out = []
    s = 0.0
    q = []
    for i, v in enumerate(xs):
        q.append(v)
        s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot % de pickers em highway ao longo dos steps (a partir do CSV do vlc_logger).")
    ap.add_argument("--csv", required=True, help="Ficheiro CSV gerado por scripts/vlc_logger.py (steps).")
    ap.add_argument("--out", default="logs/vlc_pct_on_highway.png", help="Ficheiro de saída (PNG).")
    ap.add_argument("--smooth", type=int, default=1, help="Janela de média móvel (ex.: 25). 1 = sem suavização.")
    ap.add_argument("--title", default="Percentagem de Pickers em Highway ao longo dos Steps", help="Título do gráfico.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = load_steps_csv(args.csv)
    if not rows:
        raise SystemExit("CSV vazio / não encontrado.")

    series_by_ep = compute_percent_series(rows)

    # Um único gráfico, várias linhas (uma por episódio)
    plt.figure(figsize=(10, 5))
    for ep, series in sorted(series_by_ep.items()):
        steps = [s for s, _ in series]
        pcts = [p for _, p in series]
        pcts_s = moving_average(pcts, args.smooth)
        plt.plot(steps, pcts_s, label=f"Episódio {ep}")

    plt.xlabel("Step")
    plt.ylabel("% Pickers em Highway")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Gráfico guardado em: {args.out}")


if __name__ == "__main__":
    main()
