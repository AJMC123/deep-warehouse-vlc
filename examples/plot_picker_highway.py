# scripts/plot_picker_highway.py
import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

REQ_COLS = {"episode", "step", "pickers_on_highway", "pickers_off_highway"}


def load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"CSV vazio: {path}")
    missing = [c for c in REQ_COLS if c not in reader.fieldnames]
    if missing:
        raise SystemExit(
            f"O ficheiro '{path}' não tem as colunas necessárias {sorted(REQ_COLS)}.\n"
            f"Este gráfico usa o CSV gerado por 'scripts/vlc_logger.py' (ex.: logs/vlc_usage_steps.csv).\n"
            f"Faltam: {missing}"
        )
    return rows


def _to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def series_pct_on_highway(rows: List[dict]) -> Dict[int, List[Tuple[int, float]]]:
    """
    Devolve {episode: [(step, pct_on_highway), ...]} com:
    pct = 100 * pickers_on_highway / (pickers_on_highway + pickers_off_highway)
    """
    by_ep: Dict[int, List[Tuple[int, float]]] = {}
    for r in rows:
        ep = _to_int(r.get("episode", 1), 1)
        step = _to_int(r.get("step", 0), 0)
        on_hw = _to_int(r.get("pickers_on_highway", 0), 0)
        off_hw = _to_int(r.get("pickers_off_highway", 0), 0)
        denom = on_hw + off_hw
        pct = (100.0 * on_hw / denom) if denom > 0 else 0.0
        by_ep.setdefault(ep, []).append((step, pct))
    for ep in by_ep:
        by_ep[ep].sort(key=lambda t: t[0])
    return by_ep


def moving_average(xs: List[float], k: int) -> List[float]:
    if k <= 1:
        return xs
    out, acc = [], 0.0
    q: List[float] = []
    for v in xs:
        q.append(v); acc += v
        if len(q) > k:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def main():
    ap = argparse.ArgumentParser(description="Plota % de pickers em highway ao longo dos steps (por episódio).")
    ap.add_argument("--csv", required=True, help="CSV de passos do vlc_logger (ex.: logs/vlc_usage_steps.csv).")
    ap.add_argument("--output", required=True, help="Ficheiro PNG de saída (ex.: plots/picker_highway.png).")
    ap.add_argument("--smooth", type=int, default=1, help="Janela de média móvel (1 = sem suavização).")
    ap.add_argument("--title", default="Percentagem de Pickers em Highway ao longo dos Steps")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows = load_rows(args.csv)
    data = series_pct_on_highway(rows)

    plt.figure(figsize=(10, 5))
    for ep, series in sorted(data.items()):
        steps = [s for s, _ in series]
        pcts = [p for _, p in series]
        pcts = moving_average(pcts, args.smooth)
        plt.plot(steps, pcts, label=f"Episódio {ep}")

    plt.xlabel("Step")
    plt.ylabel("% Pickers em Highway")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"[OK] Gráfico salvo em: {args.output}")


if __name__ == "__main__":
    main()
