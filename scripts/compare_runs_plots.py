# scripts/compare_runs_plots.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CAND_EP = ["episode","episodes","ep","Episode","Episodes"]
DEFAULT_METRICS = ["global_reward","deliveries","collisions","stucks","pickrate","steps"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--central", type=str, help="CSV central/heuristic")
    p.add_argument("--rl", type=str, help="CSV RL/PPO")
    p.add_argument("--csv1", type=str, help="Alias para --central")
    p.add_argument("--csv2", type=str, help="Alias para --rl")
    p.add_argument("--outdir", type=str, default="logs/compare", help="Diretório de saída dos PNGs")
    p.add_argument("--metrics", nargs="*", default=None, help="Métricas a plotar (se vazio: tenta as comuns)")
    p.add_argument("--smooth", type=int, default=0, help="Janela de média móvel (0=sem suavização)")
    return p.parse_args()

def pick_path(primary, alias) -> Path:
    v = primary or alias
    if not v:
        raise SystemExit("Erro: falta CSV. Usa --central/--csv1 e --rl/--csv2")
    p = Path(v)
    if not p.exists():
        raise SystemExit(f"Erro: ficheiro não existe: {p}")
    return p

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ep_col = None
    for c in CAND_EP:
        if c in df.columns:
            ep_col = c
            break
    if ep_col is None:
        df["episode"] = np.arange(1, len(df)+1)
        ep_col = "episode"
    if ep_col != "episode":
        df = df.rename(columns={ep_col: "episode"})
    try:
        df["episode"] = df["episode"].astype(int)
    except Exception:
        pass
    return df

def smooth(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(window=k, min_periods=1).mean() if k and k > 1 else s

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    central_path = pick_path(args.central, args.csv1)
    rl_path = pick_path(args.rl, args.csv2)

    df_c = load_csv(central_path)
    df_r = load_csv(rl_path)

    if args.metrics:
        metrics = args.metrics
    else:
        # tenta métricas default + união de colunas existentes
        metrics = list({*DEFAULT_METRICS, *(set(df_c.columns) | set(df_r.columns))} - {"episode"})
    metrics = [m for m in metrics if m in df_c.columns or m in df_r.columns]
    if not metrics:
        raise SystemExit("Nenhuma métrica encontrada para plotar.")

    for m in metrics:
        plt.figure()
        if m in df_c.columns:
            plt.plot(df_c["episode"], smooth(df_c[m], args.smooth), label=f"Central: {m}")
        if m in df_r.columns:
            plt.plot(df_r["episode"], smooth(df_r[m], args.smooth), label=f"RL: {m}")
        plt.xlabel("Episode")
        plt.ylabel(m)
        plt.title(f"Comparison: {m}")
        plt.legend()
        plt.tight_layout()
        out = outdir / f"{m}.png"
        plt.savefig(out)
        plt.close()
        print(f"[OK] {out}")

    # Bónus: reward combinado se existir
    for key in ["global_reward", "reward", "return"]:
        if key in df_c.columns or key in df_r.columns:
            plt.figure()
            if key in df_c.columns:
                plt.plot(df_c["episode"], smooth(df_c[key], args.smooth), label="Central")
            if key in df_r.columns:
                plt.plot(df_r["episode"], smooth(df_r[key], args.smooth), label="RL")
            plt.xlabel("Episode")
            plt.ylabel(key)
            plt.title("Reward over Episodes")
            plt.legend()
            plt.tight_layout()
            out = outdir / "reward_over_episodes.png"
            plt.savefig(out)
            plt.close()
            print(f"[OK] {out}")
            break

if __name__ == "__main__":
    main()
