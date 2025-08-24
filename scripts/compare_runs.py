# scripts/compare_runs.py
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
CAND_EP = ["episode", "episodes", "ep", "Episode", "Episodes"]
# normalização de nomes (lado esquerdo = nome no CSV; direito = nome unificado)
NORMALIZE_MAP: Dict[str, str] = {
    # reward
    "return": "global_reward",
    "total_reward": "global_reward",
    "global_return": "global_reward",
    # entregas / colisões / stucks
    "shelf_deliveries": "deliveries",
    "clashes": "collisions",
    "stuck": "stucks",
    # pick rate / passos
    "pick_rate": "pickrate",
    "pickRate": "pickrate",
    "num_steps": "steps",
    # tempos / distâncias
    "agv_idle_time": "agvs_idle_time",
    "picker_idle_time": "pickers_idle_time",
    "agv_distance_travelled": "agvs_distance_travelled",
    "picker_distance_travelled": "pickers_distance_travelled",
}

DEFAULT_FOCUS = ["global_reward","deliveries","collisions","stucks","pickrate","steps"]

# ---------- Args ----------
def parse_args():
    p = argparse.ArgumentParser()
    # nomes originais
    p.add_argument("--central", type=str, help="CSV da baseline/heuristic/central")
    p.add_argument("--rl", type=str, help="CSV do RL/PPO/DQN")
    # aliases que usaste
    p.add_argument("--csv1", type=str, help="Alias para --central")
    p.add_argument("--csv2", type=str, help="Alias para --rl")
    p.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS,
                   help="Métricas a comparar (se existirem)")
    # opções dos gráficos
    p.add_argument("--outdir", type=str, default="logs/compare", help="Diretório de saída dos PNGs")
    p.add_argument("--smooth", type=int, default=0, help="Janela média móvel para suavização (0=off)")
    return p.parse_args()

def pick_path(primary: str|None, alias: str|None) -> Path:
    v = primary or alias
    if not v:
        raise SystemExit("Erro: falta caminho para CSV. Usa --central/--csv1 e --rl/--csv2.")
    p = Path(v)
    if not p.exists():
        raise SystemExit(f"Erro: ficheiro não existe: {p}")
    return p

# ---------- Load & normalize ----------
def load_csv(path: Path) -> pd.DataFrame:
    # robust loader: tenta autodetectar separador/encoding/aspas
    import csv
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    # tenta detetar separador com csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0] if text else ",")
        sep_guess = dialect.delimiter
    except Exception:
        # fallback: tenta vírgula; se falhar tentamos outros
        sep_guess = None

    try_order = []
    # 1) autodetectado
    try_order.append({"sep": sep_guess, "engine": "python"})
    # 2) vírgula / ponto e vírgula / tab
    try_order += [
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
    ]

    last_err = None
    for opt in try_order:
        try:
            df = pd.read_csv(
                path,
                sep=opt["sep"],
                engine=opt["engine"],
                encoding="utf-8-sig",
                on_bad_lines="skip",
                dtype_backend="numpy_nullable",  # ignora tipos estranhos
            )
            break
        except Exception as e:
            last_err = e
            df = None

    if df is None:
        raise SystemExit(f"Falha a ler CSV '{path}': {last_err}")

    # normalizar coluna de episódio
    ep_col = None
    for c in CAND_EP:
        if c in df.columns:
            ep_col = c; break
    if ep_col is None:
        df["episode"] = np.arange(1, len(df)+1)
        ep_col = "episode"
    if ep_col != "episode":
        df = df.rename(columns={ep_col: "episode"})

    # normalizar nomes de colunas (mapa)
    ren = {c: NORMALIZE_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=ren)

    # episódio inteiro, se possível
    try:
        df["episode"] = pd.to_numeric(df["episode"], errors="coerce").fillna(method="ffill").fillna(1).astype(int)
    except Exception:
        pass

    return df


# ---------- Summary ----------
def summarize(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    out = []
    for m in metrics:
        if m not in df.columns:
            continue
        s = df[m].dropna()
        if s.empty:
            continue
        out.append({
            "metric": m,
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "last": float(s.iloc[-1]),
        })
    return pd.DataFrame(out)

def print_table(df: pd.DataFrame, title: str):
    if df.empty:
        print(f"--- {title}: (sem métricas encontradas) ---\n"); return
    cols = ["metric","count","mean","median","std","min","max","last"]
    print(f"--- {title} ---")
    print(df[cols].to_string(index=False, justify="center"))
    print()

# ---------- Plots ----------
def smooth_series(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(window=k, min_periods=1).mean() if k and k > 1 else s

def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

def plot_metric(df_c: pd.DataFrame, df_r: pd.DataFrame, metric: str, outdir: Path, smooth:int=0):
    if metric not in df_c.columns and metric not in df_r.columns:
        return
    plt.figure()
    if metric in df_c.columns:
        plt.plot(df_c["episode"], smooth_series(df_c[metric], smooth), label=f"Central: {metric}")
    if metric in df_r.columns:
        plt.plot(df_r["episode"], smooth_series(df_r[metric], smooth), label=f"RL: {metric}")
    plt.xlabel("Episode"); plt.ylabel(metric); plt.title(f"Comparison: {metric}")
    plt.legend(); plt.tight_layout()
    out = outdir / f"{metric}.png"
    plt.savefig(out); plt.close()
    print(f"[OK] {out}")

def auto_plot(df_c: pd.DataFrame, df_r: pd.DataFrame, outdir: Path, smooth:int=0):
    # conjunto de métricas = união + defaults
    metrics = list({*DEFAULT_FOCUS, *(set(df_c.columns)|set(df_r.columns))} - {"episode"})
    for m in sorted(metrics):
        plot_metric(df_c, df_r, m, outdir, smooth)

    # gráfico específico de reward se existir
    for key in ["global_reward","reward","return"]:
        if key in df_c.columns or key in df_r.columns:
            plt.figure()
            if key in df_c.columns:
                plt.plot(df_c["episode"], smooth_series(df_c[key], smooth), label="Central")
            if key in df_r.columns:
                plt.plot(df_r["episode"], smooth_series(df_r[key], smooth), label="RL")
            plt.xlabel("Episode"); plt.ylabel(key); plt.title("Reward over Episodes")
            plt.legend(); plt.tight_layout()
            out = outdir / "reward_over_episodes.png"
            plt.savefig(out); plt.close()
            print(f"[OK] {out}")
            break

# ---------- Main ----------
def main():
    args = parse_args()
    central_path = pick_path(args.central, args.csv1)
    rl_path = pick_path(args.rl, args.csv2)

    df_c = load_csv(central_path)
    df_r = load_csv(rl_path)

    print(f"\n[INFO] Carregado CENTRAL: {central_path}  ({len(df_c)} linhas)")
    print(f"[INFO] Carregado RL     : {rl_path}       ({len(df_r)} linhas)\n")

    # sumários
    sum_c = summarize(df_c, args.focus)
    sum_r = summarize(df_r, args.focus)
    print_table(sum_c, "CENTRAL / HEURISTIC")
    print_table(sum_r, "RL / PPO")

    # comparação lado a lado
    all_metrics = sorted(set(sum_c.get("metric", pd.Series([]))).union(set(sum_r.get("metric", pd.Series([])))))
    rows = []
    for m in all_metrics:
        row = {"metric": m}
        def get(df, col):
            try:
                return float(df.loc[df["metric"]==m, col].values[0])
            except Exception:
                return np.nan
        for col in ["mean","median","std","min","max","last"]:
            row[f"central_{col}"] = get(sum_c, col)
            row[f"rl_{col}"] = get(sum_r, col)
            if col in ("mean","median","last"):
                row[f"delta_{col}"] = row[f"rl_{col}"] - row[f"central_{col}"]
        rows.append(row)
    comp = pd.DataFrame(rows)
    if not comp.empty:
        order = ["metric",
                 "central_mean","rl_mean","delta_mean",
                 "central_median","rl_median","delta_median",
                 "central_last","rl_last","delta_last"]
        print("--- COMPARAÇÃO (RL - CENTRAL) ---")
        print(comp[order].to_string(index=False, justify="center"))
        print()

    # gráficos automáticos
    outdir = Path(args.outdir)
    ensure_outdir(outdir)
    auto_plot(df_c, df_r, outdir, smooth=args.smooth)

if __name__ == "__main__":
    main()
