#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path

MOVE_TO_EXAMPLES = [
    "scripts/capture_figure_4_2.py",
    "scripts/capture_figure_4_3_vlcgrid.py",
    "scripts/capture_vlc_frames.py",
    "scripts/capture_vlc_rgmb_letters.py",
    "scripts/render_vlc_layout.py",
    "scripts/plot_picker_highway.py",
    "scripts/vlc_plot.py",
    "tarware/quick_render.py",
]

REMOVE_PATHS = [".venv","logs","plots_thesis","scripts/OLD",".git"]

def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def move_file(src: Path, dst: Path, dry: bool):
    if src.exists():
        ensure_dir(dst.parent)
        print(f"[MOVE] {src} -> {dst}")
        if not dry: shutil.move(str(src), str(dst))

def remove_path(p: Path, dry: bool):
    if p.exists():
        print(f"[REMOVE] {p}")
        if not dry:
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(".").resolve()
    examples = root/"examples"; ensure_dir(examples)
    for rel in MOVE_TO_EXAMPLES:
        move_file(root/rel, examples/Path(rel).name, args.dry_run)
    for rel in REMOVE_PATHS:
        remove_path(root/rel, args.dry_run)
    print("[DONE] Limpeza concluída", "(dry-run)" if args.dry_run else "")

if __name__ == "__main__":
    main()
