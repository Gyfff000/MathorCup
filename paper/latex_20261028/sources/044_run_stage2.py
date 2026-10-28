"""AI-assisted research extension, OpenAI Codex / GPT-6, 2026-10-24."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from src.common import environment, write_json
from src.data import find_data, load_data
from src.stage2 import run_extension

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--out", default="outputs/stage2")
    parser.add_argument("--base", default="outputs/stage1")
    args = parser.parse_args()
    cfg = json.loads(Path("config.json").read_text())
    ext = json.loads(Path("config_stage2.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_extension(load_data(find_data(args.data_dir)), Path(args.base), out, cfg, ext)
    write_json(out / "run_manifest.json", {**environment(), "base_config": cfg, "extension_config": ext})
