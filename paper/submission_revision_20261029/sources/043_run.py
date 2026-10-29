"""AI-assisted orchestration: OpenAI Codex / GPT-6, 2026-10-24."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from src.common import environment, progress, write_json
from src.data import find_data, load_data


def main():
    parser = argparse.ArgumentParser(description="Reproducible A-track research baselines")
    parser.add_argument("stage", choices=["audit", "q1", "q2", "q3", "report", "all"])
    parser.add_argument("--data-dir")
    parser.add_argument("--out", default="outputs/stage1")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(find_data(args.data_dir))
    if args.stage in ["all", "audit"]:
        from src.audit import run_audit
        run_audit(data, out, cfg)
    if args.stage in ["all", "q1"]:
        from src.q1 import run_q1
        run_q1(data, out, cfg)
    if args.stage in ["all", "q2"]:
        from src.q2 import run_q2
        run_q2(data, out, cfg)
    if args.stage in ["all", "q3"]:
        from src.q3 import run_q3
        run_q3(data, out, cfg)
    if args.stage in ["all", "report"]:
        from src.reporting import make_report
        make_report(data, out, cfg)
    manifest = environment()
    manifest["stage"] = args.stage
    manifest["config"] = cfg
    manifest["files"] = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file() and "private" not in p.parts and p.suffix in {".csv", ".json", ".md", ".png"})
    write_json(out / "run_manifest.json", manifest)
    progress(f"Completed {args.stage}; outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
