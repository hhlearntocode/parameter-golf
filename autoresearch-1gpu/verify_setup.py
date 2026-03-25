#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH = ROOT / "autoresearch-1gpu"


def check_path(path: Path, label: str) -> tuple[bool, str]:
    ok = path.exists()
    status = "OK" if ok else "MISSING"
    return ok, f"[{status}] {label}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"


def main() -> int:
    scaffold_checks = [
        (ROOT / "train_gpt.py", "1-GPU research target"),
        (AUTORESEARCH / "README.md", "autoresearch hub"),
        (AUTORESEARCH / "program.md", "agent program"),
        (AUTORESEARCH / "results.tsv", "aggregate results table"),
        (AUTORESEARCH / "runs" / "TEMPLATE_RUN" / "manifest.json", "run manifest template"),
        (AUTORESEARCH / "runs" / "TEMPLATE_RUN" / "metrics.json", "run metrics template"),
        (AUTORESEARCH / "runs" / "TEMPLATE_RUN" / "decision.txt", "run decision template"),
        (AUTORESEARCH / "runs" / "TEMPLATE_RUN" / "patch.diff", "run patch template"),
        (AUTORESEARCH / "runs" / "TEMPLATE_RUN" / "stdout.log", "run log template"),
        (AUTORESEARCH / "submission-template" / "README.md", "submission README template"),
        (AUTORESEARCH / "submission-template" / "submission.json", "submission metadata template"),
        (AUTORESEARCH / "submission-template" / "CHECKLIST.md", "submission checklist"),
        (
            AUTORESEARCH / "export-template" / "records" / "track_non_record_16mb" / "TEMPLATE_RUN" / "README.md",
            "clean export README template",
        ),
        (
            AUTORESEARCH / "export-template" / "records" / "track_non_record_16mb" / "TEMPLATE_RUN" / "submission.json",
            "clean export metadata template",
        ),
        (
            AUTORESEARCH / "export-template" / "records" / "track_non_record_16mb" / "TEMPLATE_RUN" / "train_gpt.py",
            "clean export code placeholder",
        ),
        (
            AUTORESEARCH / "export-template" / "records" / "track_non_record_16mb" / "TEMPLATE_RUN" / "train.log",
            "clean export log placeholder",
        ),
    ]

    print("== Scaffold Checks ==")
    scaffold_ok = True
    for path, label in scaffold_checks:
        ok, line = check_path(path, label)
        print(line)
        scaffold_ok &= ok

    results_path = AUTORESEARCH / "results.tsv"
    if results_path.exists():
        header = results_path.read_text(encoding="utf-8").splitlines()[:1]
        expected = "run_id\tbase_commit\thead_commit\ttarget_file\tval_bpb\tartifact_bytes\tstatus\tdescription"
        header_ok = bool(header) and header[0] == expected
        print(f"[{'OK' if header_ok else 'BAD'}] results.tsv header")
        scaffold_ok &= header_ok

    print("\n== CUDA Checks ==")
    nvidia_smi = shutil.which("nvidia-smi")
    cuda_ok = False
    if nvidia_smi is None:
        print("[MISSING] nvidia-smi not found")
    else:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "unknown error"
            print(f"[BAD] nvidia-smi query failed: {stderr}")
        else:
            gpu_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            cuda_ok = len(gpu_lines) >= 1
            print(f"[{'OK' if cuda_ok else 'BAD'}] visible NVIDIA GPUs: {len(gpu_lines)}")
            for line in gpu_lines:
                print(f"  - {line}")
            if not cuda_ok:
                print("[INFO] autoresearch-1gpu expects at least 1 visible GPU")

    print("\n== Data Checks ==")
    tokenizer = ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    dataset_dir = ROOT / "data" / "datasets" / "fineweb10B_sp1024"
    train_shards = sorted(dataset_dir.glob("fineweb_train_*.bin")) if dataset_dir.exists() else []
    val_shards = sorted(dataset_dir.glob("fineweb_val_*.bin")) if dataset_dir.exists() else []

    tok_ok, tok_line = check_path(tokenizer, "tokenizer model")
    print(tok_line)
    ds_ok, ds_line = check_path(dataset_dir, "dataset directory")
    print(ds_line)
    train_ok = len(train_shards) > 0
    val_ok = len(val_shards) > 0
    print(f"[{'OK' if train_ok else 'MISSING'}] train shards: {len(train_shards)} found")
    print(f"[{'OK' if val_ok else 'MISSING'}] val shards: {len(val_shards)} found")
    data_ok = tok_ok and ds_ok and train_ok and val_ok

    print("\n== Summary ==")
    print(f"scaffold_ready={'yes' if scaffold_ok else 'no'}")
    print(f"cuda_ready={'yes' if cuda_ok else 'no'}")
    print(f"data_ready={'yes' if data_ok else 'no'}")
    if not data_ok:
        print("next_action=run `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1` for a smoke setup")
    if not cuda_ok:
        print("next_action=use a machine with 1 visible NVIDIA GPU before starting this 1-GPU research loop")
    return 0 if scaffold_ok and cuda_ok and data_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
