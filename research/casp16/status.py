"""Compact live status, including resumed attempts and process liveness."""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs/issue-settings-csv-seed0"
rows = []
for target in ["L1000", "L3000"]:
    run = ROOT / target
    progress_path = run / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    logs = list(run.glob("stdout*.log"))
    latest = max(logs, key=lambda path: path.stat().st_mtime) if logs else None
    log = latest.read_text(errors="replace") if latest else ""
    lines = [line for line in log.splitlines() if "Predicting DataLoader" in line]
    errors = [line for line in log.splitlines() if "Traceback" in line or "Skipping." in line or "OutOfMemory" in line]
    pid = progress.get("pid")
    try:
        os.kill(pid, 0)
        alive = True
    except (OSError, TypeError):
        alive = False
    result_path = run / "result.json"
    rows.append({"target": target,
                 "structure_pairs": len({path.parent for path in (run / "output").rglob("confidence_*.json")}),
                 "affinity_pairs": len(list((run / "output").rglob("affinity_*.json"))),
                 "expected": 17 if target == "L1000" else 123,
                 "alive": alive, "pid": pid, "log": latest.name if latest else None,
                 "last_progress": lines[-1] if lines else None, "errors": errors[-2:],
                 "result": json.loads(result_path.read_text()) if result_path.exists() else None})
print(json.dumps(rows, ensure_ascii=False))
