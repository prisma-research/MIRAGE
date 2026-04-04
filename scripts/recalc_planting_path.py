"""
Recalculate planting_path for existing trial JSONs where it was set to 'none'
due to the observe_planting_path filename-matching bug.

The original checker only looked for artifact_id in file *content*, not filename.
Agents often name the file `memory/ss_<artifact_id>.md` without including the
artifact_id string in the body. This script re-derives planting_path from
signals available in the saved JSON:

  1. startup_exposure_log (S3): exec/read entries may show artifact_id in path
  2. tool_trace: any call/result mentioning artifact_id in a file path
  3. active_context_view: artifact_id may appear in planting-turn context

If artifact_id is found in any of these, reclassifies none → pre_compaction_flush.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"


def _contains_artifact(obj, artifact_id: str) -> bool:
    """Recursively check if artifact_id appears in any string within obj."""
    if isinstance(obj, str):
        return artifact_id in obj
    if isinstance(obj, dict):
        return any(_contains_artifact(v, artifact_id) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_artifact(item, artifact_id) for item in obj)
    return False


def recalc_trial(path: Path) -> tuple[str, str] | None:
    """Returns (old, new) planting_path if changed, else None."""
    with path.open() as f:
        trial = json.load(f)

    old = trial.get("planting_path")
    if old != "none":
        return None  # Already correctly classified

    artifact_id = trial.get("artifact_id", "")
    if not artifact_id:
        return None

    # Check all available evidence
    evidence_fields = [
        trial.get("startup_exposure_log", []),
        trial.get("tool_trace", []),
        trial.get("active_context_view", []),
    ]

    for field in evidence_fields:
        if _contains_artifact(field, artifact_id):
            trial["planting_path"] = "pre_compaction_flush"
            trial["_planting_path_recalculated"] = True
            with path.open("w") as f:
                json.dump(trial, f, indent=2)
            return (old, "pre_compaction_flush")

    return None


def main():
    dirs = sorted(LOGS_DIR.glob("trials*"))
    if len(sys.argv) > 1:
        dirs = [LOGS_DIR / d for d in sys.argv[1:]]

    total_changed = 0
    for trial_dir in dirs:
        if not trial_dir.is_dir():
            continue
        changed = []
        for json_file in sorted(trial_dir.glob("*.json")):
            result = recalc_trial(json_file)
            if result:
                changed.append((json_file.name, result))
        if changed:
            print(f"\n{trial_dir.name}: {len(changed)} reclassified")
            for name, (old, new) in changed:
                print(f"  {name}: {old} → {new}")
        else:
            print(f"{trial_dir.name}: no changes")
        total_changed += len(changed)

    print(f"\nTotal reclassified: {total_changed}")


if __name__ == "__main__":
    main()
