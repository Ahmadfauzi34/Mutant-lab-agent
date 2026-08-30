"""V2.32 focused sandbox — one SQLite language-neutral cognitive state."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agen_lab as core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="v231_portable_sandbox_") as tmp:
        root = Path(tmp)
        agent = core.IntegratedCognitiveAgent(
            "developer-portable",
            5,
            5,
            epistemic_archive_path=str(root / "cold.sqlite3"),
        )

        agent.register_action("ROBUST_PATCH", "patcher-r1", valid_from=0)
        agent.register_state_equivalence(
            "repo:clean", "clean-v1", aliases=("raw-clean",)
        )
        for outcome in (
            {"task_progress": 0.9, "correctness": 1.0, "execution_cost": 0.25},
            {"task_progress": 0.8, "correctness": 0.95, "execution_cost": 0.20},
        ):
            agent.record_world_model_outcome(
                "raw-clean",
                "ROBUST_PATCH",
                None,
                True,
                objective_outcome=outcome,
            )

        decision = agent.choose_action("raw-clean", ["ROBUST_PATCH"])
        agent.record_objective_experience(
            decision.decision_id,
            objective_outcome={
                "task_progress": 0.95,
                "correctness": 1.0,
                "execution_cost": 0.20,
            },
            success=True,
        )

        portable = root / "agent_state.db"
        metadata = agent.save_portable_state(portable)
        source_hash = sha(portable)

        # Simulates a non-Python implementation reading the stable storage
        # contract: no agen_lab import in this child process.
        reader = f"""
import json, sqlite3
p={str(portable)!r}
with sqlite3.connect(p) as db:
    m=json.loads(db.execute('SELECT manifest_json FROM portable_state_manifest WHERE singleton=1').fetchone()[0])
    types=[r[0] for r in db.execute('SELECT DISTINCT type_id FROM portable_state_nodes WHERE type_id IS NOT NULL ORDER BY type_id')]
    nodes=db.execute('SELECT COUNT(*) FROM portable_state_nodes').fetchone()[0]
    obj=db.execute('SELECT COUNT(*) FROM objective_experiences').fetchone()[0]
print('RESULT='+json.dumps({{'magic':m['magic'],'schema':m['portable_schema_version'],'pickle':m['python_pickle'],'nodes':nodes,'objective_rows':obj,'semantic_types':types[:6]}}))
"""
        proc = subprocess.run(
            [sys.executable, "-c", reader],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout)
        line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT="))
        generic = json.loads(line.split("=", 1)[1])

        restored = core.IntegratedCognitiveAgent.load_portable_state(portable)
        replay = restored.replay_objective_experience_utility(
            "raw-clean", "ROBUST_PATCH"
        )
        learning_before = restored.learning_state()
        restored.record_world_model_outcome(
            "raw-clean",
            "ROBUST_PATCH",
            None,
            True,
            objective_outcome={"task_progress": 1.0, "correctness": 1.0},
        )
        source_unchanged = sha(portable) == source_hash

        checks = {
            "core_is_v2_32": core.CORE_VERSION == "2.42",
            "portable_file_is_sqlite": portable.read_bytes()[:16] == b"SQLite format 3\x00",
            "manifest_is_language_neutral": metadata["language_neutral"] is True,
            "manifest_says_no_pickle": metadata["python_pickle"] is False,
            "generic_reader_needs_no_agen_lab": generic["magic"] == core.PORTABLE_STATE_MAGIC,
            "generic_reader_sees_schema_1": generic["schema"] == 1,
            "semantic_type_ids_are_agen_names": all(t.startswith("agen/") for t in generic["semantic_types"]),
            "cold_objective_rows_share_same_db": generic["objective_rows"] == 3,
            "restored_state_alias_is_exact": restored.resolve_state_identity("raw-clean").canonical_id == "repo:clean",
            "restored_replay_matches_joint_history": replay["joint_agreement"]["exact_within_tolerance"],
            "restored_agent_can_continue_learning": restored.epistemic_archive.objective_experience_count() == 4,
            "source_db_is_immutable_after_runtime_learning": source_unchanged,
        }

        result = {
            "portable_manifest": metadata,
            "generic_sqlite_reader": generic,
            "replay": replay,
            "checks": checks,
        }
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        failed = [name for name, ok in checks.items() if not ok]
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
