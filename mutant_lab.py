#!/usr/bin/env python3
"""Minimal Mutant Lab control tool for the promoted Seed G0 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED_PATH = ROOT / "RUNTIME_SEED.json"
INCUBATOR = ROOT / "lab" / "incubator"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed() -> dict:
    try:
        data = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot read RUNTIME_SEED.json: {exc}") from exc
    if data.get("status") != "FROZEN_PROMOTED":
        raise SystemExit("seed runtime is not FROZEN_PROMOTED")
    return data


def _baseline_paths(seed: dict) -> list[Path]:
    spec = seed.get("canonical_sandbox") or {}
    sandbox_root = ROOT / str(spec.get("layout", "sandboxes/"))
    runner = ROOT / str(spec.get("runner", "run_sandboxes.py"))
    game_runner = ROOT / str(spec.get("game_runner", "run_game_e2e.py"))
    if not sandbox_root.is_dir() or not runner.is_file() or not game_runner.is_file():
        raise SystemExit("canonical baseline sandbox is incomplete")
    files = [runner, game_runner]
    files.extend(p for p in sandbox_root.rglob("*") if p.is_file())
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())


def baseline_fingerprint(seed: dict) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = _baseline_paths(seed)
    for path in files:
        if path.is_symlink():
            raise SystemExit(f"canonical baseline must not contain symlink: {path}")
        rel = path.relative_to(ROOT).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest(), len(files)


def assert_baseline_pristine(seed: dict) -> tuple[str, int]:
    spec = seed.get("canonical_sandbox") or {}
    expected = spec.get("content_fingerprint_sha256")
    expected_count = spec.get("file_count")
    if not expected:
        raise SystemExit("seed manifest has no canonical baseline content fingerprint")
    actual, count = baseline_fingerprint(seed)
    if expected_count is not None and int(expected_count) != count:
        raise SystemExit(
            f"canonical baseline file count changed: expected {expected_count}, got {count}"
        )
    if actual != expected:
        raise SystemExit(
            "canonical baseline content changed; refusing to use a non-pristine Seed G0 baseline\n"
            f"expected: {expected}\nactual:   {actual}"
        )
    return actual, count


def _mutant_dir(mutant_id: str) -> Path:
    if not SAFE_ID.fullmatch(mutant_id):
        raise SystemExit(
            "mutant id must be 1-96 chars using letters, numbers, '.', '_' or '-'"
        )
    return INCUBATOR / mutant_id


def cmd_spawn(args: argparse.Namespace) -> int:
    seed = _seed()
    fingerprint, file_count = assert_baseline_pristine(seed)
    target = _mutant_dir(args.mutant_id)
    if target.exists():
        raise SystemExit(f"mutant already exists: {target.relative_to(ROOT)}")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("reference width/height must be positive")

    target.mkdir(parents=True, exist_ok=False)
    brain = target / "brain.db"
    manifest_path = target / "MUTANT_MANIFEST.json"
    try:
        from agen_lab import IntegratedCognitiveAgent

        agent = IntegratedCognitiveAgent(
            args.domain,
            args.width,
            args.height,
        )
        portable = agent.save_portable_state(brain)
        archive = getattr(agent, "epistemic_archive", None)
        close = getattr(archive, "close", None)
        if callable(close):
            close()

        brain_sha = _sha256(brain)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest = {
            "mutant_id": args.mutant_id,
            "status": "CREATED",
            "mutation_class": args.mutation_class,
            "purpose": args.purpose,
            "parent_seed": seed["seed_id"],
            "parent_runtime_lineage": seed["runtime_lineage"],
            "parent_semantic_version": seed["semantic_version"],
            "parent_checkpoint_sha256": seed["checkpoint_sha256"],
            "canonical_baseline_fingerprint_sha256": fingerprint,
            "canonical_baseline_file_count": file_count,
            "domain": args.domain,
            "reference_space": {"width": args.width, "height": args.height},
            "brain_file": "brain.db",
            "brain_sha256": brain_sha,
            "portable_schema_version": portable["portable_schema_version"],
            "portable_core_version": portable["core_version"],
            "created_at": now,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    seed = _seed()
    current_fingerprint, current_file_count = assert_baseline_pristine(seed)
    target = _mutant_dir(args.mutant_id)
    manifest_path = target / "MUTANT_MANIFEST.json"
    brain = target / "brain.db"
    if not manifest_path.is_file() or not brain.is_file():
        raise SystemExit(f"incomplete mutant workspace: {target.relative_to(ROOT)}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mutant_id") != args.mutant_id:
        raise SystemExit("mutant manifest id does not match workspace id")
    if manifest.get("parent_seed") != seed.get("seed_id"):
        raise SystemExit("mutant parent seed does not match current promoted seed")
    if manifest.get("parent_checkpoint_sha256") != seed.get("checkpoint_sha256"):
        raise SystemExit("mutant parent checkpoint does not match current promoted seed")
    if manifest.get("canonical_baseline_fingerprint_sha256") != current_fingerprint:
        raise SystemExit("mutant was not spawned from the current canonical baseline")
    if int(manifest.get("canonical_baseline_file_count", -1)) != current_file_count:
        raise SystemExit("mutant canonical baseline file count does not match current seed")
    actual_sha = _sha256(brain)
    if actual_sha != manifest.get("brain_sha256"):
        raise SystemExit(
            f"brain hash mismatch: expected {manifest.get('brain_sha256')}, got {actual_sha}"
        )

    from agen_lab import IntegratedCognitiveAgent

    portable = IntegratedCognitiveAgent.inspect_portable_state(brain)
    if portable.get("sqlite_integrity") != "ok":
        raise SystemExit("portable brain SQLite integrity is not ok")
    if str(portable.get("core_version")) != str(manifest.get("parent_semantic_version")):
        raise SystemExit("portable brain core version does not match parent seed")

    result = {
        "mutant": manifest,
        "brain": {
            "sha256": actual_sha,
            "sqlite_integrity": portable["sqlite_integrity"],
            "portable_schema_version": portable["portable_schema_version"],
            "core_version": portable["core_version"],
            "domain_name": portable["domain_name"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_baseline(_args: argparse.Namespace) -> int:
    seed = _seed()
    fingerprint, file_count = assert_baseline_pristine(seed)
    print(f"BASELINE CONTENT: {file_count}/{file_count} PRISTINE {fingerprint}", flush=True)

    with tempfile.TemporaryDirectory(prefix="mutant-lab-baseline-") as tmp:
        work = Path(tmp)
        shutil.copytree(ROOT / "agen_lab", work / "agen_lab")
        shutil.copytree(ROOT / "core", work / "core")
        shutil.copytree(ROOT / "sandboxes", work / "sandboxes")
        shutil.copy2(ROOT / "run_sandboxes.py", work / "run_sandboxes.py")
        shutil.copy2(ROOT / "run_game_e2e.py", work / "run_game_e2e.py")
        for runner in ("run_sandboxes.py", "run_game_e2e.py"):
            print(f"\n=== {runner} (isolated) ===", flush=True)
            proc = subprocess.run([sys.executable, runner], cwd=work)
            if proc.returncode:
                return proc.returncode
    print("\nCANONICAL BASELINE PASS; repository workspace left clean", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal local control tool for Mutant Lab Seed G0."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    spawn = sub.add_parser("spawn", help="create an isolated local mutant brain")
    spawn.add_argument("mutant_id")
    spawn.add_argument("--purpose", required=True)
    spawn.add_argument(
        "--mutation-class",
        choices=("state_mutation", "knowledge_mutation"),
        default="state_mutation",
    )
    spawn.add_argument("--domain", default="mutant-lab")
    spawn.add_argument("--width", type=int, default=16, help="reference width, not a runtime limit")
    spawn.add_argument("--height", type=int, default=16, help="reference height, not a runtime limit")
    spawn.set_defaults(func=cmd_spawn)

    inspect = sub.add_parser("inspect", help="verify one local mutant brain + manifest")
    inspect.add_argument("mutant_id")
    inspect.set_defaults(func=cmd_inspect)

    baseline = sub.add_parser(
        "baseline",
        help="verify frozen baseline fingerprint and run canonical sandboxes in isolation",
    )
    baseline.set_defaults(func=cmd_baseline)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
