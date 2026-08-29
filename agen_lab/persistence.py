"""Trusted-local checkpoint/restart — physically extracted in M4.

The V2.28 checkpoint container and serialization format are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import uuid
import zipfile

from pathlib import Path
from typing import Dict

PERSISTENCE_SCHEMA_VERSION = 1
PERSISTENCE_MAGIC = "AGEN_KOGNITIF_FULL_CHECKPOINT"


def _core_version() -> str:
    module = sys.modules.get("agen_kognitif_v2_28")
    if module is None:
        raise RuntimeError(
            "Canonical module agen_kognitif_v2_28 belum tersedia"
        )
    return module.CORE_VERSION


class AgentPersistenceError(RuntimeError):
    pass


class AgentPersistenceManager:
    """
    Durable FULL checkpoint for trusted local files.

    Container:
        ZIP
        ├── metadata.json
        └── agent.pkl

    Why pickle here:
    - V2.18 needs exact restart semantics for heterogeneous Python records,
      temporal histories, tuple-keyed Q/world-model maps, and legacy planner
      state without prematurely inventing a lossy interchange schema.
    - The payload is protected by SHA-256 and strict schema/core-version
      checks.

    SECURITY:
    Never load an untrusted checkpoint. Python pickle is intentionally used
    only for local trusted persistence. A portable safe interchange format is
    separate future work.
    """

    @staticmethod
    def save(
        agent,
        path,
    ) -> Dict:
        target = Path(path)
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            payload = pickle.dumps(
                agent,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception as exc:
            raise AgentPersistenceError(
                "Agent tidak dapat diserialisasi. "
                "Periksa callback/semantic validator eksternal "
                "yang mungkin tidak picklable."
            ) from exc

        payload_sha256 = hashlib.sha256(
            payload
        ).hexdigest()

        archive_bytes = (
            agent.epistemic_archive
            .snapshot_bytes()
        )
        archive_sha256 = hashlib.sha256(
            archive_bytes
        ).hexdigest()

        lifecycle = (
            agent.memory_lifecycle_state()
            if hasattr(
                agent,
                "memory_lifecycle_state",
            )
            else None
        )

        metadata = {
            "magic": PERSISTENCE_MAGIC,
            "schema_version": (
                PERSISTENCE_SCHEMA_VERSION
            ),
            "core_version": _core_version(),
            "python_version": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "domain_name": agent.domain.name,
            "belief_context_id": (
                agent.belief_contexts.current_id
            ),
            "belief_time": (
                agent.belief_contexts.now
            ),
            "interaction_clock": (
                agent.interaction_clock
            ),
            "payload_sha256": payload_sha256,
            "payload_bytes": len(payload),
            "epistemic_archive_sha256":
                archive_sha256,
            "epistemic_archive_bytes":
                len(archive_bytes),
            "epistemic_archive_state":
                agent.epistemic_archive_state(),
            "decision_counter": (
                agent._decision_counter
            ),
            "objective_profile": (
                agent.objective_profile_state()
                if hasattr(
                    agent,
                    "objective_profile_state",
                )
                else None
            ),
            "objective_profile_instance_id": (
                agent.objective_profile.instance_id
                if hasattr(
                    agent,
                    "objective_profile",
                )
                and hasattr(
                    agent.objective_profile,
                    "instance_id",
                )
                else None
            ),
            "prediction_counter": (
                agent._outcome_prediction_counter
            ),
            "memory_lifecycle": lifecycle,
            "format_note": (
                "trusted-local exact Python checkpoint"
            ),
        }

        temp = target.with_name(
            target.name + ".tmp"
        )
        if temp.exists():
            temp.unlink()

        try:
            with zipfile.ZipFile(
                temp,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr(
                    "metadata.json",
                    json.dumps(
                        metadata,
                        indent=2,
                        sort_keys=True,
                    ),
                )
                archive.writestr(
                    "agent.pkl",
                    payload,
                )
                archive.writestr(
                    "epistemic_archive.sqlite3",
                    archive_bytes,
                )

            os.replace(
                temp,
                target,
            )
        except Exception:
            if temp.exists():
                temp.unlink()
            raise

        return metadata

    @staticmethod
    def inspect(path) -> Dict:
        target = Path(path)

        try:
            with zipfile.ZipFile(
                target,
                "r",
            ) as archive:
                raw = archive.read(
                    "metadata.json"
                )
        except Exception as exc:
            raise AgentPersistenceError(
                "Checkpoint container tidak valid"
            ) from exc

        try:
            return json.loads(
                raw.decode("utf-8")
            )
        except Exception as exc:
            raise AgentPersistenceError(
                "metadata.json checkpoint tidak valid"
            ) from exc

    @staticmethod
    def load(path):
        target = Path(path)

        try:
            with zipfile.ZipFile(
                target,
                "r",
            ) as archive:
                metadata = json.loads(
                    archive.read(
                        "metadata.json"
                    ).decode("utf-8")
                )
                payload = archive.read(
                    "agent.pkl"
                )
                archive_bytes = archive.read(
                    "epistemic_archive.sqlite3"
                )
        except Exception as exc:
            raise AgentPersistenceError(
                "Checkpoint tidak dapat dibaca"
            ) from exc

        if metadata.get(
            "magic"
        ) != PERSISTENCE_MAGIC:
            raise AgentPersistenceError(
                "Checkpoint magic tidak cocok"
            )

        if metadata.get(
            "schema_version"
        ) != PERSISTENCE_SCHEMA_VERSION:
            raise AgentPersistenceError(
                "Persistence schema tidak didukung: "
                f"{metadata.get('schema_version')}"
            )

        checkpoint_core_version = metadata.get("core_version")
        current_core_version = _core_version()
        compatible_versions = {current_core_version}
        if current_core_version == "2.29":
            compatible_versions.add("2.28")
        elif current_core_version == "2.30":
            # Explicit trusted-local compatibility with the two frozen
            # baselines that can appear in inherited regression/checkpoints.
            # This is still not a general pickle portability claim.
            compatible_versions.update({
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.31":
            # Legacy pickle checkpoints remain an explicit trusted-local
            # compatibility path. V2.31 portable state itself is pickle-free.
            compatible_versions.update({
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.32":
            # Explicit trusted-local compatibility with frozen predecessors.
            # Missing V2.32 pattern state is backfilled EMPTY; no historical
            # pattern observations are fabricated.
            compatible_versions.update({
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.33":
            # Frozen predecessor compatibility. V2.33 spatial scene history is
            # backfilled EMPTY; planner/grid state is never reinterpreted as
            # object-centric spatial observations.
            compatible_versions.update({
                "2.32",
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.34":
            # V2.34 adds stateless transformation algebra. Frozen predecessor
            # spatial/pattern history can load unchanged; no transform
            # experience/history is fabricated.
            compatible_versions.update({
                "2.33",
                "2.32",
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.35":
            # V2.35 adds stateless counterfactual manipulation operators.
            # Frozen durable state loads unchanged; no simulated manipulation
            # is fabricated as experience or history.
            compatible_versions.update({
                "2.34",
                "2.33",
                "2.32",
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.36":
            # V2.36 adds stateless bounded manipulation planning. Frozen
            # durable state loads unchanged; no counterfactual plan history or
            # execution experience is fabricated.
            compatible_versions.update({
                "2.35",
                "2.34",
                "2.33",
                "2.32",
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.37":
            # V2.37 adds a durable bounded execution-ticket/actual-observation
            # journal. Frozen predecessor states receive an EMPTY store; no
            # dispatch or feedback history is fabricated.
            compatible_versions.update({
                "2.36",
                "2.35",
                "2.34",
                "2.33",
                "2.32",
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.38":
            # V2.38 adds a bounded durable replan-attempt journal. Frozen
            # predecessor states receive an EMPTY store; no historical
            # deviation-triggered replans are fabricated.
            compatible_versions.update({
                "2.37",
                "2.36",
                "2.35",
                "2.34",
                "2.33",
                "2.32",
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.39":
            # V2.39 adds a bounded durable recovery-decision journal. Frozen
            # predecessor states receive EMPTY recovery history; no old
            # feedback/replans are silently reclassified as policy decisions.
            compatible_versions.update({
                "2.38",
                "2.37",
                "2.36",
                "2.35",
                "2.34",
                "2.33",
                "2.32",
                "2.31",
                "2.30",
                "2.29",
                "2.28",
            })
        elif current_core_version == "2.40":
            # V2.40 adds bounded empirical manipulation-reliability state.
            # Predecessors receive EMPTY reliability history; old CLOSED
            # feedback is never retroactively counted.
            compatible_versions.update({
                "2.39", "2.38", "2.37", "2.36", "2.35", "2.34",
                "2.33", "2.32", "2.31", "2.30", "2.29", "2.28",
            })
        elif current_core_version == "2.41":
            # V2.41 adds stateless read-only reliability-aware ranking over
            # equal-depth V2.36 candidates. No ranking history is fabricated
            # and no reliability sample is replayed on predecessor migration.
            compatible_versions.update({
                "2.40", "2.39", "2.38", "2.37", "2.36", "2.35",
                "2.34", "2.33", "2.32", "2.31", "2.30", "2.29", "2.28",
            })
        elif current_core_version == "2.42":
            # V2.42 adds only a stateless/read-only ranked view over completed
            # V2.38 replan records using the existing V2.41 ranker and V2.40
            # reliability store. No ranking journal or synthetic samples exist.
            compatible_versions.update({
                "2.41", "2.40", "2.39", "2.38", "2.37", "2.36",
                "2.35", "2.34", "2.33", "2.32", "2.31", "2.30",
                "2.29", "2.28",
            })

        if checkpoint_core_version not in compatible_versions:
            raise AgentPersistenceError(
                "Core version checkpoint tidak cocok: "
                f"{checkpoint_core_version} not in {sorted(compatible_versions)}"
            )

        digest = hashlib.sha256(
            payload
        ).hexdigest()

        if digest != metadata.get(
            "payload_sha256"
        ):
            raise AgentPersistenceError(
                "Checkpoint payload gagal SHA-256 integrity check"
            )

        if len(payload) != metadata.get(
            "payload_bytes"
        ):
            raise AgentPersistenceError(
                "Checkpoint payload length tidak cocok"
            )

        archive_digest = hashlib.sha256(
            archive_bytes
        ).hexdigest()
        if archive_digest != metadata.get(
            "epistemic_archive_sha256"
        ):
            raise AgentPersistenceError(
                "Cold epistemic archive gagal SHA-256 integrity check"
            )

        if len(
            archive_bytes
        ) != metadata.get(
            "epistemic_archive_bytes"
        ):
            raise AgentPersistenceError(
                "Cold epistemic archive length tidak cocok"
            )

        try:
            agent = pickle.loads(
                payload
            )
        except Exception as exc:
            raise AgentPersistenceError(
                "Checkpoint payload gagal direstore"
            ) from exc

        if not hasattr(
            agent,
            "_repair_runtime_links",
        ):
            raise AgentPersistenceError(
                "Payload bukan cognitive-agent checkpoint yang kompatibel"
            )

        sidecar = target.with_name(
            target.name
            + "."
            + uuid.uuid4().hex
            + ".cold.sqlite3"
        )
        temp_sidecar = sidecar.with_name(
            sidecar.name + ".tmp"
        )
        temp_sidecar.write_bytes(
            archive_bytes
        )
        os.replace(
            temp_sidecar,
            sidecar,
        )

        agent.epistemic_archive.rebind(
            sidecar,
            owned_path=True,
        )
        agent._repair_runtime_links()
        return agent, metadata


# Trusted-local checkpoint compatibility.
_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"

for _cls in (
    AgentPersistenceError,
    AgentPersistenceManager,
):
    _cls.__module__ = _CANONICAL_PICKLE_MODULE

del _cls

__all__ = [
    "PERSISTENCE_SCHEMA_VERSION",
    "PERSISTENCE_MAGIC",
    "AgentPersistenceError",
    "AgentPersistenceManager",
]

