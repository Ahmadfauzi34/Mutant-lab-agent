
"""V2.33 focused sandbox — object-centric 2D spatial reasoning."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agen_lab as core


def obj(object_id, x, y, w=1.0, h=1.0, labels=()):
    return core.SpatialObject2D(
        object_id=object_id,
        pose=core.SpatialPose2D(x, y),
        extent=core.SpatialExtent2D(w, h),
        labels=tuple(labels),
    )


def types(relations):
    return sorted(
        relation.relation_type.value
        for relation in relations
    )


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def main():
    with tempfile.TemporaryDirectory(
        prefix="v233_spatial_lab_"
    ) as temp:
        root = Path(temp)
        agent = core.IntegratedCognitiveAgent(
            "spatial-laboratory",
            12,
            12,
            epistemic_archive_path=str(
                root / "cold.sqlite3"
            ),
        )

        empirical_before = {
            "evidence": len(agent.all_evidence()),
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        scene = agent.register_spatial_scene(
            (
                obj("BALL", 0, 0, 0, 0, ("movable",)),
                obj("BOX", 0, 0, 4, 4, ("container",)),
                obj("TABLE", 7, 0, 4, 2, ("surface",)),
                obj("MARKER", 7, 3, 1, 1),
            ),
            namespace="room",
            frame_id="room-frame",
            scene_id="room-1",
        )["scene"]

        ball_box = types(
            agent.query_spatial_relations(
                scene.scene_id,
                "BALL",
                "BOX",
            )
        )
        box_table = types(
            agent.query_spatial_relations(
                scene.scene_id,
                "BOX",
                "TABLE",
            )
        )
        marker_table = types(
            agent.query_spatial_relations(
                scene.scene_id,
                "MARKER",
                "TABLE",
            )
        )

        # Same scene translated globally: exact coordinates change, structural
        # translation-normalized and relational signatures do not.
        translated = agent.register_spatial_scene(
            (
                obj("BALL", 100, -20, 0, 0, ("movable",)),
                obj("BOX", 100, -20, 4, 4, ("container",)),
                obj("TABLE", 107, -20, 4, 2, ("surface",)),
                obj("MARKER", 107, -17, 1, 1),
            ),
            namespace="room",
            frame_id="room-frame",
            scene_id="room-2",
        )["scene"]

        sig1 = agent.spatial_scene_signatures("room-1")
        sig2 = agent.spatial_scene_signatures("room-2")

        # Abstract relation algebra over partial observations.
        left_ab = core.SpatialRelationAlgebra.make(
            core.SpatialRelationType.LEFT_OF,
            "A",
            "B",
        )
        left_bc = core.SpatialRelationAlgebra.make(
            core.SpatialRelationType.LEFT_OF,
            "B",
            "C",
        )
        closure = agent.close_spatial_relations(
            (left_ab, left_bc)
        )
        closure_keys = {
            (
                relation.subject_id,
                relation.relation_type.value,
                relation.object_id,
            )
            for relation in closure
        }

        inside_ball_box = core.SpatialRelationAlgebra.make(
            core.SpatialRelationType.INSIDE,
            "BALL",
            "BOX",
        )
        inside_box_room = core.SpatialRelationAlgebra.make(
            core.SpatialRelationType.INSIDE,
            "BOX",
            "ROOM",
        )
        containment = agent.close_spatial_relations(
            (inside_ball_box, inside_box_room)
        )
        containment_keys = {
            (
                relation.subject_id,
                relation.relation_type.value,
                relation.object_id,
            )
            for relation in containment
        }

        cycle_rejected = False
        try:
            agent.close_spatial_relations(
                (
                    core.SpatialRelationAlgebra.make(
                        core.SpatialRelationType.ABOVE,
                        "A",
                        "B",
                    ),
                    core.SpatialRelationAlgebra.make(
                        core.SpatialRelationType.ABOVE,
                        "B",
                        "A",
                    ),
                )
            )
        except core.SpatialRelationConflict:
            cycle_rejected = True

        tokens = agent.spatial_relation_tokens(
            "room-1"
        )
        patterns_after_tokenization = (
            agent.structural_pattern_state()[
                "patterns"
            ]
        )

        # Context isolation: same geometry in new Belief Context is another
        # spatial-scene observation scope.
        agent.advance_belief_context(
            observed_at=1,
            reason="room configuration changed",
        )
        ctx1_scene = agent.register_spatial_scene(
            (
                obj("BALL", 0, 0, 0, 0),
                obj("BOX", 0, 0, 4, 4),
            ),
            namespace="room",
            frame_id="room-frame",
            scene_id="room-ctx1",
        )["scene"]

        empirical_after = {
            "evidence": len(agent.all_evidence()),
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        # Portable long-lived state.
        portable = root / "spatial_state.db"
        metadata = agent.save_portable_state(portable)
        source_hash_before = file_hash(portable)
        restored = (
            core.IntegratedCognitiveAgent
            .load_portable_state(portable)
        )

        restored_ball_box = types(
            restored.query_spatial_relations(
                "room-1",
                "BALL",
                "BOX",
            )
        )

        # Runtime mutation after restore must not mutate source portable DB.
        restored.register_spatial_scene(
            (obj("TEMP", 0, 0),),
            namespace="runtime-only",
            scene_id="runtime-scene",
        )
        source_hash_after = file_hash(portable)

        # Fresh process spatial query.
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import json
import sys
from pathlib import Path
sys.path.insert(0, {str(PROJECT_ROOT)!r})
import agen_lab as core

agent = core.IntegratedCognitiveAgent.load_portable_state(
    Path({str(portable)!r})
)
rels = agent.query_spatial_relations(
    "room-1",
    "BOX",
    "TABLE",
)
print("RESULT=" + json.dumps({{
    "version": core.CORE_VERSION,
    "context": agent.belief_contexts.current_id,
    "relations": sorted(r.relation_type.value for r in rels),
    "spatial_scenes": agent.spatial_state(
        namespace="room",
        belief_context_id="ctx-0",
    )["operational_scenes"],
}}))
""",
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        probe_line = next(
            (
                line
                for line in probe.stdout.splitlines()
                if line.startswith("RESULT=")
            ),
            None,
        )
        probe_payload = (
            json.loads(probe_line.split("=", 1)[1])
            if probe_line
            else {}
        )

        checks = {
            "core_v2_33":
                core.CORE_VERSION == "2.42",
            "ball_inside_box":
                ball_box == ["inside"],
            "box_left_of_table":
                (
                    "left_of" in box_table
                    and "disjoint" in box_table
                ),
            "marker_above_table":
                (
                    "above" in marker_table
                    and "disjoint" in marker_table
                ),
            "translation_changes_exact_signature":
                sig1["exact_signature"]
                != sig2["exact_signature"],
            "translation_preserves_normalized_signature":
                sig1["translation_normalized_signature"]
                == sig2["translation_normalized_signature"],
            "translation_preserves_relational_signature":
                sig1["relational_signature"]
                == sig2["relational_signature"],
            "left_relation_transitive":
                ("A", "left_of", "C")
                in closure_keys,
            "inverse_relation_derived":
                ("C", "right_of", "A")
                in closure_keys,
            "inside_relation_transitive":
                ("BALL", "inside", "ROOM")
                in containment_keys,
            "contains_inverse_transitive":
                ("ROOM", "contains", "BALL")
                in containment_keys,
            "strict_cycle_rejected":
                cycle_rejected,
            "relation_tokens_are_explicit":
                len(tokens) > 0
                and all(
                    isinstance(item, tuple)
                    and len(item) == 3
                    for item in tokens
                ),
            "tokenization_does_not_auto_train_pattern":
                patterns_after_tokenization == 0,
            "spatial_ops_do_not_touch_truth_q_world_pattern":
                empirical_before == empirical_after,
            "new_belief_context_has_separate_scene_scope":
                ctx1_scene.belief_context_id == "ctx-1",
            "ctx0_scenes_remain_queryable":
                agent.spatial_state(
                    namespace="room",
                    belief_context_id="ctx-0",
                )["operational_scenes"] == 2,
            "portable_roundtrip_keeps_spatial_relations":
                restored_ball_box == ["inside"],
            "portable_source_db_immutable_after_runtime_change":
                source_hash_before == source_hash_after,
            "portable_state_remains_language_neutral":
                (
                    metadata["language_neutral"] is True
                    and metadata["python_pickle"] is False
                ),
            "fresh_process_spatial_query":
                (
                    probe.returncode == 0
                    and probe_payload.get("version") == "2.42"
                    and probe_payload.get("context") == "ctx-1"
                    and "left_of" in probe_payload.get("relations", [])
                ),
            "transform_now_explicit_but_manipulation_still_absent":
                (
                    agent.spatial_state()["cross_frame_transform"] is True
                    and agent.spatial_state()["cross_frame_transform_model"]
                    == "V2.34_D4_plus_translation"
                    and agent.spatial_state()["manipulation_semantics"] is False
                ),
        }

        result = {
            "core_version": core.CORE_VERSION,
            "room_1": {
                "ball_box": ball_box,
                "box_table": box_table,
                "marker_table": marker_table,
                "signatures": sig1,
            },
            "room_2_signatures": sig2,
            "algebra": {
                "left_closure": sorted(closure_keys),
                "containment_closure": sorted(containment_keys),
                "cycle_rejected": cycle_rejected,
            },
            "portable": metadata,
            "fresh_process": probe_payload,
            "checks": checks,
        }

        failed = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        print(
            "\nFINAL: "
            f"{len(checks)-len(failed)}"
            f"/{len(checks)} PASS"
        )

        if failed:
            print("FAILED:", failed)
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
