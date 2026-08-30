
"""V2.35 focused sandbox — counterfactual spatial manipulation."""

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


def obj(i, x, y, w=1.0, h=1.0, labels=()):
    return core.SpatialObject2D(
        object_id=i,
        pose=core.SpatialPose2D(x, y),
        extent=core.SpatialExtent2D(w, h),
        labels=tuple(labels),
    )


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="v235_manip_lab_") as td:
        root = Path(td)
        agent = core.IntegratedCognitiveAgent(
            "manipulation-lab",
            10,
            10,
            epistemic_archive_path=str(root/"cold.sqlite3"),
        )

        source = agent.register_spatial_scene(
            (
                obj("BLOCK", -8, 0, 2, 1, ("movable",)),
                obj("BOX", 0, 0, 6, 6, ("container",)),
                obj("TABLE", 8, 0, 6, 2, ("support",)),
            ),
            namespace="lab",
            frame_id="world",
            scene_id="source",
        )["scene"]

        collision_scene = agent.register_spatial_scene(
            (
                obj("BLOCK", -8, 0, 2, 2),
                obj("TABLE", 8, 0, 6, 2),
                obj("OBSTACLE", 8, 2, 2, 2),
            ),
            namespace="lab",
            frame_id="world",
            scene_id="collision-scene",
        )["scene"]

        empirical_before = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }
        scene_count_before = agent.spatial_state(
            namespace="lab"
        )["total_scenes_registered"]

        move_op = core.SpatialManipulationOperator.move(
            "BLOCK", -1, 0
        )
        move = agent.simulate_spatial_manipulation(
            "source",
            move_op,
            predicted_scene_id="cf-move",
        )

        rotate_op = core.SpatialManipulationOperator.rotate(
            "BLOCK", 1
        )
        rotate = agent.simulate_spatial_manipulation(
            "source",
            rotate_op,
            predicted_scene_id="cf-rotate",
        )

        place_op = core.SpatialManipulationOperator.place_inside(
            "BLOCK", "BOX"
        )
        place = agent.simulate_spatial_manipulation(
            "source",
            place_op,
            predicted_scene_id="cf-place",
        )

        stack_op = core.SpatialManipulationOperator.stack_above(
            "BLOCK", "TABLE"
        )
        stack = agent.simulate_spatial_manipulation(
            "source",
            stack_op,
            predicted_scene_id="cf-stack",
        )

        # Chaining over ephemeral scene state.
        chain_rotate = agent.simulate_spatial_manipulation(
            "source",
            rotate_op,
            predicted_scene_id="chain-rotate",
        )
        chain_place = agent.simulate_spatial_manipulation_on_scene(
            chain_rotate.predicted_scene,
            place_op,
            predicted_scene_id="chain-place",
        )

        # Adversarial failures.
        zero_move = agent.simulate_spatial_manipulation(
            "source",
            core.SpatialManipulationOperator.move(
                "BLOCK", 0, 0
            ),
        )
        rotate_180 = agent.simulate_spatial_manipulation(
            "source",
            core.SpatialManipulationOperator.rotate(
                "BLOCK", 2
            ),
        )
        too_large = agent.simulate_spatial_manipulation(
            "source",
            core.SpatialManipulationOperator.place_inside(
                "BOX", "BLOCK"
            ),
        )
        collision_stack = agent.simulate_spatial_manipulation(
            "collision-scene",
            core.SpatialManipulationOperator.stack_above(
                "BLOCK", "TABLE"
            ),
        )

        empirical_after_sim = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }
        scene_count_after = agent.spatial_state(
            namespace="lab"
        )["total_scenes_registered"]

        # Default token is operator-class abstraction only.
        move_token = agent.spatial_manipulation_token(move_op)
        rotate_token = agent.spatial_manipulation_token(rotate_op)
        place_token = agent.spatial_manipulation_token(place_op)
        stack_token = agent.spatial_manipulation_token(stack_op)

        pattern_before_explicit = agent.structural_pattern_state()[
            "patterns"
        ]
        agent.observe_structural_sequence(
            (
                move_token,
                rotate_token,
                move_token,
                rotate_token,
            ),
            namespace="manipulation-operators",
            source_id="manip-sequence",
        )
        operator_prediction = agent.predict_structural_next(
            (
                move_token,
                rotate_token,
                move_token,
            ),
            namespace="manipulation-operators",
        )

        descriptor = json.loads(
            json.dumps(place_op.to_descriptor())
        )
        descriptor_roundtrip = (
            core.SpatialManipulationOperator
            .from_descriptor(descriptor)
        )

        portable = root/"state.db"
        metadata = agent.save_portable_state(portable)
        db_hash_before = file_hash(portable)
        restored = core.IntegratedCognitiveAgent.load_portable_state(
            portable
        )
        restored_place = restored.simulate_spatial_manipulation(
            "source",
            place_op,
        )
        restored.register_spatial_scene(
            (obj("RUNTIME", 100, 100),),
            namespace="runtime",
            scene_id="runtime-scene",
        )
        db_hash_after = file_hash(portable)

        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import json,sys
from pathlib import Path
sys.path.insert(0,{str(PROJECT_ROOT)!r})
import agen_lab as core
a=core.IntegratedCognitiveAgent.load_portable_state(
    Path({str(portable)!r})
)
op=core.SpatialManipulationOperator.place_inside(
    "BLOCK","BOX"
)
r=a.simulate_spatial_manipulation("source",op)
print("RESULT="+json.dumps({{
    "version":core.CORE_VERSION,
    "feasible":r.feasible,
    "inside":"inside" in [
        x.value for x in r.predicted_subject_target_relations
    ],
    "executed":r.was_executed,
    "counterfactual":a.spatial_state()[
        "counterfactual_manipulation"
    ],
    "physical":a.spatial_state()[
        "physical_manipulation_execution"
    ],
}}))
""",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        line = next(
            (
                x for x in probe.stdout.splitlines()
                if x.startswith("RESULT=")
            ),
            None,
        )
        probe_data = (
            json.loads(line.split("=", 1)[1])
            if line else {}
        )

        checks = {
            "core_v2_35": core.CORE_VERSION == "2.42",
            "move_feasible": move.feasible,
            "move_pose_correct": (
                move.subject_after.pose
                == core.SpatialPose2D(-9, 0)
            ),
            "rotate_feasible": rotate.feasible,
            "rotate_swaps_extent": (
                rotate.subject_after.extent
                == core.SpatialExtent2D(1, 2)
            ),
            "place_inside_feasible": place.feasible,
            "place_predicts_inside": (
                core.SpatialRelationType.INSIDE
                in place.predicted_subject_target_relations
            ),
            "stack_feasible": stack.feasible,
            "stack_predicts_touching": (
                core.SpatialRelationType.TOUCHING
                in stack.predicted_subject_target_relations
            ),
            "counterfactual_chain_feasible": (
                chain_rotate.feasible
                and chain_place.feasible
                and chain_place.source_scene_id
                == "chain-rotate"
            ),
            "zero_move_rejected": not zero_move.feasible,
            "rotation_180_not_silently_faked": (
                not rotate_180.feasible
            ),
            "oversize_place_rejected": not too_large.feasible,
            "collision_stack_rejected": (
                not collision_stack.feasible
                and collision_stack.collisions
                and collision_stack.collisions[0].other_object_id
                == "OBSTACLE"
            ),
            "simulation_is_nonexperience": (
                not place.is_experience
                and not place.is_evidence
                and not place.was_executed
            ),
            "simulations_do_not_mutate_q_world_evidence_pattern": (
                empirical_before == empirical_after_sim
            ),
            "simulations_do_not_create_scene_history": (
                scene_count_before == scene_count_after
            ),
            "predicted_scene_not_registered": (
                "cf-place" not in agent.spatial_scenes.scenes
            ),
            "token_default_is_class_only": (
                move_token
                == ("spatial_manipulation", "move")
            ),
            "token_generation_did_not_auto_train": (
                pattern_before_explicit == 0
            ),
            "explicit_pattern_adapter_predicts_operator_class": (
                operator_prediction.expected_symbol
                == rotate_token
            ),
            "operator_descriptor_json_roundtrip": (
                descriptor_roundtrip == place_op
            ),
            "portable_language_neutral": (
                metadata["language_neutral"] is True
                and metadata["python_pickle"] is False
            ),
            "restored_scene_can_simulate": (
                restored_place.feasible
                and core.SpatialRelationType.INSIDE
                in restored_place.predicted_subject_target_relations
            ),
            "source_portable_db_immutable_after_restored_runtime_change": (
                db_hash_before == db_hash_after
            ),
            "fresh_process_counterfactual_simulation": (
                probe.returncode == 0
                and probe_data.get("version") == "2.42"
                and probe_data.get("feasible") is True
                and probe_data.get("inside") is True
                and probe_data.get("executed") is False
            ),
            "counterfactual_capability_explicit": (
                probe_data.get("counterfactual") is True
            ),
            "physical_execution_still_absent": (
                probe_data.get("physical") is False
            ),
            "transform_capability_preserved": (
                agent.spatial_state()[
                    "cross_frame_transform_model"
                ] == "V2.34_D4_plus_translation"
            ),
        }

        failed = [
            name for name, passed in checks.items()
            if not passed
        ]

        print(json.dumps({
            "checks": checks,
            "place_descriptor": descriptor,
            "place_preconditions": [
                {
                    "kind": c.kind.value,
                    "passed": c.passed,
                    "detail": c.detail,
                }
                for c in place.preconditions
            ],
            "collision_failure": [
                {
                    "other": c.other_object_id,
                    "relations": [
                        r.value for r in c.relation_types
                    ],
                }
                for c in collision_stack.collisions
            ],
            "fresh_process": probe_data,
        }, indent=2, sort_keys=True))

        print(
            f"\nFINAL: "
            f"{len(checks)-len(failed)}/{len(checks)} PASS"
        )
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
