
"""V2.34 focused sandbox — spatial transformation algebra."""
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


def objects():
    return (
        obj("A", 0, 0, 2, 1, ("a",)),
        obj("B", 3, 1, 1, 3, ("b",)),
        obj("C", -2, 4, 4, 1, ("c",)),
    )


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="v234_transform_lab_") as td:
        root = Path(td)
        agent = core.IntegratedCognitiveAgent(
            "transform-lab", 8, 8,
            epistemic_archive_path=str(root/"cold.sqlite3"),
        )
        source = agent.register_spatial_scene(
            objects(), namespace="lab", frame_id="world", scene_id="source"
        )["scene"]

        empirical_before = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        r90 = core.SpatialTransform2D.from_kind(
            core.SpatialLinearTransformKind.ROTATE_90_CCW,
            source_frame_id="world", target_frame_id="camera",
            tx=10, ty=-2,
        )
        target = agent.apply_spatial_transform(
            "source", r90, target_scene_id="target", register=True
        )
        inference = agent.infer_spatial_transform("source", "target")
        inverse = agent.invert_spatial_transform(r90)
        recovered = inverse.apply_scene(target, scene_id="recovered")
        source_map = source.object_map()
        recovered_map = recovered.object_map()

        translation = core.SpatialTransform2D.translation(
            5, 3, source_frame_id="camera", target_frame_id="screen"
        )
        composed = agent.compose_spatial_transforms(r90, translation)
        p_seq = translation.apply_xy(*r90.apply_xy(2, 5))
        p_cmp = composed.apply_xy(2, 5)

        reflection = core.SpatialTransform2D.from_kind(
            core.SpatialLinearTransformKind.REFLECT_Y_EQ_X,
            source_frame_id="world", tx=2, ty=7,
        )
        reflected = reflection.apply_scene(source, scene_id="reflected")
        reflect_match = core.SpatialTransformationMatcher.infer(
            source, reflected
        )

        symmetric_source = core.make_spatial_scene(
            (obj("S", 0, 0, 2, 2),),
            namespace="sym", belief_context_id="ctx-0",
            frame_id="world", scene_id="sym-source",
        )
        symmetric_target = core.make_spatial_scene(
            (obj("S", 4, 6, 2, 2),),
            namespace="sym", belief_context_id="ctx-0",
            frame_id="world", scene_id="sym-target",
        )
        ambiguous = core.SpatialTransformationMatcher.infer(
            symmetric_source, symmetric_target
        )

        token_r = agent.spatial_transform_token(r90)
        token_f = agent.spatial_transform_token(reflection)
        pattern_before = agent.structural_pattern_state()["patterns"]
        # Explicit adapter learning only.
        agent.observe_structural_sequence(
            (token_r, token_f, token_r, token_f),
            namespace="transform-ops",
            source_id="operator-sequence",
        )
        pred = agent.predict_structural_next(
            (token_r, token_f, token_r),
            namespace="transform-ops",
        )

        descriptor = json.loads(json.dumps(r90.to_descriptor()))
        descriptor_roundtrip = core.SpatialTransform2D.from_descriptor(descriptor)

        empirical_after_transform_before_explicit_pattern = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
        }

        portable = root/"state.db"
        metadata = agent.save_portable_state(portable)
        hash_before = sha(portable)
        restored = core.IntegratedCognitiveAgent.load_portable_state(portable)
        restored_inference = restored.infer_spatial_transform("source", "target")
        restored.apply_spatial_transform(
            "source",
            core.SpatialTransform2D.translation(
                1, 1, source_frame_id="world"
            ),
            target_scene_id="runtime-only",
            register=True,
        )
        hash_after = sha(portable)

        probe = subprocess.run(
            [
                sys.executable, "-c",
                f"""
import json,sys
from pathlib import Path
sys.path.insert(0,{str(PROJECT_ROOT)!r})
import agen_lab as core
a=core.IntegratedCognitiveAgent.load_portable_state(Path({str(portable)!r}))
i=a.infer_spatial_transform("source","target")
t=i.unique_transform
print("RESULT="+json.dumps({{
 "version":core.CORE_VERSION,
 "unique":i.unique,
 "kind":t.linear_kind.value if t else None,
 "source_frame":t.source_frame_id if t else None,
 "target_frame":t.target_frame_id if t else None,
}}))
""",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=180,
        )
        line = next(
            (x for x in probe.stdout.splitlines() if x.startswith("RESULT=")),
            None,
        )
        probe_data = json.loads(line.split("=",1)[1]) if line else {}

        checks = {
            "core_v2_34": core.CORE_VERSION == "2.42",
            "rotate90_target_frame": target.frame_id == "camera",
            "rotate90_swaps_extent": (
                target.object_map()["A"].extent
                == core.SpatialExtent2D(1,2)
            ),
            "unique_before_after_inference": inference.unique,
            "inferred_transform_exact": inference.unique_transform == r90,
            "inference_nonexperience": not inference.is_experience,
            "inference_nontruth": not inference.is_truth,
            "inverse_recovers_positions": all(
                recovered_map[k].pose == source_map[k].pose for k in source_map
            ),
            "inverse_recovers_extents": all(
                recovered_map[k].extent == source_map[k].extent for k in source_map
            ),
            "compose_matches_sequential": p_seq == p_cmp,
            "compose_frame_chain": (
                composed.source_frame_id == "world"
                and composed.target_frame_id == "screen"
            ),
            "reflection_inference_unique": reflect_match.unique,
            "reflection_kind_correct": (
                reflect_match.unique_transform.linear_kind.value
                == "reflect_y_eq_x"
            ),
            "symmetric_scene_reports_ambiguity": (
                ambiguous.ambiguous and len(ambiguous.candidates) == 8
            ),
            "transform_token_does_not_auto_train": pattern_before == 0,
            "explicit_pattern_adapter_predicts_transform_class": (
                pred.expected_symbol == token_f
            ),
            "descriptor_json_roundtrip": descriptor_roundtrip == r90,
            "transform_ops_do_not_touch_q_world_evidence": (
                empirical_before["q"] == empirical_after_transform_before_explicit_pattern["q"]
                and empirical_before["world"] == empirical_after_transform_before_explicit_pattern["world"]
                and empirical_before["joint"] == empirical_after_transform_before_explicit_pattern["joint"]
                and empirical_before["evidence"] == empirical_after_transform_before_explicit_pattern["evidence"]
            ),
            "portable_language_neutral": (
                metadata["language_neutral"] is True
                and metadata["python_pickle"] is False
            ),
            "portable_keeps_scene_transform_inference": restored_inference.unique,
            "portable_source_immutable_after_restored_runtime_change": hash_before == hash_after,
            "fresh_process_transform_inference": (
                probe.returncode == 0
                and probe_data.get("version") == "2.42"
                and probe_data.get("unique") is True
                and probe_data.get("kind") == "rotate_90_ccw"
                and probe_data.get("source_frame") == "world"
                and probe_data.get("target_frame") == "camera"
            ),
            "cross_frame_transform_explicit": (
                agent.spatial_state()["cross_frame_transform"] is True
            ),
            "manipulation_still_absent": (
                agent.spatial_state()["manipulation_semantics"] is False
            ),
        }
        failed = [k for k,v in checks.items() if not v]
        print(json.dumps({
            "checks": checks,
            "inference_candidates": len(inference.candidates),
            "ambiguous_candidates": len(ambiguous.candidates),
            "transform": r90.to_descriptor(),
            "composed": composed.to_descriptor(),
            "fresh_process": probe_data,
        }, indent=2, sort_keys=True))
        print(f"\\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
