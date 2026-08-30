
"""V2.32 focused sandbox — structural pattern cognition."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agen_lab as core


def main():
    with tempfile.TemporaryDirectory(
        prefix="v232_pattern_lab_"
    ) as temp:
        root = Path(temp)

        agent = core.IntegratedCognitiveAgent(
            "pattern-laboratory",
            4,
            4,
            epistemic_archive_path=str(
                root / "cold.sqlite3"
            ),
        )

        empirical_before = {
            "evidence": len(agent.all_evidence()),
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
        }

        colors = agent.observe_structural_sequence(
            ("red", "blue", "red", "blue"),
            namespace="alternation",
            source_id="colors-1",
        )
        numbers = agent.observe_structural_sequence(
            (7, 3, 7, 3),
            namespace="alternation",
            source_id="numbers-1",
        )

        periodic = [
            item
            for item in agent.structural_pattern_hypotheses(
                namespace="alternation",
                kind=core.PatternKind.PERIODIC_SEQUENCE,
            )
        ][0]

        same_scoped_hypothesis = (
            next(
                item
                for item in colors["discovered_patterns"]
                if item.definition.kind
                == core.PatternKind.PERIODIC_SEQUENCE
            )
            is next(
                item
                for item in numbers["discovered_patterns"]
                if item.definition.kind
                == core.PatternKind.PERIODIC_SEQUENCE
            )
        )

        support_before_prediction = periodic.support_count
        prediction = agent.predict_structural_next(
            ("LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT"),
            namespace="alternation",
        )
        assessment = agent.assess_structural_prediction(
            prediction.prediction_id,
            "RIGHT",
        )
        reliability_after_success = (
            periodic.prediction_reliability
        )

        wrong_prediction = agent.predict_structural_next(
            ("UP", "DOWN", "UP", "DOWN", "UP"),
            namespace="alternation",
        )
        wrong_assessment = agent.assess_structural_prediction(
            wrong_prediction.prediction_id,
            "NOT_DOWN",
        )
        reliability_after_failure = (
            periodic.prediction_reliability
        )

        mirror_a = agent.discover_structural_patterns(
            ("A", "B", "C", "B", "A")
        )
        mirror_b = agent.discover_structural_patterns(
            (1, 2, 3, 2, 1)
        )
        mirror_sig_a = next(
            item.semantic_signature
            for item in mirror_a
            if item.kind == core.PatternKind.MIRROR_SYMMETRY
        )
        mirror_sig_b = next(
            item.semantic_signature
            for item in mirror_b
            if item.kind == core.PatternKind.MIRROR_SYMMETRY
        )

        near_pattern = agent.discover_structural_patterns(
            ("A", "B", "A", "B", "C")
        )

        completion = (
            agent.structural_pattern_relational_completion(
                colors["instance"].instance_id,
                max_depth=2,
            )
        )
        reached = {
            item["node_id"]
            for item in completion["reached"]
        }

        topology_before = copy.deepcopy(
            agent.structural_patterns.patterns
        )
        topology = agent.structural_pattern_topology_audit(
            namespace="alternation",
        )
        topology_after = copy.deepcopy(
            agent.structural_patterns.patterns
        )

        old_pattern_id = periodic.definition.pattern_id
        old_semantic_signature = periodic.definition.semantic_signature

        agent.advance_belief_context(
            observed_at=1,
            reason="new structural regime",
        )
        prediction_before_learning_ctx1 = (
            agent.predict_structural_next(
                ("a", "b", "a"),
                namespace="alternation",
            )
        )

        ctx1 = agent.observe_structural_sequence(
            ("foo", "bar", "foo", "bar"),
            namespace="alternation",
            source_id="ctx1-1",
        )
        ctx1_periodic = next(
            item
            for item in ctx1["discovered_patterns"]
            if item.definition.kind
            == core.PatternKind.PERIODIC_SEQUENCE
        )

        empirical_after = {
            "evidence": len(agent.all_evidence()),
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
        }

        portable = root / "pattern_state.db"
        metadata = agent.save_portable_state(portable)
        restored = (
            core.IntegratedCognitiveAgent
            .load_portable_state(portable)
        )
        restored_ctx1 = (
            restored.structural_pattern_hypotheses(
                namespace="alternation",
                belief_context_id="ctx-1",
                kind=core.PatternKind.PERIODIC_SEQUENCE,
            )[0]
        )

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
p = agent.predict_structural_next(
    ("sun", "moon", "sun"),
    namespace="alternation",
    belief_context_id="ctx-1",
)
print("RESULT=" + json.dumps({{
    "version": core.CORE_VERSION,
    "expected": p.expected_symbol if p else None,
    "context": agent.belief_contexts.current_id,
}}))
""",
            ],
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
            "core_v2_32":
                core.CORE_VERSION == "2.42",
            "symbol_renaming_reuses_same_hypothesis":
                same_scoped_hypothesis,
            "support_count_two":
                periodic.support_count == 2,
            "pattern_prediction_binds_raw_symbol":
                prediction.expected_symbol == "RIGHT",
            "correct_prediction_increases_reliability":
                (
                    assessment.correct
                    and reliability_after_success > 0.5
                ),
            "wrong_prediction_lowers_reliability":
                (
                    not wrong_assessment.correct
                    and reliability_after_failure
                    < reliability_after_success
                ),
            "prediction_does_not_add_pattern_support":
                periodic.support_count
                == support_before_prediction,
            "mirror_signature_is_renaming_invariant":
                mirror_sig_a == mirror_sig_b,
            "near_pattern_not_fuzzily_accepted":
                near_pattern == (),
            "relational_completion_reaches_sibling_instance":
                numbers["instance"].instance_id
                in reached,
            "topology_audit_nonlearning":
                topology_before == topology_after,
            "topology_one_pattern_two_instances":
                (
                    topology["pattern_count"] == 1
                    and topology["instance_count"] == 2
                    and topology["connected_components"] == 1
                ),
            "belief_context_shift_starts_fresh_pattern_scope":
                prediction_before_learning_ctx1 is None,
            "same_structure_new_context_same_semantic_signature":
                ctx1_periodic.definition.semantic_signature
                == old_semantic_signature,
            "same_structure_new_context_distinct_pattern_identity":
                ctx1_periodic.definition.pattern_id
                != old_pattern_id,
            "pattern_subsystem_does_not_touch_truth_q_world":
                empirical_before == empirical_after,
            "portable_roundtrip_preserves_pattern_scope":
                (
                    metadata["core_version"] == "2.42"
                    and restored_ctx1.definition.pattern_id
                    == ctx1_periodic.definition.pattern_id
                ),
            "fresh_process_portable_pattern_prediction":
                (
                    probe.returncode == 0
                    and probe_payload.get("version") == "2.42"
                    and probe_payload.get("expected") == "moon"
                    and probe_payload.get("context") == "ctx-1"
                ),
        }

        result = {
            "checks": checks,
            "periodic": {
                "pattern_id": periodic.definition.pattern_id,
                "semantic_signature":
                    periodic.definition.semantic_signature,
                "support_count": periodic.support_count,
                "prediction_trials": periodic.prediction_trials,
                "prediction_correct": periodic.prediction_correct,
                "prediction_reliability":
                    periodic.prediction_reliability,
            },
            "prediction": {
                "expected": prediction.expected_symbol,
                "correct": assessment.correct,
                "reliability_after_success":
                    reliability_after_success,
                "reliability_after_failure":
                    reliability_after_failure,
            },
            "topology": topology,
            "ctx1_pattern": {
                "pattern_id": ctx1_periodic.definition.pattern_id,
                "semantic_signature":
                    ctx1_periodic.definition.semantic_signature,
            },
            "portable": metadata,
            "fresh_process": probe_payload,
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
