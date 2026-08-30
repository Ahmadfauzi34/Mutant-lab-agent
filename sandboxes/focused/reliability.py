"""V2.40 focused sandbox — empirical manipulation reliability."""
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


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_place_plan(agent, source_id="place-source"):
    if source_id not in agent.spatial_scenes.scenes:
        agent.register_spatial_scene(
            (
                obj("BLOCK", -6, 0, 2, 2, ("movable",)),
                obj("BOX", 0, 0, 6, 6, ("container",)),
            ),
            namespace="lab",
            scene_id=source_id,
            observed_at=0,
        )
    return agent.plan_spatial_manipulation(
        source_id,
        core.SpatialRelationGoal(
            "BLOCK", core.SpatialRelationType.INSIDE, "BOX"
        ),
        (core.SpatialManipulationOperator.place_inside("BLOCK", "BOX"),),
    ).best_plan


def execute(agent, plan, ordinal, mode="match"):
    t = agent.prepare_spatial_plan_execution_step(plan)
    agent.acknowledge_spatial_execution_dispatch(
        t.ticket_id,
        external_receipt=f"lab:{ordinal}",
        dispatched_at=ordinal * 2 - 1,
    )
    emap = t.predicted_scene.object_map()
    block = emap["BLOCK"]
    items = [block, emap["BOX"]]
    namespace = t.predicted_scene.namespace
    if mode == "geometry":
        items[0] = obj(
            "BLOCK",
            block.pose.x + 0.25,
            block.pose.y,
            block.extent.width,
            block.extent.height,
            block.labels,
        )
    elif mode == "namespace":
        namespace = "other"
    actual = core.make_spatial_scene(
        tuple(items),
        namespace=namespace,
        belief_context_id=t.predicted_scene.belief_context_id,
        frame_id=t.predicted_scene.frame_id,
        scene_id=f"actual-{ordinal}-{mode}",
        observed_at=ordinal * 2,
    )
    f = agent.submit_spatial_execution_observation(
        t.ticket_id,
        actual,
        register_actual_scene=False,
    )
    return t, f


def make_recovery(agent):
    agent.register_spatial_scene(
        (
            obj("BLOCK", -6, 0, 4, 2, ("movable",)),
            obj("BOX", 0, 0, 3, 5, ("container",)),
        ),
        namespace="lab",
        scene_id="recovery-source",
        observed_at=0,
    )
    original = agent.plan_spatial_manipulation(
        "recovery-source",
        core.SpatialRelationGoal(
            "BLOCK", core.SpatialRelationType.INSIDE, "BOX"
        ),
        (
            core.SpatialManipulationOperator.rotate("BLOCK", 1),
            core.SpatialManipulationOperator.place_inside("BLOCK", "BOX"),
        ),
    ).best_plan
    t = agent.prepare_spatial_plan_execution_step(original, 1)
    agent.acknowledge_spatial_execution_dispatch(
        t.ticket_id, external_receipt="recovery:trigger", dispatched_at=100
    )
    emap = t.predicted_scene.object_map()
    block = emap["BLOCK"]
    actual = core.make_spatial_scene(
        (
            obj(
                "BLOCK",
                block.pose.x + 0.25,
                block.pose.y,
                block.extent.width,
                block.extent.height,
                block.labels,
            ),
            emap["BOX"],
        ),
        namespace="lab",
        belief_context_id="ctx-0",
        frame_id="world",
        scene_id="recovery-actual",
        observed_at=101,
    )
    feedback = agent.submit_spatial_execution_observation(
        t.ticket_id, actual, register_actual_scene=False
    )
    replan = agent.replan_spatial_after_execution_deviation(
        original,
        t.ticket_id,
        (core.SpatialManipulationOperator.place_inside("BLOCK", "BOX"),),
        requested_at=101,
    )
    decision = agent.evaluate_spatial_recovery(
        original,
        t.ticket_id,
        replan_id=replan.replan_id,
        evaluated_at=101,
    )
    return original, t, feedback, replan, decision


def main():
    with tempfile.TemporaryDirectory(prefix="v240_rel_lab_") as td:
        root = Path(td)
        agent = core.IntegratedCognitiveAgent(
            "reliability-lab",
            10,
            10,
            epistemic_archive_path=str(root / "cold.sqlite3"),
        )
        plan = make_place_plan(agent)
        empirical_before = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        updates = []
        for i in range(1, 5):
            _, f = execute(agent, plan, i, "match")
            updates.append(agent.spatial_reliability_update(f.feedback_id))

        estimate = agent.spatial_reliability.estimate_operator(
            plan.final_scene, plan.steps[0].operator
        )

        # Non-comparable namespace feedback must not lower reliability revision.
        rev_before_excluded = agent.spatial_reliability.reliability_revision
        _, excluded_feedback = execute(agent, plan, 5, "namespace")
        excluded_update = agent.spatial_reliability_update(
            excluded_feedback.feedback_id
        )
        rev_after_excluded = agent.spatial_reliability.reliability_revision

        _, trigger, _, replan, decision = make_recovery(agent)
        assessment = agent.assess_spatial_recovery_reliability(
            decision.recovery_id
        )
        gated_ticket = agent.prepare_reliability_gated_spatial_recovery_handoff(
            decision.recovery_id,
            assessment_id=assessment.assessment_id,
        )

        empirical_after = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        # Separate low-reliability branch: 3 MATCH + 1 GEOMETRY deviation.
        low = core.IntegratedCognitiveAgent(
            "low-reliability",
            8,
            8,
            epistemic_archive_path=str(root / "low.sqlite3"),
        )
        low_plan = make_place_plan(low)
        for i in range(1, 4):
            execute(low, low_plan, i, "match")
        execute(low, low_plan, 4, "geometry")
        _, _, _, _, low_decision = make_recovery(low)
        low_assessment = low.assess_spatial_recovery_reliability(
            low_decision.recovery_id
        )
        low_gate_blocked = False
        try:
            low.prepare_reliability_gated_spatial_recovery_handoff(
                low_decision.recovery_id,
                assessment_id=low_assessment.assessment_id,
            )
        except core.SpatialReliabilityGateBlocked:
            low_gate_blocked = True

        # Persistence and fresh-process recovery confidence.
        portable = root / "state.db"
        metadata = agent.save_portable_state(portable)
        source_hash_before = file_hash(portable)
        restored = core.IntegratedCognitiveAgent.load_portable_state(portable)
        restored_assessment = restored.spatial_reliability_assessment(
            assessment.assessment_id
        )
        source_hash_after = file_hash(portable)

        probe = subprocess.run(
            [sys.executable, "-c", f'''
import json,sys
from pathlib import Path
sys.path.insert(0,{str(PROJECT_ROOT)!r})
import agen_lab as core
a=core.IntegratedCognitiveAgent.load_portable_state(Path({str(portable)!r}))
ass=a.spatial_reliability_assessment({assessment.assessment_id!r})
t=a.prepare_reliability_gated_spatial_recovery_handoff(
    {decision.recovery_id!r}, assessment_id=ass.assessment_id
)
print("RESULT="+json.dumps({{
 "version":core.CORE_VERSION,
 "assessment":ass.status.value,
 "coverage":ass.coverage,
 "ticket":t.status.value,
 "receipt":t.external_receipt,
 "revision":a.spatial_reliability_state()["reliability_revision"],
 "physical":a.spatial_state()["physical_manipulation_execution"],
 "auto_gate":a.spatial_state()["automatic_reliability_gate"],
}}))
'''],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        line = next(
            (x for x in probe.stdout.splitlines() if x.startswith("RESULT=")),
            None,
        )
        probe_data = json.loads(line.split("=", 1)[1]) if line else {}

        checks = {
            "core_v2_40": core.CORE_VERSION == "2.42",
            "four_actual_matches_counted": all(u.counted and u.matched for u in updates),
            "revision_four_after_four_matches": rev_before_excluded == 4,
            "exact_estimate_selected": estimate.aggregation_level == core.SpatialReliabilityAggregationLevel.EXACT_OPERATOR,
            "exact_sample_count_four": estimate.sample_count == 4,
            "exact_match_rate_one": estimate.empirical_match_rate == 1.0,
            "posterior_uncertainty_nonzero": estimate.posterior_stddev > 0.0,
            "wilson_lower_bound_passes_default": estimate.wilson_lower_bound >= 0.50,
            "namespace_mismatch_excluded": excluded_update.disposition == core.SpatialReliabilityUpdateDisposition.EXCLUDED_NONCOMPARABLE,
            "excluded_feedback_does_not_change_revision": rev_before_excluded == rev_after_excluded,
            "recovery_structurally_handoff_eligible": decision.action == core.SpatialRecoveryAction.HANDOFF_REPLACEMENT,
            "reliability_assessment_trusted": assessment.status == core.SpatialRecoveryReliabilityStatus.TRUSTED,
            "reliability_full_coverage": assessment.coverage == 1.0,
            "conservative_score_not_joint_probability": not assessment.is_joint_success_probability,
            "trusted_gate_prepares_ticket": gated_ticket.status == core.SpatialExecutionTicketStatus.PREPARED,
            "trusted_gate_does_not_dispatch": gated_ticket.external_receipt is None,
            "trusted_gate_does_not_execute": not gated_ticket.was_executed,
            "low_reliability_below_threshold": low_assessment.status == core.SpatialRecoveryReliabilityStatus.BELOW_THRESHOLD,
            "low_reliability_gate_blocked": low_gate_blocked,
            "learning_does_not_touch_q_world_evidence_pattern": empirical_before == empirical_after,
            "actual_feedback_only_flag": agent.spatial_reliability_state()["actual_closed_feedback_only"] is True,
            "simulation_training_false": agent.spatial_reliability_state()["simulation_training"] is False,
            "planning_training_false": agent.spatial_reliability_state()["planning_training"] is False,
            "portable_language_neutral": metadata["language_neutral"] is True and metadata["python_pickle"] is False,
            "assessment_survives_portable_restart": restored_assessment == assessment,
            "portable_source_immutable": source_hash_before == source_hash_after,
            "fresh_process_trusted_gate": probe.returncode == 0 and probe_data.get("version") == "2.42" and probe_data.get("assessment") == "trusted" and probe_data.get("ticket") == "prepared",
            "fresh_process_no_dispatch": probe_data.get("receipt") is None,
            "physical_execution_still_false": probe_data.get("physical") is False,
            "automatic_reliability_gate_still_false": probe_data.get("auto_gate") is False,
        }

        failed = [name for name, passed in checks.items() if not passed]
        print(json.dumps({
            "checks": checks,
            "estimate": {
                "level": estimate.aggregation_level.value,
                "samples": estimate.sample_count,
                "matches": estimate.match_count,
                "rate": estimate.empirical_match_rate,
                "posterior_mean": estimate.posterior_mean,
                "posterior_stddev": estimate.posterior_stddev,
                "wilson_lower_bound": estimate.wilson_lower_bound,
            },
            "assessment": assessment.to_descriptor(),
            "low_assessment": low_assessment.to_descriptor(),
            "state": agent.spatial_reliability_state(),
            "fresh_process": probe_data,
        }, indent=2, sort_keys=True))
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
