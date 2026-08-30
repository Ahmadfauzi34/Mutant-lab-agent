
"""V2.37 focused sandbox — ticketed spatial execution + actual feedback."""
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


def actual_from_ticket(ticket, scene_id, observed_at, dx=0.0, dy=0.0):
    expected = ticket.predicted_scene
    items = []
    for item in expected.objects:
        if item.object_id == "BLOCK":
            items.append(
                core.SpatialObject2D(
                    object_id=item.object_id,
                    pose=core.SpatialPose2D(
                        item.pose.x + dx,
                        item.pose.y + dy,
                    ),
                    extent=item.extent,
                    labels=item.labels,
                )
            )
        else:
            items.append(item)
    return core.make_spatial_scene(
        tuple(items),
        namespace=expected.namespace,
        belief_context_id=expected.belief_context_id,
        frame_id=expected.frame_id,
        scene_id=scene_id,
        observed_at=observed_at,
    )


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="v237_exec_lab_") as td:
        root = Path(td)
        agent = core.IntegratedCognitiveAgent(
            "execution-lab",
            10,
            10,
            epistemic_archive_path=str(root/"cold.sqlite3"),
        )
        agent.register_spatial_scene(
            (
                obj("BLOCK", -6, 0, 4, 2, ("movable",)),
                obj("BOX", 0, 0, 3, 5, ("container",)),
            ),
            namespace="lab",
            scene_id="source",
            observed_at=0,
        )

        plan_result = agent.plan_spatial_manipulation(
            "source",
            core.SpatialRelationGoal(
                "BLOCK",
                core.SpatialRelationType.INSIDE,
                "BOX",
            ),
            (
                core.SpatialManipulationOperator.rotate("BLOCK", 1),
                core.SpatialManipulationOperator.place_inside("BLOCK", "BOX"),
            ),
        )
        plan = plan_result.best_plan

        empirical_before = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        # Step 1 external execution contract.
        t1 = agent.prepare_spatial_plan_execution_step(plan, 1)
        dispatch_json = json.loads(json.dumps(t1.dispatch_descriptor()))
        agent.acknowledge_spatial_execution_dispatch(
            t1.ticket_id,
            external_receipt="robot:step1",
            dispatched_at=1,
        )
        actual1 = actual_from_ticket(
            t1,
            "actual-step1",
            2,
            dx=5e-10,
        )
        f1 = agent.submit_spatial_execution_observation(
            t1.ticket_id,
            actual1,
            tolerance=1e-9,
        )

        # Step 2 is rebased on actual1, not blindly on the original predicted
        # intermediate scene.
        t2 = agent.prepare_spatial_plan_execution_step(
            plan,
            2,
            prepared_at=2,
        )
        agent.acknowledge_spatial_execution_dispatch(
            t2.ticket_id,
            external_receipt="robot:step2",
            dispatched_at=3,
        )
        actual2 = actual_from_ticket(
            t2,
            "actual-step2",
            4,
        )
        f2 = agent.submit_spatial_execution_observation(
            t2.ticket_id,
            actual2,
        )
        feedback_json = json.loads(json.dumps(f2.to_descriptor()))

        # Deviation path must stop continuation.
        agent2 = core.IntegratedCognitiveAgent(
            "deviation-lab",
            8,
            8,
            epistemic_archive_path=str(root/"cold2.sqlite3"),
        )
        agent2.register_spatial_scene(
            (
                obj("BLOCK", -6, 0, 4, 2),
                obj("BOX", 0, 0, 3, 5),
            ),
            namespace="lab",
            scene_id="source",
        )
        plan2 = agent2.plan_spatial_manipulation(
            "source",
            core.SpatialRelationGoal(
                "BLOCK",
                core.SpatialRelationType.INSIDE,
                "BOX",
            ),
            (
                core.SpatialManipulationOperator.rotate("BLOCK", 1),
                core.SpatialManipulationOperator.place_inside("BLOCK", "BOX"),
            ),
        ).best_plan
        d1 = agent2.prepare_spatial_plan_execution_step(plan2, 1)
        agent2.acknowledge_spatial_execution_dispatch(
            d1.ticket_id,
            external_receipt="robot:deviate",
            dispatched_at=1,
        )
        df = agent2.submit_spatial_execution_observation(
            d1.ticket_id,
            actual_from_ticket(
                d1,
                "deviated",
                2,
                dx=0.25,
            ),
            register_actual_scene=False,
        )
        continuation_blocked = False
        try:
            agent2.prepare_spatial_plan_execution_step(plan2, 2)
        except core.SpatialExecutionContinuationBlocked:
            continuation_blocked = True

        empirical_after = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        # Durable mid-flight boundary: prepared ticket survives restart.
        pending = agent.prepare_spatial_plan_execution_step(plan, 1)
        portable = root/"execution-state.db"
        metadata = agent.save_portable_state(portable)
        source_hash_before = file_hash(portable)
        restored = core.IntegratedCognitiveAgent.load_portable_state(portable)
        restored_pending = restored.spatial_execution_ticket(
            pending.ticket_id
        )
        restored_pending_was_prepared = (
            restored_pending.status
            == core.SpatialExecutionTicketStatus.PREPARED
        )
        restored.acknowledge_spatial_execution_dispatch(
            pending.ticket_id,
            external_receipt="robot:after-restart",
            dispatched_at=10,
        )
        restored_feedback = restored.submit_spatial_execution_observation(
            pending.ticket_id,
            actual_from_ticket(
                restored_pending,
                "restart-actual",
                11,
            ),
        )
        source_hash_after = file_hash(portable)

        # Fresh process can read durable ticket and complete feedback.
        fresh_pending = agent.prepare_spatial_plan_execution_step(plan, 1)
        fresh_db = root/"fresh-state.db"
        agent.save_portable_state(fresh_db)
        probe = subprocess.run(
            [sys.executable, "-c", f"""
import json,sys
from pathlib import Path
sys.path.insert(0,{str(PROJECT_ROOT)!r})
import agen_lab as core
a=core.IntegratedCognitiveAgent.load_portable_state(Path({str(fresh_db)!r}))
t=a.spatial_execution_ticket({fresh_pending.ticket_id!r})
a.acknowledge_spatial_execution_dispatch(
    t.ticket_id,
    external_receipt="fresh:receipt",
    dispatched_at=20,
)
e=t.predicted_scene
actual=core.make_spatial_scene(
    e.objects,
    namespace=e.namespace,
    belief_context_id=e.belief_context_id,
    frame_id=e.frame_id,
    scene_id="fresh-actual",
    observed_at=21,
)
f=a.submit_spatial_execution_observation(t.ticket_id,actual)
print("RESULT="+json.dumps({{
    "version":core.CORE_VERSION,
    "ticket_status":a.spatial_execution_ticket(t.ticket_id).status.value,
    "feedback":f.status.value,
    "actual":f.is_actual_observation,
    "q_experience":f.is_q_experience,
    "physical_core":a.spatial_execution_state()[
        "physical_execution_performed_by_core"
    ],
    "replanning":a.spatial_state()["autonomous_spatial_replanning"],
}}))
"""],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        line = next(
            (x for x in probe.stdout.splitlines() if x.startswith("RESULT=")),
            None,
        )
        probe_data = json.loads(line.split("=",1)[1]) if line else {}

        checks = {
            "core_v2_37": core.CORE_VERSION == "2.42",
            "planner_still_finds_two_steps": plan.step_count == 2,
            "ticket_prepared_nonexperience": (
                not t1.is_experience and not t1.was_executed
            ),
            "dispatch_contract_json_safe": (
                dispatch_json["schema"] == "agen-spatial-execution-ticket-v1"
                and dispatch_json["operator"]["kind"] == "rotate"
            ),
            "dispatch_receipt_does_not_claim_execution": (
                t1.external_dispatch_acknowledged
                and not t1.was_executed
            ),
            "step1_actual_matches_with_tolerance": f1.matched,
            "step1_actual_is_not_q_experience": (
                f1.is_actual_observation
                and not f1.is_q_experience
                and not f1.is_evidence
            ),
            "step2_rebased_on_actual_scene": (
                t2.source_scene_id == "actual-step1"
                and t2.source_scene_signature
                == core.SpatialSceneCanonicalizer.exact_signature(actual1)
            ),
            "final_feedback_matches": f2.matched,
            "final_actual_goal_satisfied": f2.goal_observed_satisfied,
            "feedback_contract_json_safe": (
                feedback_json["schema"]
                == "agen-spatial-execution-feedback-v1"
                and feedback_json["status"] == "match"
            ),
            "actual_scenes_registered": (
                agent.spatial_scene("actual-step1") == actual1
                and agent.spatial_scene("actual-step2") == actual2
            ),
            "geometry_deviation_classified": (
                df.status
                == core.SpatialExecutionFeedbackStatus.GEOMETRY_DEVIATION
            ),
            "deviation_blocks_next_step": continuation_blocked,
            "execution_boundary_does_not_train_empirical_models": (
                empirical_before == empirical_after
            ),
            "execution_store_retains_closed_feedback": (
                agent.spatial_execution_feedback(t1.ticket_id) == f1
                and agent.spatial_execution_feedback(t2.ticket_id) == f2
            ),
            "portable_language_neutral": (
                metadata["language_neutral"] is True
                and metadata["python_pickle"] is False
            ),
            "prepared_ticket_survives_restart": (
                restored_pending_was_prepared
            ),
            "restored_ticket_can_complete_actual_feedback": (
                restored_feedback.matched
            ),
            "portable_source_snapshot_immutable": (
                source_hash_before == source_hash_after
            ),
            "fresh_process_completes_ticket": (
                probe.returncode == 0
                and probe_data.get("version") == "2.42"
                and probe_data.get("ticket_status") == "closed"
                and probe_data.get("feedback") == "match"
            ),
            "fresh_feedback_actual_not_q_experience": (
                probe_data.get("actual") is True
                and probe_data.get("q_experience") is False
            ),
            "core_never_claims_physical_execution": (
                probe_data.get("physical_core") is False
            ),
            "autonomous_replanning_still_absent": (
                probe_data.get("replanning") is False
            ),
            "interaction_clock_reaches_actual_observation": (
                agent.interaction_clock == 4
            ),
            "execution_store_is_bounded": (
                agent.spatial_execution_state()["limit"] == 512
            ),
            "planning_capability_preserved": (
                agent.spatial_state()[
                    "spatial_planning_model"
                ] == "V2.36_BOUNDED_BFS_RELATION_GOALS"
            ),
            "counterfactual_manipulation_preserved": (
                agent.spatial_state()["counterfactual_manipulation"] is True
            ),
            "physical_execution_capability_still_false": (
                agent.spatial_state()["physical_manipulation_execution"] is False
            ),
            "feedback_boundary_capability_explicit": (
                agent.spatial_state()["ticketed_spatial_execution_feedback"] is True
            ),
        }

        failed = [k for k,v in checks.items() if not v]
        print(json.dumps({
            "checks": checks,
            "ticket1": dispatch_json,
            "feedback2": feedback_json,
            "deviation_status": df.status.value,
            "execution_state": agent.spatial_execution_state(),
            "fresh_process": probe_data,
        }, indent=2, sort_keys=True))
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
