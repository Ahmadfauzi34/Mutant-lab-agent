"""Focused sandbox — reliability-ranked derived view of completed replans."""
from __future__ import annotations

import copy
import json
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


def train(agent, operator, matches, deviations, start):
    scene_id = f"rank-train-{start}-{abs(hash(operator.semantic_signature)) % 100000}"
    agent.register_spatial_scene(
        (
            obj("BLOCK", -6, 20 + start, 2, 4, ("movable",)),
            obj("BOX", 0, 20 + start, 3, 5, ("container",)),
        ),
        namespace="lab",
        scene_id=scene_id,
        observed_at=0,
    )
    plan = agent.plan_spatial_manipulation(
        scene_id,
        core.SpatialRelationGoal(
            "BLOCK", core.SpatialRelationType.INSIDE, "BOX"
        ),
        (operator,),
    ).best_plan
    n = start
    for mode, count in (("match", matches), ("deviation", deviations)):
        for _ in range(count):
            ticket = agent.prepare_spatial_plan_execution_step(plan, 1)
            agent.acknowledge_spatial_execution_dispatch(
                ticket.ticket_id,
                external_receipt=f"rank:{mode}:{n}",
                dispatched_at=n * 2 - 1,
            )
            expected = ticket.predicted_scene
            if mode == "match":
                objects = expected.objects
            else:
                emap = expected.object_map()
                block = emap["BLOCK"]
                objects = (
                    obj(
                        "BLOCK",
                        block.pose.x + 0.25,
                        block.pose.y,
                        block.extent.width,
                        block.extent.height,
                        block.labels,
                    ),
                    emap["BOX"],
                )
            actual = core.make_spatial_scene(
                objects,
                namespace=expected.namespace,
                belief_context_id=expected.belief_context_id,
                frame_id=expected.frame_id,
                scene_id=f"rank-actual-{mode}-{n}",
                observed_at=n * 2,
            )
            agent.submit_spatial_execution_observation(
                ticket.ticket_id,
                actual,
                register_actual_scene=False,
            )
            n += 1
    return n


def main():
    with tempfile.TemporaryDirectory(prefix="replan_rank_sandbox_") as td:
        root = Path(td)
        agent = core.IntegratedCognitiveAgent(
            "replan-rank-sandbox",
            10,
            10,
            epistemic_archive_path=str(root / "cold.sqlite3"),
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
        goal = core.SpatialRelationGoal(
            "BLOCK", core.SpatialRelationType.INSIDE, "BOX"
        )
        rotate = core.SpatialManipulationOperator.rotate("BLOCK", 1)
        place = core.SpatialManipulationOperator.place_inside("BLOCK", "BOX")
        offset = core.SpatialManipulationOperator.place_inside(
            "BLOCK", "BOX", offset_x=0.2
        )
        original = agent.plan_spatial_manipulation(
            "source", goal, (rotate, place), max_solutions=8
        ).best_plan
        ticket = agent.prepare_spatial_plan_execution_step(original, 1)
        agent.acknowledge_spatial_execution_dispatch(
            ticket.ticket_id,
            external_receipt="sandbox:deviation",
            dispatched_at=1,
        )
        emap = ticket.predicted_scene.object_map()
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
            namespace=ticket.predicted_scene.namespace,
            belief_context_id=ticket.predicted_scene.belief_context_id,
            frame_id=ticket.predicted_scene.frame_id,
            scene_id="actual-deviation",
            observed_at=2,
        )
        feedback = agent.submit_spatial_execution_observation(
            ticket.ticket_id, actual
        )
        record = agent.replan_spatial_after_execution_deviation(
            original,
            ticket.ticket_id,
            (place, offset),
            max_depth=3,
            max_nodes=64,
            max_solutions=8,
            requested_at=3,
        )
        first, second = record.planning_result.solutions
        n = train(agent, first.steps[0].operator, 3, 1, 10)
        train(agent, second.steps[0].operator, 4, 0, n)

        before = {
            "clock": agent.interaction_clock,
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "evidence": len(agent.all_evidence()),
            "replan": copy.deepcopy(agent.spatial_replanning.state()),
            "reliability": copy.deepcopy(agent.spatial_reliability.state()),
        }
        view = agent.rank_spatial_replan_by_reliability(
            record.replan_id, ranked_at=777
        )
        after = {
            "clock": agent.interaction_clock,
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "evidence": len(agent.all_evidence()),
            "replan": copy.deepcopy(agent.spatial_replanning.state()),
            "reliability": copy.deepcopy(agent.spatial_reliability.state()),
        }

        recovery = agent.evaluate_spatial_recovery(
            original,
            ticket.ticket_id,
            replan_id=record.replan_id,
        )
        portable = root / "state.db"
        meta = agent.save_portable_state(portable)
        restored = core.IntegratedCognitiveAgent.load_portable_state(portable)
        restored_view = restored.rank_spatial_replan_by_reliability(
            record.replan_id
        )

        descriptor = view.to_descriptor()
        json.dumps(descriptor, sort_keys=True, allow_nan=False)
        state = agent.spatial_state()
        checks = {
            "core_semantic_version": core.CORE_VERSION == "2.42",
            "actual_deviation_closed": feedback.status
            == core.SpatialExecutionFeedbackStatus.GEOMETRY_DEVIATION,
            "replan_found": record.status == core.SpatialPlanningStatus.FOUND,
            "two_equal_depth_candidates": len(record.planning_result.solutions) == 2
            and len({p.step_count for p in record.planning_result.solutions}) == 1,
            "original_replacement_is_deterministic_first": record.replacement_plan is first,
            "full_coverage_ranked": view.status
            == core.SpatialPlanReliabilityRankingStatus.RANKED,
            "ranked_replacement_prefers_empirically_stronger_second": view.ranked_replacement_plan is second,
            "ranked_view_changes_only_interpretation": view.ranking_changed,
            "original_replan_record_unchanged": record.replacement_plan is first,
            "rank_view_read_only": before == after,
            "rank_view_nonexperience": not view.is_experience,
            "rank_view_nonevidence": not view.is_evidence,
            "rank_view_nonexecution": not view.was_executed,
            "not_joint_success_probability": not view.is_joint_success_probability,
            "recovery_policy_still_uses_deterministic_replacement": recovery.replacement_plan_id == first.plan_id,
            "specific_capability_true": state["reliability_aware_replacement_plan_ranking"] is True,
            "replanner_semantics_still_false": state["reliability_aware_spatial_replanning"] is False,
            "portable_language_neutral": meta["language_neutral"] is True and meta["python_pickle"] is False,
            "portable_recomputes_same_ranked_view": restored_view.ranked_replacement_plan.plan_id == second.plan_id,
            "no_rank_journal": not hasattr(restored, "spatial_replan_rankings"),
            "descriptor_no_rewrite": descriptor["rewrites_replan_record"] is False,
            "descriptor_no_dispatch": descriptor["automatic_dispatch"] is False,
        }
        failed = [name for name, ok in checks.items() if not ok]
        print(json.dumps({
            "checks": checks,
            "replan_id": record.replan_id,
            "original_replacement_plan_id": first.plan_id,
            "ranked_replacement_plan_id": second.plan_id,
            "ranking_status": view.status.value,
            "reliability_revision": view.reliability_revision,
            "capabilities": {
                "replacement_rank": state["reliability_aware_replacement_plan_ranking"],
                "replanning": state["reliability_aware_spatial_replanning"],
            },
        }, indent=2, sort_keys=True))
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
