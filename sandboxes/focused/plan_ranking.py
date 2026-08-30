"""V2.41 focused sandbox — reliability-aware equal-depth plan ranking."""
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


def register_source(agent, scene_id="rank-source"):
    agent.register_spatial_scene(
        (obj("A", 0, 0), obj("B", 0, 0)),
        namespace="lab",
        frame_id="world",
        scene_id=scene_id,
        observed_at=0,
    )


def make_candidates(agent):
    if "rank-source" not in agent.spatial_scenes.scenes:
        register_source(agent)
    return agent.plan_spatial_manipulation(
        "rank-source",
        core.SpatialRelationGoal("A", core.SpatialRelationType.ABOVE, "B"),
        (
            core.SpatialManipulationOperator.move("A", 0, 3),
            core.SpatialManipulationOperator.move("A", 0, 4),
        ),
    )


def execute(agent, plan, ordinal, mode="match"):
    t = agent.prepare_spatial_plan_execution_step(plan, 1)
    agent.acknowledge_spatial_execution_dispatch(
        t.ticket_id,
        external_receipt=f"rank:{ordinal}",
        dispatched_at=ordinal * 2 - 1,
    )
    e = t.predicted_scene
    items = list(e.objects)
    if mode == "geometry":
        subject_id = plan.steps[0].operator.subject_id
        emap = e.object_map()
        s = emap[subject_id]
        changed = obj(
            s.object_id,
            s.pose.x + 0.25,
            s.pose.y,
            s.extent.width,
            s.extent.height,
            s.labels,
        )
        items = [changed if x.object_id == subject_id else x for x in items]
    actual = core.make_spatial_scene(
        tuple(items),
        namespace=e.namespace,
        belief_context_id=e.belief_context_id,
        frame_id=e.frame_id,
        scene_id=f"actual-{ordinal}-{mode}",
        observed_at=ordinal * 2,
    )
    return agent.submit_spatial_execution_observation(
        t.ticket_id, actual, register_actual_scene=False
    )


def train(agent, plan, matches, deviations, ordinal):
    for _ in range(matches):
        execute(agent, plan, ordinal, "match")
        ordinal += 1
    for _ in range(deviations):
        execute(agent, plan, ordinal, "geometry")
        ordinal += 1
    return ordinal


def main():
    with tempfile.TemporaryDirectory(prefix="v241_rank_lab_") as td:
        root = Path(td)
        agent = core.IntegratedCognitiveAgent(
            "ranking-lab",
            10,
            10,
            epistemic_archive_path=str(root / "cold.sqlite3"),
        )
        base = make_candidates(agent)
        original_ids = tuple(p.plan_id for p in base.solutions)
        bad, good = base.solutions[0], base.solutions[1]

        before_learning = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        no_data = agent.rank_spatial_planning_result_by_reliability(base)
        ordinal = train(agent, bad, 3, 1, 1)
        ordinal = train(agent, good, 4, 0, ordinal)
        ranked = agent.rank_spatial_planning_result_by_reliability(
            base, ranked_at=500
        )
        descriptor = json.loads(json.dumps(ranked.to_descriptor()))

        after_learning = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
        }

        # Different kinds: train MOVE only; STACK remains truly unscorable.
        partial_scene = core.make_spatial_scene(
            (obj("A",0,0), obj("B",0,0)),
            namespace="partial",
            belief_context_id="ctx-0",
            frame_id="world",
            scene_id="partial-source",
            observed_at=0,
        )
        partial = agent.plan_spatial_manipulation_on_scene(
            partial_scene,
            core.SpatialRelationGoal("A",core.SpatialRelationType.ABOVE,"B"),
            (
                core.SpatialManipulationOperator.move("A",0,3),
                core.SpatialManipulationOperator.stack_above("A","B",gap=0.5),
            ),
        )
        partial_rank = agent.rank_spatial_planning_result_by_reliability(partial)

        # Staleness only after counted actual feedback.
        rev_snapshot = ranked.reliability_revision
        execute(agent, bad, ordinal, "match")
        stale = ranked.is_stale_against(agent.spatial_reliability)

        # Combined API should produce same current best after re-evaluation.
        combined = agent.plan_spatial_manipulation_reliability_aware(
            "rank-source",
            base.goal,
            tuple(p.steps[0].operator for p in base.solutions),
        )

        # Portable state preserves V2.40 reliability; ranking itself is stateless.
        portable = root / "state.db"
        meta = agent.save_portable_state(portable)
        hash_before = file_hash(portable)
        restored = core.IntegratedCognitiveAgent.load_portable_state(portable)
        restored_base = restored.plan_spatial_manipulation(
            "rank-source",
            base.goal,
            tuple(p.steps[0].operator for p in base.solutions),
        )
        restored_rank = restored.rank_spatial_planning_result_by_reliability(
            restored_base
        )
        # Mutate runtime reliability after restore; source DB must remain immutable.
        execute(restored, restored_base.solutions[0], 999, "match")
        hash_after = file_hash(portable)

        probe = subprocess.run(
            [sys.executable, "-c", f'''
import json,sys
from pathlib import Path
sys.path.insert(0,{str(PROJECT_ROOT)!r})
import agen_lab as core
a=core.IntegratedCognitiveAgent.load_portable_state(Path({str(portable)!r}))
ops=(core.SpatialManipulationOperator.move("A",0,3),core.SpatialManipulationOperator.move("A",0,4))
r=a.plan_spatial_manipulation_reliability_aware(
 "rank-source",core.SpatialRelationGoal("A",core.SpatialRelationType.ABOVE,"B"),ops
)
print("RESULT="+json.dumps({{
 "version":core.CORE_VERSION,
 "status":r.status.value,
 "changed":r.ranking_changed,
 "full":r.all_candidates_fully_scorable,
 "joint":r.is_joint_success_probability,
 "tickets":a.spatial_execution_state()["retained_tickets"],
 "ranking_model":a.spatial_state()["spatial_plan_ranking_model"],
 "replan_aware":a.spatial_state()["reliability_aware_spatial_replanning"],
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
        probe_data = json.loads(line.split("=",1)[1]) if line else {}

        scores = {c.plan_id: c.conservative_score for c in ranked.candidates}
        checks = {
            "core_v2_41": core.CORE_VERSION == "2.42",
            "two_shortest_candidates": len(base.solutions) == 2 and base.shortest_depth == 1,
            "no_data_preserves_order": (
                no_data.status == core.SpatialPlanReliabilityRankingStatus.PRESERVED_INCOMPLETE_COVERAGE
                and no_data.ranked_plan_ids == no_data.original_plan_ids
            ),
            "full_coverage_ranked": ranked.status == core.SpatialPlanReliabilityRankingStatus.RANKED,
            "ranking_changed": ranked.ranking_changed,
            "empirically_better_plan_wins": ranked.best_plan.plan_id == good.plan_id,
            "better_wilson_bottleneck_wins": scores[good.plan_id] > scores[bad.plan_id],
            "shortest_depth_preserved": all(c.plan.step_count == 1 for c in ranked.candidates),
            "all_candidates_full_coverage": ranked.all_candidates_fully_scorable,
            "ranking_not_joint_probability": not ranked.is_joint_success_probability,
            "candidate_scores_not_joint_probability": all(not c.is_joint_success_probability for c in ranked.candidates),
            "ranking_nonexperience": not ranked.is_experience,
            "ranking_nonevidence": not ranked.is_evidence,
            "descriptor_json_safe": descriptor["schema"] == "agen-spatial-plan-reliability-ranking-v1",
            "descriptor_pins_revision": descriptor["reliability_revision"] == rev_snapshot,
            "descriptor_preserves_original_ids": tuple(descriptor["original_plan_ids"]) == original_ids,
            "partial_coverage_preserves_order": (
                partial_rank.status == core.SpatialPlanReliabilityRankingStatus.PRESERVED_INCOMPLETE_COVERAGE
                and partial_rank.original_plan_ids == partial_rank.ranked_plan_ids
            ),
            "ranking_read_only_over_empirical_models": before_learning == after_learning,
            "ranking_timestamp_does_not_move_clock": ranked.ranked_at == 500 and agent.interaction_clock != 500,
            "new_actual_feedback_stales_old_ranking": stale,
            "combined_api_returns_current_ranked_result": combined.status == core.SpatialPlanReliabilityRankingStatus.RANKED,
            "base_planner_order_not_mutated": tuple(p.plan_id for p in base.solutions) == original_ids,
            "ranking_does_not_auto_dispatch": agent.spatial_execution_state()["status_counts"]["dispatched"] == 0,
            "portable_language_neutral": meta["language_neutral"] is True and meta["python_pickle"] is False,
            "portable_restore_preserves_ranking_ability": restored_rank.status == core.SpatialPlanReliabilityRankingStatus.RANKED,
            "portable_source_immutable_after_restored_learning": hash_before == hash_after,
            "fresh_process_can_rank": (
                probe.returncode == 0
                and probe_data.get("version") == "2.42"
                and probe_data.get("status") == "ranked"
                and probe_data.get("full") is True
            ),
            "fresh_process_ranking_not_joint": probe_data.get("joint") is False,
            "capability_model_explicit": probe_data.get("ranking_model") == "V2.41_EQUAL_DEPTH_FULL_COVERAGE_WILSON",
            "reliability_aware_replanning_still_absent": probe_data.get("replan_aware") is False,
        }

        failed = [k for k,v in checks.items() if not v]
        print(json.dumps({
            "checks": checks,
            "original_plan_ids": original_ids,
            "ranked_plan_ids": ranked.ranked_plan_ids,
            "scores": scores,
            "ranking_descriptor": descriptor,
            "fresh_process": probe_data,
        }, indent=2, sort_keys=True))
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
