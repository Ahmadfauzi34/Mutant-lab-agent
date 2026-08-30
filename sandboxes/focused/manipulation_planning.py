
"""V2.36 focused sandbox — bounded counterfactual manipulation planning."""
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


def obj(i, x, y, w=1.0, h=1.0):
    return core.SpatialObject2D(
        object_id=i,
        pose=core.SpatialPose2D(x, y),
        extent=core.SpatialExtent2D(w, h),
    )


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="v236_plan_lab_") as td:
        root = Path(td)
        agent = core.IntegratedCognitiveAgent(
            "planning-lab", 10, 10,
            epistemic_archive_path=str(root/"cold.sqlite3"),
        )
        agent.register_spatial_scene(
            (
                obj("BLOCK", -6, 0, 4, 2),
                obj("BOX", 0, 0, 3, 5),
                obj("TABLE", 8, 0, 6, 2),
            ),
            namespace="lab",
            scene_id="source",
        )

        goal = core.SpatialRelationGoal(
            "BLOCK",
            core.SpatialRelationType.INSIDE,
            "BOX",
        )
        rotate = core.SpatialManipulationOperator.rotate("BLOCK", 1)
        place = core.SpatialManipulationOperator.place_inside("BLOCK", "BOX")
        move = core.SpatialManipulationOperator.move("BLOCK", -1, 0)

        empirical_before = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
            "scenes": agent.spatial_state(namespace="lab")[
                "total_scenes_registered"
            ],
        }

        result = agent.plan_spatial_manipulation(
            "source",
            goal,
            (place, rotate, move),
            max_depth=4,
            max_nodes=128,
        )
        plan = result.best_plan

        empirical_after = {
            "q": copy.deepcopy(agent.decision_policy.scoped_counts),
            "world": copy.deepcopy(agent.contextual_world_model._stats),
            "joint": copy.deepcopy(agent.joint_objective_model._groups),
            "evidence": len(agent.all_evidence()),
            "patterns": copy.deepcopy(agent.structural_patterns.patterns),
            "scenes": agent.spatial_state(namespace="lab")[
                "total_scenes_registered"
            ],
        }

        direct_limited = agent.plan_spatial_manipulation(
            "source",
            goal,
            (place, rotate),
            max_depth=1,
        )

        impossible = agent.plan_spatial_manipulation(
            "source",
            goal,
            (place,),
        )

        above_scene = core.make_spatial_scene(
            (
                obj("A", 0, 0),
                obj("B", 0, 0),
            ),
            namespace="multi",
            belief_context_id="ctx-0",
            scene_id="multi",
        )
        above_goal = core.SpatialRelationGoal(
            "A",
            core.SpatialRelationType.ABOVE,
            "B",
        )
        multi = agent.plan_spatial_manipulation_on_scene(
            above_scene,
            above_goal,
            (
                core.SpatialManipulationOperator.move("A", 0, 4),
                core.SpatialManipulationOperator.move("A", 0, 3),
            ),
        )
        multi_reversed = agent.plan_spatial_manipulation_on_scene(
            above_scene,
            above_goal,
            (
                core.SpatialManipulationOperator.move("A", 0, 3),
                core.SpatialManipulationOperator.move("A", 0, 4),
            ),
        )

        already_scene = core.make_spatial_scene(
            (obj("A",0,3), obj("B",0,0)),
            namespace="already",
            belief_context_id="ctx-0",
            scene_id="already",
        )
        already = agent.plan_spatial_manipulation_on_scene(
            already_scene,
            above_goal,
            (core.SpatialManipulationOperator.move("A",0,1),),
        )

        token = agent.spatial_plan_token(plan)
        patterns_before_token = agent.structural_pattern_state()["patterns"]

        goal_descriptor = json.loads(json.dumps(goal.to_descriptor()))
        goal_roundtrip = core.SpatialRelationGoal.from_descriptor(goal_descriptor)

        portable = root/"state.db"
        metadata = agent.save_portable_state(portable)
        hash_before = file_hash(portable)
        restored = core.IntegratedCognitiveAgent.load_portable_state(portable)
        restored_result = restored.plan_spatial_manipulation(
            "source",
            goal,
            (place, rotate),
        )
        restored.register_spatial_scene(
            (obj("TEMP",100,100),),
            namespace="runtime",
            scene_id="runtime",
        )
        hash_after = file_hash(portable)

        probe = subprocess.run(
            [sys.executable, "-c", f"""
import json,sys
from pathlib import Path
sys.path.insert(0,{str(PROJECT_ROOT)!r})
import agen_lab as core
a=core.IntegratedCognitiveAgent.load_portable_state(Path({str(portable)!r}))
goal=core.SpatialRelationGoal("BLOCK",core.SpatialRelationType.INSIDE,"BOX")
r=a.plan_spatial_manipulation(
 "source",goal,
 (
  core.SpatialManipulationOperator.rotate("BLOCK",1),
  core.SpatialManipulationOperator.place_inside("BLOCK","BOX"),
 )
)
p=r.best_plan
print("RESULT="+json.dumps({{
 "version":core.CORE_VERSION,
 "status":r.status.value,
 "depth":r.shortest_depth,
 "ops":[s.operator.kind.value for s in p.steps],
 "executed":r.was_executed,
 "planning":a.spatial_state()["bounded_spatial_manipulation_planning"],
 "physical":a.spatial_state()["physical_manipulation_execution"],
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
            "core_v2_36": core.CORE_VERSION == "2.42",
            "planner_found": result.status == core.SpatialPlanningStatus.FOUND,
            "shortest_depth_two": result.shortest_depth == 2,
            "plan_rotate_then_place": tuple(
                s.operator.kind.value for s in plan.steps
            ) == ("rotate", "place_inside"),
            "final_goal_satisfied": goal.satisfied_by(plan.final_scene),
            "plan_nonexperience": not plan.is_experience,
            "plan_nonexecuted": not plan.was_executed,
            "result_nonexecuted": not result.was_executed,
            "direct_place_infeasible_counted": result.infeasible_edges > 0,
            "depth_limit_explicit": (
                direct_limited.status == core.SpatialPlanningStatus.LIMIT_REACHED
                and "max_depth" in direct_limited.limit_reason
            ),
            "infeasible_catalog_exhausted": (
                impossible.status == core.SpatialPlanningStatus.EXHAUSTED
            ),
            "multiple_shortest_solutions": len(multi.solutions) == 2,
            "deterministic_ranking": tuple(
                p.semantic_signature for p in multi.solutions
            ) == tuple(
                p.semantic_signature for p in multi_reversed.solutions
            ),
            "already_satisfied_no_zero_plan": (
                already.status == core.SpatialPlanningStatus.ALREADY_SATISFIED
                and already.solutions == ()
            ),
            "plan_token_operator_classes": token == (
                ("spatial_manipulation","rotate"),
                ("spatial_manipulation","place_inside"),
            ),
            "plan_token_nonlearning": (
                patterns_before_token
                == agent.structural_pattern_state()["patterns"]
            ),
            "goal_json_roundtrip": goal_roundtrip == goal,
            "planning_does_not_mutate_empirical_state": (
                empirical_before == empirical_after
            ),
            "planning_does_not_register_counterfactual_scenes": (
                empirical_before["scenes"] == empirical_after["scenes"]
            ),
            "portable_language_neutral": (
                metadata["language_neutral"] is True
                and metadata["python_pickle"] is False
            ),
            "restored_state_can_plan": restored_result.shortest_depth == 2,
            "portable_source_immutable": hash_before == hash_after,
            "fresh_process_can_plan": (
                probe.returncode == 0
                and probe_data.get("version") == "2.42"
                and probe_data.get("status") == "found"
                and probe_data.get("depth") == 2
                and probe_data.get("ops") == ["rotate","place_inside"]
            ),
            "planning_capability_explicit": probe_data.get("planning") is True,
            "physical_execution_still_absent": probe_data.get("physical") is False,
        }

        failed = [k for k,v in checks.items() if not v]
        print(json.dumps({
            "checks": checks,
            "plan_id": plan.plan_id,
            "plan_token": token,
            "nodes_expanded": result.nodes_expanded,
            "feasible_edges": result.feasible_edges,
            "infeasible_edges": result.infeasible_edges,
            "duplicate_states_pruned": result.duplicate_states_pruned,
            "goal_descriptor": goal_descriptor,
            "fresh_process": probe_data,
        }, indent=2, sort_keys=True))
        print(f"\nFINAL: {len(checks)-len(failed)}/{len(checks)} PASS")
        if failed:
            raise AssertionError(failed)


if __name__ == "__main__":
    main()
