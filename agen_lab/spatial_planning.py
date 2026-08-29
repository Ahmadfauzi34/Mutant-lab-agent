"""Bounded spatial manipulation planning — V2.36.

Deterministic breadth-first search over a finite caller-supplied catalog of
V2.35 counterfactual manipulation operators.

V2.36 deliberately plans only for explicit spatial-relation goals. Planning is
counterfactual and read-only with respect to empirical cognition: it does not
execute actions, register predicted scenes, update Q/world models, create
Evidence, or train structural patterns.
"""
from __future__ import annotations

import hashlib
import json

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .spatial import (
    SpatialError,
    SpatialGeometry2D,
    SpatialRelationType,
    SpatialScene2D,
    SpatialSceneCanonicalizer,
)
from .spatial_manipulation import (
    CounterfactualSpatialManipulation,
    SpatialManipulationOperator,
    SpatialManipulationSimulator,
)


DEFAULT_SPATIAL_PLAN_MAX_DEPTH = 6
DEFAULT_SPATIAL_PLAN_MAX_NODES = 2048
DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS = 8
MAX_SPATIAL_PLAN_OPERATOR_CATALOG = 256


class SpatialPlanningError(SpatialError):
    pass


class SpatialPlanningStatus(Enum):
    ALREADY_SATISFIED = "already_satisfied"
    FOUND = "found"
    EXHAUSTED = "exhausted"
    LIMIT_REACHED = "limit_reached"


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _signature(prefix: str, payload) -> str:
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return f"{prefix}:sha256:{digest}"


@dataclass(frozen=True)
class SpatialRelationGoal:
    subject_id: str
    relation_type: SpatialRelationType
    object_id: str

    def __post_init__(self):
        if not isinstance(self.subject_id, str) or not self.subject_id:
            raise SpatialPlanningError(
                "goal subject_id tidak boleh kosong"
            )
        if not isinstance(self.object_id, str) or not self.object_id:
            raise SpatialPlanningError(
                "goal object_id tidak boleh kosong"
            )
        if self.subject_id == self.object_id:
            raise SpatialPlanningError(
                "goal subject/object harus berbeda"
            )
        if not isinstance(
            self.relation_type,
            SpatialRelationType,
        ):
            raise SpatialPlanningError(
                "goal relation_type tidak valid"
            )

    def to_descriptor(self) -> Dict:
        return {
            "schema": "agen-spatial-relation-goal-v1",
            "subject_id": self.subject_id,
            "relation_type": self.relation_type.value,
            "object_id": self.object_id,
        }

    @classmethod
    def from_descriptor(
        cls,
        descriptor: Dict,
    ) -> "SpatialRelationGoal":
        if not isinstance(descriptor, dict):
            raise SpatialPlanningError(
                "goal descriptor harus dict"
            )
        if descriptor.get("schema") != "agen-spatial-relation-goal-v1":
            raise SpatialPlanningError(
                "goal descriptor schema tidak didukung"
            )
        try:
            relation_type = SpatialRelationType(
                descriptor.get("relation_type")
            )
        except Exception as exc:
            raise SpatialPlanningError(
                "goal relation_type tidak valid"
            ) from exc
        return cls(
            subject_id=descriptor.get("subject_id"),
            relation_type=relation_type,
            object_id=descriptor.get("object_id"),
        )

    @property
    def semantic_signature(self) -> str:
        return _signature(
            "spatial_relation_goal",
            self.to_descriptor(),
        )

    def satisfied_by(
        self,
        scene: SpatialScene2D,
    ) -> bool:
        if not isinstance(scene, SpatialScene2D):
            raise SpatialPlanningError(
                "goal evaluation membutuhkan SpatialScene2D"
            )
        object_map = scene.object_map()
        subject = object_map.get(self.subject_id)
        object_ = object_map.get(self.object_id)
        if subject is None or object_ is None:
            return False

        relations = SpatialGeometry2D.direct_relation_types(
            subject,
            object_,
        )
        return self.relation_type in relations


@dataclass(frozen=True)
class SpatialManipulationPlanStep:
    step_index: int
    operator: SpatialManipulationOperator
    source_scene_signature: str
    predicted_scene_signature: str
    simulation: CounterfactualSpatialManipulation

    def __post_init__(self):
        if int(self.step_index) != self.step_index or self.step_index <= 0:
            raise SpatialPlanningError(
                "step_index harus integer positif"
            )
        if not isinstance(
            self.operator,
            SpatialManipulationOperator,
        ):
            raise SpatialPlanningError(
                "plan step operator tidak valid"
            )
        if not isinstance(
            self.simulation,
            CounterfactualSpatialManipulation,
        ):
            raise SpatialPlanningError(
                "plan step simulation tidak valid"
            )
        if not self.simulation.feasible:
            raise SpatialPlanningError(
                "plan hanya boleh berisi feasible simulation"
            )
        if self.simulation.predicted_scene is None:
            raise SpatialPlanningError(
                "feasible plan step membutuhkan predicted scene"
            )

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def was_executed(self) -> bool:
        return False


@dataclass(frozen=True)
class SpatialManipulationPlan:
    plan_id: str
    source_scene_id: str
    goal: SpatialRelationGoal
    steps: Tuple[SpatialManipulationPlanStep, ...]
    final_scene: SpatialScene2D

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise SpatialPlanningError(
                "plan_id tidak boleh kosong"
            )
        if not isinstance(self.source_scene_id, str) or not self.source_scene_id:
            raise SpatialPlanningError(
                "source_scene_id tidak boleh kosong"
            )
        if not isinstance(self.goal, SpatialRelationGoal):
            raise SpatialPlanningError(
                "goal tidak valid"
            )
        steps = tuple(self.steps)
        if not steps:
            raise SpatialPlanningError(
                "SpatialManipulationPlan FOUND harus memiliki step"
            )
        if not isinstance(self.final_scene, SpatialScene2D):
            raise SpatialPlanningError(
                "final_scene tidak valid"
            )
        if not self.goal.satisfied_by(self.final_scene):
            raise SpatialPlanningError(
                "final_scene tidak memenuhi goal"
            )
        for index, step in enumerate(steps, start=1):
            if step.step_index != index:
                raise SpatialPlanningError(
                    "plan step indices harus contiguous mulai 1"
                )
        object.__setattr__(self, "steps", steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def operator_signature_sequence(self) -> Tuple[str, ...]:
        return tuple(
            step.operator.semantic_signature
            for step in self.steps
        )

    @property
    def semantic_signature(self) -> str:
        return _signature(
            "spatial_manipulation_plan",
            {
                "source_scene_id": self.source_scene_id,
                "goal": self.goal.semantic_signature,
                "operators": list(
                    self.operator_signature_sequence
                ),
                "final_scene_signature": (
                    SpatialSceneCanonicalizer.exact_signature(
                        self.final_scene
                    )
                ),
            },
        )

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def was_executed(self) -> bool:
        return False


@dataclass(frozen=True)
class SpatialManipulationPlanningResult:
    status: SpatialPlanningStatus
    source_scene_id: str
    goal: SpatialRelationGoal
    solutions: Tuple[SpatialManipulationPlan, ...]
    nodes_expanded: int
    feasible_edges: int
    infeasible_edges: int
    duplicate_states_pruned: int
    max_depth: int
    max_nodes: int
    operator_catalog_size: int
    limit_reason: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.status, SpatialPlanningStatus):
            raise SpatialPlanningError(
                "planning status tidak valid"
            )
        if not isinstance(self.goal, SpatialRelationGoal):
            raise SpatialPlanningError(
                "planning goal tidak valid"
            )
        if self.nodes_expanded < 0:
            raise SpatialPlanningError(
                "nodes_expanded tidak boleh negatif"
            )
        object.__setattr__(
            self,
            "solutions",
            tuple(self.solutions),
        )

        if self.status == SpatialPlanningStatus.FOUND:
            if not self.solutions:
                raise SpatialPlanningError(
                    "FOUND membutuhkan minimal satu solution"
                )
        elif self.status == SpatialPlanningStatus.ALREADY_SATISFIED:
            if self.solutions:
                raise SpatialPlanningError(
                    "ALREADY_SATISFIED tidak membuat synthetic zero-step plan"
                )
        elif self.solutions:
            raise SpatialPlanningError(
                "non-FOUND status tidak boleh memiliki solutions"
            )

    @property
    def best_plan(self) -> Optional[SpatialManipulationPlan]:
        if not self.solutions:
            return None
        return self.solutions[0]

    @property
    def shortest_depth(self) -> Optional[int]:
        if not self.solutions:
            return 0 if self.status == SpatialPlanningStatus.ALREADY_SATISFIED else None
        return self.solutions[0].step_count

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def was_executed(self) -> bool:
        return False


@dataclass(frozen=True)
class _SearchNode:
    scene: SpatialScene2D
    steps: Tuple[SpatialManipulationPlanStep, ...]


class BoundedSpatialManipulationPlanner:
    """Deterministic shortest-depth planner.

    Search semantics:
    - BFS over a finite operator catalog;
    - exact V2.33 scene signatures for state deduplication;
    - V2.35 simulator supplies feasibility/effects;
    - all shortest-depth solutions are collected up to max_solutions;
    - deterministic tie ranking uses operator semantic-signature sequence.
    """

    @staticmethod
    def _normalize_operator_catalog(
        operators: Sequence[SpatialManipulationOperator],
    ) -> Tuple[SpatialManipulationOperator, ...]:
        if not isinstance(operators, (tuple, list)):
            operators = tuple(operators)

        dedup: Dict[str, SpatialManipulationOperator] = {}
        for operator in operators:
            if not isinstance(
                operator,
                SpatialManipulationOperator,
            ):
                raise SpatialPlanningError(
                    "operator catalog harus berisi SpatialManipulationOperator"
                )
            dedup.setdefault(
                operator.semantic_signature,
                operator,
            )

        if not dedup:
            raise SpatialPlanningError(
                "operator catalog tidak boleh kosong"
            )
        if len(dedup) > MAX_SPATIAL_PLAN_OPERATOR_CATALOG:
            raise SpatialPlanningError(
                "operator catalog melewati batas "
                f"{MAX_SPATIAL_PLAN_OPERATOR_CATALOG}"
            )

        return tuple(
            dedup[key]
            for key in sorted(dedup)
        )

    @classmethod
    def search(
        cls,
        scene: SpatialScene2D,
        goal: SpatialRelationGoal,
        operators: Sequence[SpatialManipulationOperator],
        *,
        max_depth: int = DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
        max_nodes: int = DEFAULT_SPATIAL_PLAN_MAX_NODES,
        max_solutions: int = DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
    ) -> SpatialManipulationPlanningResult:
        if not isinstance(scene, SpatialScene2D):
            raise SpatialPlanningError(
                "planner source harus SpatialScene2D"
            )
        if not isinstance(goal, SpatialRelationGoal):
            raise SpatialPlanningError(
                "planner goal harus SpatialRelationGoal"
            )

        max_depth = int(max_depth)
        max_nodes = int(max_nodes)
        max_solutions = int(max_solutions)
        if max_depth <= 0:
            raise SpatialPlanningError(
                "max_depth harus positif"
            )
        if max_nodes <= 0:
            raise SpatialPlanningError(
                "max_nodes harus positif"
            )
        if max_solutions <= 0:
            raise SpatialPlanningError(
                "max_solutions harus positif"
            )

        catalog = cls._normalize_operator_catalog(
            operators
        )

        if goal.satisfied_by(scene):
            return SpatialManipulationPlanningResult(
                status=SpatialPlanningStatus.ALREADY_SATISFIED,
                source_scene_id=scene.scene_id,
                goal=goal,
                solutions=(),
                nodes_expanded=0,
                feasible_edges=0,
                infeasible_edges=0,
                duplicate_states_pruned=0,
                max_depth=max_depth,
                max_nodes=max_nodes,
                operator_catalog_size=len(catalog),
            )

        initial_signature = (
            SpatialSceneCanonicalizer.exact_signature(
                scene
            )
        )
        visited = {initial_signature}
        queue = deque([
            _SearchNode(
                scene=scene,
                steps=(),
            )
        ])

        nodes_expanded = 0
        feasible_edges = 0
        infeasible_edges = 0
        duplicate_states_pruned = 0
        solutions = []
        found_depth = None
        hit_depth_limit = False
        hit_node_limit = False

        while queue:
            node = queue.popleft()
            depth = len(node.steps)

            if found_depth is not None and depth >= found_depth:
                break

            if depth >= max_depth:
                hit_depth_limit = True
                continue

            if nodes_expanded >= max_nodes:
                hit_node_limit = True
                break

            nodes_expanded += 1

            source_signature = (
                SpatialSceneCanonicalizer.exact_signature(
                    node.scene
                )
            )

            for operator in catalog:
                simulation = SpatialManipulationSimulator.simulate(
                    node.scene,
                    operator,
                )

                if not simulation.feasible:
                    infeasible_edges += 1
                    continue

                feasible_edges += 1
                predicted_scene = simulation.predicted_scene
                assert predicted_scene is not None

                predicted_signature = (
                    SpatialSceneCanonicalizer.exact_signature(
                        predicted_scene
                    )
                )

                step = SpatialManipulationPlanStep(
                    step_index=depth + 1,
                    operator=operator,
                    source_scene_signature=source_signature,
                    predicted_scene_signature=predicted_signature,
                    simulation=simulation,
                )
                next_steps = node.steps + (step,)

                if goal.satisfied_by(predicted_scene):
                    plan = SpatialManipulationPlan(
                        plan_id=_signature(
                            "spatial_plan",
                            {
                                "source_scene_id": scene.scene_id,
                                "goal": goal.semantic_signature,
                                "operators": [
                                    item.operator.semantic_signature
                                    for item in next_steps
                                ],
                                "final_scene_signature": (
                                    predicted_signature
                                ),
                            },
                        ),
                        source_scene_id=scene.scene_id,
                        goal=goal,
                        steps=next_steps,
                        final_scene=predicted_scene,
                    )
                    if found_depth is None:
                        found_depth = len(next_steps)
                    if len(next_steps) == found_depth:
                        solutions.append(plan)
                        solutions = sorted(
                            solutions,
                            key=lambda item: (
                                item.step_count,
                                item.operator_signature_sequence,
                                item.semantic_signature,
                            ),
                        )[:max_solutions]
                    continue

                if predicted_signature in visited:
                    duplicate_states_pruned += 1
                    continue

                if len(visited) >= max_nodes:
                    hit_node_limit = True
                    continue

                visited.add(predicted_signature)

                if depth + 1 >= max_depth:
                    hit_depth_limit = True
                    continue

                queue.append(
                    _SearchNode(
                        scene=predicted_scene,
                        steps=next_steps,
                    )
                )

        if solutions:
            ranked = tuple(solutions)
            return SpatialManipulationPlanningResult(
                status=SpatialPlanningStatus.FOUND,
                source_scene_id=scene.scene_id,
                goal=goal,
                solutions=ranked,
                nodes_expanded=nodes_expanded,
                feasible_edges=feasible_edges,
                infeasible_edges=infeasible_edges,
                duplicate_states_pruned=duplicate_states_pruned,
                max_depth=max_depth,
                max_nodes=max_nodes,
                operator_catalog_size=len(catalog),
            )

        if hit_node_limit or hit_depth_limit:
            reasons = []
            if hit_node_limit:
                reasons.append("max_nodes")
            if hit_depth_limit:
                reasons.append("max_depth")
            return SpatialManipulationPlanningResult(
                status=SpatialPlanningStatus.LIMIT_REACHED,
                source_scene_id=scene.scene_id,
                goal=goal,
                solutions=(),
                nodes_expanded=nodes_expanded,
                feasible_edges=feasible_edges,
                infeasible_edges=infeasible_edges,
                duplicate_states_pruned=duplicate_states_pruned,
                max_depth=max_depth,
                max_nodes=max_nodes,
                operator_catalog_size=len(catalog),
                limit_reason="+".join(reasons),
            )

        return SpatialManipulationPlanningResult(
            status=SpatialPlanningStatus.EXHAUSTED,
            source_scene_id=scene.scene_id,
            goal=goal,
            solutions=(),
            nodes_expanded=nodes_expanded,
            feasible_edges=feasible_edges,
            infeasible_edges=infeasible_edges,
            duplicate_states_pruned=duplicate_states_pruned,
            max_depth=max_depth,
            max_nodes=max_nodes,
            operator_catalog_size=len(catalog),
        )


def spatial_plan_token(
    plan: SpatialManipulationPlan,
) -> Tuple[Tuple[str, str], ...]:
    """Read-only operator-class sequence bridge for pattern adapters."""
    if not isinstance(plan, SpatialManipulationPlan):
        raise SpatialPlanningError(
            "plan harus SpatialManipulationPlan"
        )
    return tuple(
        (
            "spatial_manipulation",
            step.operator.kind.value,
        )
        for step in plan.steps
    )


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialPlanningError,
    SpatialPlanningStatus,
    SpatialRelationGoal,
    SpatialManipulationPlanStep,
    SpatialManipulationPlan,
    SpatialManipulationPlanningResult,
    BoundedSpatialManipulationPlanner,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_SPATIAL_PLAN_MAX_DEPTH",
    "DEFAULT_SPATIAL_PLAN_MAX_NODES",
    "DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS",
    "MAX_SPATIAL_PLAN_OPERATOR_CATALOG",
    "SpatialPlanningError",
    "SpatialPlanningStatus",
    "SpatialRelationGoal",
    "SpatialManipulationPlanStep",
    "SpatialManipulationPlan",
    "SpatialManipulationPlanningResult",
    "BoundedSpatialManipulationPlanner",
    "spatial_plan_token",
]
