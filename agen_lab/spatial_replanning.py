"""Deviation-triggered bounded spatial replanning — V2.38.

V2.38 repairs a V2.36 manipulation plan only after V2.37 has closed an
execution ticket with ACTUAL spatial feedback classified as a deviation.

The original plan and feedback remain immutable provenance. Replanning starts
from the actual observed scene, preserves the original spatial goal, uses a
finite caller-supplied V2.35 manipulation-operator catalog, and delegates
search to the deterministic bounded V2.36 planner.

Replanning is explicit/read-only with respect to empirical cognition: it does
not execute the replacement plan, mutate the original plan, update Q/world
models, create Evidence, train patterns, or revise Belief Context.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from .spatial import SpatialError, SpatialSceneCanonicalizer
from .spatial_execution import (
    SpatialExecutionFeedback,
    SpatialExecutionFeedbackStatus,
    SpatialExecutionTicket,
    SpatialExecutionTicketStatus,
)
from .spatial_manipulation import SpatialManipulationOperator
from .spatial_planning import (
    DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
    DEFAULT_SPATIAL_PLAN_MAX_NODES,
    DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
    BoundedSpatialManipulationPlanner,
    SpatialManipulationPlan,
    SpatialManipulationPlanningResult,
    SpatialPlanningStatus,
)


DEFAULT_SPATIAL_REPLAN_RECORD_LIMIT = 256


class SpatialReplanningError(SpatialError):
    pass


class SpatialReplanningConflict(SpatialReplanningError):
    pass


class SpatialReplanningTriggerStatus(Enum):
    EXECUTION_DEVIATION = "execution_deviation"


@dataclass(frozen=True)
class SpatialReplanningRecord:
    replan_id: str
    trigger_status: SpatialReplanningTriggerStatus
    original_plan_id: str
    original_plan_semantic_signature: str
    trigger_ticket_id: str
    trigger_feedback_id: str
    trigger_feedback_status: SpatialExecutionFeedbackStatus
    actual_scene_id: str
    actual_scene_signature: str
    goal_semantic_signature: str
    operator_catalog_signatures: Tuple[str, ...]
    planning_result: SpatialManipulationPlanningResult
    max_depth: int
    max_nodes: int
    max_solutions: int
    requested_at: int

    def __post_init__(self):
        for name in (
            "replan_id",
            "original_plan_id",
            "original_plan_semantic_signature",
            "trigger_ticket_id",
            "trigger_feedback_id",
            "actual_scene_id",
            "actual_scene_signature",
            "goal_semantic_signature",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SpatialReplanningError(
                    f"{name} tidak boleh kosong"
                )
        if not isinstance(
            self.trigger_status,
            SpatialReplanningTriggerStatus,
        ):
            raise SpatialReplanningError(
                "trigger_status tidak valid"
            )
        if not isinstance(
            self.trigger_feedback_status,
            SpatialExecutionFeedbackStatus,
        ):
            raise SpatialReplanningError(
                "trigger_feedback_status tidak valid"
            )
        if (
            self.trigger_feedback_status
            == SpatialExecutionFeedbackStatus.MATCH
        ):
            raise SpatialReplanningError(
                "MATCH bukan deviation trigger"
            )
        if not isinstance(
            self.planning_result,
            SpatialManipulationPlanningResult,
        ):
            raise SpatialReplanningError(
                "planning_result tidak valid"
            )
        signatures = tuple(self.operator_catalog_signatures)
        if not signatures:
            raise SpatialReplanningError(
                "operator catalog signature tidak boleh kosong"
            )
        if tuple(sorted(set(signatures))) != signatures:
            raise SpatialReplanningError(
                "operator catalog signatures harus sorted unique"
            )
        object.__setattr__(
            self,
            "operator_catalog_signatures",
            signatures,
        )
        for name in ("max_depth", "max_nodes", "max_solutions"):
            value = int(getattr(self, name))
            if value <= 0:
                raise SpatialReplanningError(
                    f"{name} harus positif"
                )
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "requested_at",
            int(self.requested_at),
        )

    @property
    def request_key(self) -> Tuple:
        return (
            self.trigger_feedback_id,
            self.operator_catalog_signatures,
            self.max_depth,
            self.max_nodes,
            self.max_solutions,
        )

    @property
    def status(self) -> SpatialPlanningStatus:
        return self.planning_result.status

    @property
    def replacement_plan(self) -> Optional[SpatialManipulationPlan]:
        return self.planning_result.best_plan

    @property
    def replanned(self) -> bool:
        return self.status == SpatialPlanningStatus.FOUND

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False

    @property
    def was_executed(self) -> bool:
        return False


class SpatialReplanningStore:
    """Bounded durable journal of completed replan attempts.

    Every record is terminal at creation time because bounded planning itself
    is synchronous. Oldest records may be evicted to preserve the configured
    operational/durable bound.
    """

    def __init__(
        self,
        *,
        limit: int = DEFAULT_SPATIAL_REPLAN_RECORD_LIMIT,
    ):
        limit = int(limit)
        if limit <= 0:
            raise SpatialReplanningError(
                "replan store limit harus positif"
            )
        self.limit = limit
        self.records: "OrderedDict[str, SpatialReplanningRecord]" = (
            OrderedDict()
        )
        self._replan_counter = 0

    def add(
        self,
        *,
        original_plan: SpatialManipulationPlan,
        ticket: SpatialExecutionTicket,
        feedback: SpatialExecutionFeedback,
        operators: Sequence[SpatialManipulationOperator],
        planning_result: SpatialManipulationPlanningResult,
        max_depth: int,
        max_nodes: int,
        max_solutions: int,
        requested_at: int,
    ) -> SpatialReplanningRecord:
        signatures = tuple(sorted({
            operator.semantic_signature
            for operator in operators
        }))
        if not signatures:
            raise SpatialReplanningError(
                "operator catalog tidak boleh kosong"
            )
        request_key = (
            feedback.feedback_id,
            signatures,
            int(max_depth),
            int(max_nodes),
            int(max_solutions),
        )
        for existing in self.records.values():
            if existing.request_key == request_key:
                if (
                    existing.original_plan_id != original_plan.plan_id
                    or existing.trigger_ticket_id != ticket.ticket_id
                ):
                    raise SpatialReplanningConflict(
                        "idempotent replan request identity conflict"
                    )
                return existing

        self._replan_counter += 1
        replan_id = f"spatial-replan-{self._replan_counter}"
        record = SpatialReplanningRecord(
            replan_id=replan_id,
            trigger_status=(
                SpatialReplanningTriggerStatus.EXECUTION_DEVIATION
            ),
            original_plan_id=original_plan.plan_id,
            original_plan_semantic_signature=(
                original_plan.semantic_signature
            ),
            trigger_ticket_id=ticket.ticket_id,
            trigger_feedback_id=feedback.feedback_id,
            trigger_feedback_status=feedback.status,
            actual_scene_id=feedback.observed_scene.scene_id,
            actual_scene_signature=(
                SpatialSceneCanonicalizer.exact_signature(
                    feedback.observed_scene
                )
            ),
            goal_semantic_signature=(
                original_plan.goal.semantic_signature
            ),
            operator_catalog_signatures=signatures,
            planning_result=planning_result,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
            requested_at=requested_at,
        )
        self.records[replan_id] = record
        while len(self.records) > self.limit:
            self.records.popitem(last=False)
        return record

    def get(self, replan_id: str) -> SpatialReplanningRecord:
        try:
            return self.records[replan_id]
        except KeyError as exc:
            raise KeyError(
                f"Spatial replan record {replan_id} tidak ditemukan"
            ) from exc

    def latest_for_plan(
        self,
        original_plan_id: str,
    ) -> Optional[SpatialReplanningRecord]:
        matches = [
            record
            for record in self.records.values()
            if record.original_plan_id == original_plan_id
        ]
        return matches[-1] if matches else None

    def state(self) -> Dict:
        status_counts = {
            status.value: 0
            for status in SpatialPlanningStatus
        }
        for record in self.records.values():
            status_counts[record.status.value] += 1
        return {
            "limit": self.limit,
            "retained_records": len(self.records),
            "replan_counter": self._replan_counter,
            "status_counts": status_counts,
            "automatic_feedback_trigger": False,
            "physical_execution_performed_by_core": False,
            "q_world_evidence_learning": False,
        }


class DeviationTriggeredSpatialReplanner:

    @staticmethod
    def validate_trigger(
        original_plan: SpatialManipulationPlan,
        ticket: SpatialExecutionTicket,
        feedback: SpatialExecutionFeedback,
    ) -> None:
        if not isinstance(original_plan, SpatialManipulationPlan):
            raise SpatialReplanningError(
                "original_plan tidak valid"
            )
        if not isinstance(ticket, SpatialExecutionTicket):
            raise SpatialReplanningError(
                "ticket tidak valid"
            )
        if not isinstance(feedback, SpatialExecutionFeedback):
            raise SpatialReplanningError(
                "feedback tidak valid"
            )
        if ticket.status != SpatialExecutionTicketStatus.CLOSED:
            raise SpatialReplanningConflict(
                "replanning membutuhkan CLOSED execution ticket"
            )
        if feedback.ticket_id != ticket.ticket_id:
            raise SpatialReplanningConflict(
                "feedback/ticket identity mismatch"
            )
        if feedback.plan_id != original_plan.plan_id:
            raise SpatialReplanningConflict(
                "feedback bukan milik original plan"
            )
        if ticket.plan_id != original_plan.plan_id:
            raise SpatialReplanningConflict(
                "ticket bukan milik original plan"
            )
        if (
            ticket.plan_semantic_signature
            != original_plan.semantic_signature
        ):
            raise SpatialReplanningConflict(
                "original plan semantic signature berubah"
            )
        if feedback.matched:
            raise SpatialReplanningConflict(
                "MATCH feedback tidak memerlukan deviation replanning"
            )

    @classmethod
    def replan(
        cls,
        original_plan: SpatialManipulationPlan,
        ticket: SpatialExecutionTicket,
        feedback: SpatialExecutionFeedback,
        operators: Sequence[SpatialManipulationOperator],
        *,
        max_depth: int = DEFAULT_SPATIAL_PLAN_MAX_DEPTH,
        max_nodes: int = DEFAULT_SPATIAL_PLAN_MAX_NODES,
        max_solutions: int = DEFAULT_SPATIAL_PLAN_MAX_SOLUTIONS,
    ) -> SpatialManipulationPlanningResult:
        cls.validate_trigger(
            original_plan,
            ticket,
            feedback,
        )
        # Search starts strictly from ACTUAL observed scene and preserves the
        # original goal. Operator generation remains external/finite.
        return BoundedSpatialManipulationPlanner.search(
            feedback.observed_scene,
            original_plan.goal,
            operators,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_solutions=max_solutions,
        )


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialReplanningError,
    SpatialReplanningConflict,
    SpatialReplanningTriggerStatus,
    SpatialReplanningRecord,
    SpatialReplanningStore,
    DeviationTriggeredSpatialReplanner,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_SPATIAL_REPLAN_RECORD_LIMIT",
    "SpatialReplanningError",
    "SpatialReplanningConflict",
    "SpatialReplanningTriggerStatus",
    "SpatialReplanningRecord",
    "SpatialReplanningStore",
    "DeviationTriggeredSpatialReplanner",
]
