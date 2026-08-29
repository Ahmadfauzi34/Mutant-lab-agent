"""Deterministic spatial recovery policy + controlled handoff — V2.39.

V2.39 consumes the V2.37 actual execution-feedback boundary and optional V2.38
replanning result, then produces an explicit recovery decision without learning
or physical execution.

The policy is intentionally narrow and deterministic:
- MATCH -> continue the original plan;
- geometry/relation deviation -> request bounded replanning;
- scope/identity/label deviation -> require external intervention;
- FOUND replan -> replacement plan may be handed off explicitly;
- ALREADY_SATISFIED -> no further manipulation is required;
- EXHAUSTED -> abort this bounded recovery attempt;
- LIMIT_REACHED -> require external intervention / different bounds.

A HANDOFF_REPLACEMENT decision is structural eligibility only. It is not a
real-world safety guarantee, execution authorization, or physical actuation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .spatial import SpatialError, SpatialSceneCanonicalizer
from .spatial_execution import (
    SpatialExecutionFeedback,
    SpatialExecutionFeedbackStatus,
    SpatialExecutionTicket,
    SpatialExecutionTicketStatus,
)
from .spatial_planning import (
    SpatialManipulationPlan,
    SpatialPlanningStatus,
)
from .spatial_replanning import SpatialReplanningRecord


DEFAULT_SPATIAL_RECOVERY_RECORD_LIMIT = 256
DEFAULT_SPATIAL_RECOVERY_MAX_HANDOFF_STEPS = 6
SPATIAL_RECOVERY_POLICY_VERSION = "V2.39_DETERMINISTIC_RECOVERY_V1"


class SpatialRecoveryError(SpatialError):
    pass


class SpatialRecoveryConflict(SpatialRecoveryError):
    pass


class SpatialRecoveryAction(Enum):
    CONTINUE_ORIGINAL = "continue_original"
    REQUEST_REPLAN = "request_replan"
    HANDOFF_REPLACEMENT = "handoff_replacement"
    GOAL_SATISFIED = "goal_satisfied"
    ABORT_RECOVERY = "abort_recovery"
    REQUIRE_INTERVENTION = "require_intervention"


class SpatialRecoveryReason(Enum):
    ACTUAL_MATCH = "actual_match"
    GEOMETRY_DEVIATION_REPLAN = "geometry_deviation_replan"
    RELATION_DEVIATION_REPLAN = "relation_deviation_replan"
    SCOPE_OR_IDENTITY_DEVIATION = "scope_or_identity_deviation"
    REPLACEMENT_FOUND = "replacement_found"
    REPLACEMENT_TOO_DEEP = "replacement_too_deep"
    ACTUAL_GOAL_ALREADY_SATISFIED = "actual_goal_already_satisfied"
    BOUNDED_GRAPH_EXHAUSTED = "bounded_graph_exhausted"
    SEARCH_LIMIT_REACHED = "search_limit_reached"


_SCOPE_INTERVENTION_STATUSES = frozenset({
    SpatialExecutionFeedbackStatus.NAMESPACE_MISMATCH,
    SpatialExecutionFeedbackStatus.BELIEF_CONTEXT_MISMATCH,
    SpatialExecutionFeedbackStatus.FRAME_MISMATCH,
    SpatialExecutionFeedbackStatus.OBJECT_SET_MISMATCH,
    SpatialExecutionFeedbackStatus.LABEL_MISMATCH,
})

_REPLANNABLE_DEVIATIONS = frozenset({
    SpatialExecutionFeedbackStatus.GEOMETRY_DEVIATION,
    SpatialExecutionFeedbackStatus.RELATION_DEVIATION,
})


@dataclass
class SpatialRecoveryDecisionRecord:
    recovery_id: str
    policy_version: str
    original_plan_id: str
    original_plan_semantic_signature: str
    trigger_ticket_id: str
    trigger_feedback_id: str
    trigger_feedback_status: SpatialExecutionFeedbackStatus
    actual_scene_id: str
    actual_scene_signature: str
    goal_semantic_signature: str
    replan_id: Optional[str]
    replan_status: Optional[SpatialPlanningStatus]
    replacement_plan_id: Optional[str]
    action: SpatialRecoveryAction
    reason: SpatialRecoveryReason
    max_handoff_steps: int
    evaluated_at: int
    handoff_ticket_id: Optional[str] = None

    def __post_init__(self):
        for name in (
            "recovery_id",
            "policy_version",
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
                raise SpatialRecoveryError(f"{name} tidak boleh kosong")
        if not isinstance(
            self.trigger_feedback_status,
            SpatialExecutionFeedbackStatus,
        ):
            raise SpatialRecoveryError("trigger_feedback_status tidak valid")
        if not isinstance(self.action, SpatialRecoveryAction):
            raise SpatialRecoveryError("recovery action tidak valid")
        if not isinstance(self.reason, SpatialRecoveryReason):
            raise SpatialRecoveryError("recovery reason tidak valid")
        if self.replan_status is not None and not isinstance(
            self.replan_status,
            SpatialPlanningStatus,
        ):
            raise SpatialRecoveryError("replan_status tidak valid")
        self.max_handoff_steps = int(self.max_handoff_steps)
        if self.max_handoff_steps <= 0:
            raise SpatialRecoveryError("max_handoff_steps harus positif")
        self.evaluated_at = int(self.evaluated_at)
        if self.replan_id is None:
            if self.replan_status is not None or self.replacement_plan_id is not None:
                raise SpatialRecoveryError(
                    "replan status/plan tidak boleh ada tanpa replan_id"
                )
        if self.action == SpatialRecoveryAction.HANDOFF_REPLACEMENT:
            if (
                self.replan_id is None
                or self.replan_status != SpatialPlanningStatus.FOUND
                or not self.replacement_plan_id
            ):
                raise SpatialRecoveryError(
                    "HANDOFF_REPLACEMENT membutuhkan FOUND replacement plan"
                )

    def to_descriptor(self) -> Dict:
        return {
            "schema": "agen-spatial-recovery-decision-v1",
            "recovery_id": self.recovery_id,
            "policy_version": self.policy_version,
            "original_plan_id": self.original_plan_id,
            "original_plan_semantic_signature": (
                self.original_plan_semantic_signature
            ),
            "trigger_ticket_id": self.trigger_ticket_id,
            "trigger_feedback_id": self.trigger_feedback_id,
            "trigger_feedback_status": self.trigger_feedback_status.value,
            "actual_scene_id": self.actual_scene_id,
            "actual_scene_signature": self.actual_scene_signature,
            "goal_semantic_signature": self.goal_semantic_signature,
            "replan_id": self.replan_id,
            "replan_status": (
                None if self.replan_status is None else self.replan_status.value
            ),
            "replacement_plan_id": self.replacement_plan_id,
            "action": self.action.value,
            "reason": self.reason.value,
            "max_handoff_steps": self.max_handoff_steps,
            "evaluated_at": self.evaluated_at,
            "handoff_ticket_id": self.handoff_ticket_id,
            "is_experience": False,
            "is_evidence": False,
            "was_executed": False,
            "is_real_world_safety_guarantee": False,
        }

    @property
    def request_key(self) -> Tuple:
        return (
            self.trigger_feedback_id,
            self.replan_id,
            self.policy_version,
            self.max_handoff_steps,
        )

    @property
    def can_prepare_handoff(self) -> bool:
        return self.action == SpatialRecoveryAction.HANDOFF_REPLACEMENT

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False

    @property
    def was_executed(self) -> bool:
        return False

    @property
    def is_real_world_safety_guarantee(self) -> bool:
        return False


class DeterministicSpatialRecoveryPolicy:

    @staticmethod
    def validate_provenance(
        original_plan: SpatialManipulationPlan,
        ticket: SpatialExecutionTicket,
        feedback: SpatialExecutionFeedback,
        replan_record: Optional[SpatialReplanningRecord] = None,
    ) -> None:
        if not isinstance(original_plan, SpatialManipulationPlan):
            raise SpatialRecoveryError("original_plan tidak valid")
        if not isinstance(ticket, SpatialExecutionTicket):
            raise SpatialRecoveryError("ticket tidak valid")
        if not isinstance(feedback, SpatialExecutionFeedback):
            raise SpatialRecoveryError("feedback tidak valid")
        if ticket.status != SpatialExecutionTicketStatus.CLOSED:
            raise SpatialRecoveryConflict(
                "recovery decision membutuhkan CLOSED execution ticket"
            )
        if feedback.ticket_id != ticket.ticket_id:
            raise SpatialRecoveryConflict("feedback/ticket identity mismatch")
        if feedback.plan_id != original_plan.plan_id:
            raise SpatialRecoveryConflict("feedback bukan milik original plan")
        if ticket.plan_id != original_plan.plan_id:
            raise SpatialRecoveryConflict("ticket bukan milik original plan")
        if ticket.plan_semantic_signature != original_plan.semantic_signature:
            raise SpatialRecoveryConflict(
                "original plan semantic signature berubah"
            )
        if replan_record is not None:
            if not isinstance(replan_record, SpatialReplanningRecord):
                raise SpatialRecoveryError("replan_record tidak valid")
            if replan_record.original_plan_id != original_plan.plan_id:
                raise SpatialRecoveryConflict(
                    "replan record bukan milik original plan"
                )
            if replan_record.trigger_ticket_id != ticket.ticket_id:
                raise SpatialRecoveryConflict(
                    "replan record trigger ticket mismatch"
                )
            if replan_record.trigger_feedback_id != feedback.feedback_id:
                raise SpatialRecoveryConflict(
                    "replan record trigger feedback mismatch"
                )
            actual_signature = SpatialSceneCanonicalizer.exact_signature(
                feedback.observed_scene
            )
            if replan_record.actual_scene_signature != actual_signature:
                raise SpatialRecoveryConflict(
                    "replan record actual scene provenance mismatch"
                )
            if (
                replan_record.goal_semantic_signature
                != original_plan.goal.semantic_signature
            ):
                raise SpatialRecoveryConflict(
                    "replan record goal provenance mismatch"
                )

    @classmethod
    def decide(
        cls,
        original_plan: SpatialManipulationPlan,
        ticket: SpatialExecutionTicket,
        feedback: SpatialExecutionFeedback,
        *,
        replan_record: Optional[SpatialReplanningRecord] = None,
        max_handoff_steps: int = DEFAULT_SPATIAL_RECOVERY_MAX_HANDOFF_STEPS,
    ) -> Tuple[SpatialRecoveryAction, SpatialRecoveryReason, Optional[str]]:
        cls.validate_provenance(
            original_plan,
            ticket,
            feedback,
            replan_record,
        )
        max_handoff_steps = int(max_handoff_steps)
        if max_handoff_steps <= 0:
            raise SpatialRecoveryError("max_handoff_steps harus positif")

        if feedback.status == SpatialExecutionFeedbackStatus.MATCH:
            if replan_record is not None:
                raise SpatialRecoveryConflict(
                    "MATCH feedback tidak menerima replan record"
                )
            return (
                SpatialRecoveryAction.CONTINUE_ORIGINAL,
                SpatialRecoveryReason.ACTUAL_MATCH,
                None,
            )

        if feedback.status in _SCOPE_INTERVENTION_STATUSES:
            # Even if a caller manually produced a replan, scope/identity
            # mismatch is not automatically accepted as a geometry repair.
            return (
                SpatialRecoveryAction.REQUIRE_INTERVENTION,
                SpatialRecoveryReason.SCOPE_OR_IDENTITY_DEVIATION,
                None,
            )

        if feedback.status not in _REPLANNABLE_DEVIATIONS:
            raise SpatialRecoveryConflict(
                f"unsupported feedback recovery class: {feedback.status.value}"
            )

        if replan_record is None:
            reason = (
                SpatialRecoveryReason.GEOMETRY_DEVIATION_REPLAN
                if feedback.status
                == SpatialExecutionFeedbackStatus.GEOMETRY_DEVIATION
                else SpatialRecoveryReason.RELATION_DEVIATION_REPLAN
            )
            return (
                SpatialRecoveryAction.REQUEST_REPLAN,
                reason,
                None,
            )

        if replan_record.status == SpatialPlanningStatus.ALREADY_SATISFIED:
            return (
                SpatialRecoveryAction.GOAL_SATISFIED,
                SpatialRecoveryReason.ACTUAL_GOAL_ALREADY_SATISFIED,
                None,
            )

        if replan_record.status == SpatialPlanningStatus.EXHAUSTED:
            return (
                SpatialRecoveryAction.ABORT_RECOVERY,
                SpatialRecoveryReason.BOUNDED_GRAPH_EXHAUSTED,
                None,
            )

        if replan_record.status == SpatialPlanningStatus.LIMIT_REACHED:
            return (
                SpatialRecoveryAction.REQUIRE_INTERVENTION,
                SpatialRecoveryReason.SEARCH_LIMIT_REACHED,
                None,
            )

        if replan_record.status != SpatialPlanningStatus.FOUND:
            raise SpatialRecoveryConflict(
                f"unsupported replan status: {replan_record.status.value}"
            )

        replacement = replan_record.replacement_plan
        if replacement is None:
            raise SpatialRecoveryConflict(
                "FOUND replan tidak memiliki replacement plan"
            )
        if replacement.goal.semantic_signature != original_plan.goal.semantic_signature:
            raise SpatialRecoveryConflict(
                "replacement plan goal berbeda dari original goal"
            )
        actual_signature = SpatialSceneCanonicalizer.exact_signature(
            feedback.observed_scene
        )
        if replacement.steps[0].source_scene_signature != actual_signature:
            raise SpatialRecoveryConflict(
                "replacement plan tidak berakar pada actual feedback scene"
            )
        if replacement.step_count > max_handoff_steps:
            return (
                SpatialRecoveryAction.REQUIRE_INTERVENTION,
                SpatialRecoveryReason.REPLACEMENT_TOO_DEEP,
                replacement.plan_id,
            )
        return (
            SpatialRecoveryAction.HANDOFF_REPLACEMENT,
            SpatialRecoveryReason.REPLACEMENT_FOUND,
            replacement.plan_id,
        )


class SpatialRecoveryStore:
    """Bounded durable journal of deterministic recovery decisions."""

    def __init__(
        self,
        *,
        limit: int = DEFAULT_SPATIAL_RECOVERY_RECORD_LIMIT,
    ):
        limit = int(limit)
        if limit <= 0:
            raise SpatialRecoveryError("recovery store limit harus positif")
        self.limit = limit
        self.records: "OrderedDict[str, SpatialRecoveryDecisionRecord]" = (
            OrderedDict()
        )
        self._recovery_counter = 0

    def add(
        self,
        *,
        original_plan: SpatialManipulationPlan,
        ticket: SpatialExecutionTicket,
        feedback: SpatialExecutionFeedback,
        replan_record: Optional[SpatialReplanningRecord],
        action: SpatialRecoveryAction,
        reason: SpatialRecoveryReason,
        replacement_plan_id: Optional[str],
        max_handoff_steps: int,
        evaluated_at: int,
    ) -> SpatialRecoveryDecisionRecord:
        request_key = (
            feedback.feedback_id,
            None if replan_record is None else replan_record.replan_id,
            SPATIAL_RECOVERY_POLICY_VERSION,
            int(max_handoff_steps),
        )
        for existing in self.records.values():
            if existing.request_key == request_key:
                if (
                    existing.original_plan_id != original_plan.plan_id
                    or existing.trigger_ticket_id != ticket.ticket_id
                ):
                    raise SpatialRecoveryConflict(
                        "idempotent recovery decision identity conflict"
                    )
                return existing

        self._recovery_counter += 1
        recovery_id = f"spatial-recovery-{self._recovery_counter}"
        record = SpatialRecoveryDecisionRecord(
            recovery_id=recovery_id,
            policy_version=SPATIAL_RECOVERY_POLICY_VERSION,
            original_plan_id=original_plan.plan_id,
            original_plan_semantic_signature=original_plan.semantic_signature,
            trigger_ticket_id=ticket.ticket_id,
            trigger_feedback_id=feedback.feedback_id,
            trigger_feedback_status=feedback.status,
            actual_scene_id=feedback.observed_scene.scene_id,
            actual_scene_signature=(
                SpatialSceneCanonicalizer.exact_signature(
                    feedback.observed_scene
                )
            ),
            goal_semantic_signature=original_plan.goal.semantic_signature,
            replan_id=(
                None if replan_record is None else replan_record.replan_id
            ),
            replan_status=(
                None if replan_record is None else replan_record.status
            ),
            replacement_plan_id=replacement_plan_id,
            action=action,
            reason=reason,
            max_handoff_steps=max_handoff_steps,
            evaluated_at=evaluated_at,
        )
        self.records[recovery_id] = record
        while len(self.records) > self.limit:
            self.records.popitem(last=False)
        return record

    def get(self, recovery_id: str) -> SpatialRecoveryDecisionRecord:
        try:
            return self.records[recovery_id]
        except KeyError as exc:
            raise KeyError(
                f"Spatial recovery record {recovery_id} tidak ditemukan"
            ) from exc

    def latest_for_plan(
        self,
        original_plan_id: str,
    ) -> Optional[SpatialRecoveryDecisionRecord]:
        matches = [
            record
            for record in self.records.values()
            if record.original_plan_id == original_plan_id
        ]
        return matches[-1] if matches else None

    def bind_handoff_ticket(
        self,
        recovery_id: str,
        ticket_id: str,
    ) -> SpatialRecoveryDecisionRecord:
        record = self.get(recovery_id)
        if not record.can_prepare_handoff:
            raise SpatialRecoveryConflict(
                "recovery decision tidak eligible untuk replacement handoff"
            )
        if not isinstance(ticket_id, str) or not ticket_id:
            raise SpatialRecoveryError("handoff ticket_id tidak boleh kosong")
        if record.handoff_ticket_id is None:
            record.handoff_ticket_id = ticket_id
            return record
        if record.handoff_ticket_id != ticket_id:
            raise SpatialRecoveryConflict(
                "recovery decision sudah terikat ke handoff ticket berbeda"
            )
        return record

    def state(self) -> Dict:
        action_counts = {
            action.value: 0
            for action in SpatialRecoveryAction
        }
        handoff_bound = 0
        for record in self.records.values():
            action_counts[record.action.value] += 1
            if record.handoff_ticket_id is not None:
                handoff_bound += 1
        return {
            "limit": self.limit,
            "retained_records": len(self.records),
            "recovery_counter": self._recovery_counter,
            "action_counts": action_counts,
            "handoff_bound_records": handoff_bound,
            "policy_version": SPATIAL_RECOVERY_POLICY_VERSION,
            "automatic_feedback_side_effect": False,
            "automatic_replanning": False,
            "automatic_dispatch": False,
            "physical_execution_performed_by_core": False,
            "q_world_evidence_learning": False,
        }


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialRecoveryError,
    SpatialRecoveryConflict,
    SpatialRecoveryAction,
    SpatialRecoveryReason,
    SpatialRecoveryDecisionRecord,
    DeterministicSpatialRecoveryPolicy,
    SpatialRecoveryStore,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_SPATIAL_RECOVERY_RECORD_LIMIT",
    "DEFAULT_SPATIAL_RECOVERY_MAX_HANDOFF_STEPS",
    "SPATIAL_RECOVERY_POLICY_VERSION",
    "SpatialRecoveryError",
    "SpatialRecoveryConflict",
    "SpatialRecoveryAction",
    "SpatialRecoveryReason",
    "SpatialRecoveryDecisionRecord",
    "DeterministicSpatialRecoveryPolicy",
    "SpatialRecoveryStore",
]
