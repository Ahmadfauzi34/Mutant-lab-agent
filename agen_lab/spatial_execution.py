"""Ticketed spatial plan execution feedback boundary — V2.37.

This module separates:
- counterfactual plan prediction;
- external dispatch acknowledgement;
- actual observed spatial outcome;
- prediction-vs-observation comparison.

The cognitive core never claims to actuate the world. A ticket marked
DISPATCHED means an external adapter acknowledged dispatch only. Actual outcome
enters this boundary only when a caller submits a SpatialScene2D observation.

No Q/world-model/Evidence learning or autonomous replanning occurs here.
"""
from __future__ import annotations

import json
import math

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .spatial import (
    SpatialError,
    SpatialObject2D,
    SpatialRelationType,
    SpatialScene2D,
    SpatialSceneCanonicalizer,
)
from .spatial_manipulation import (
    SpatialManipulationOperator,
    SpatialManipulationSimulator,
)
from .spatial_planning import (
    SpatialManipulationPlan,
    SpatialRelationGoal,
)


DEFAULT_SPATIAL_EXECUTION_TICKET_LIMIT = 512
DEFAULT_SPATIAL_EXECUTION_MATCH_TOLERANCE = 1e-9


class SpatialExecutionError(SpatialError):
    pass


class SpatialExecutionConflict(SpatialExecutionError):
    pass


class SpatialExecutionStaleSource(SpatialExecutionError):
    pass


class SpatialExecutionContinuationBlocked(SpatialExecutionError):
    pass


class SpatialExecutionTicketStatus(Enum):
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class SpatialExecutionFeedbackStatus(Enum):
    MATCH = "match"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    BELIEF_CONTEXT_MISMATCH = "belief_context_mismatch"
    FRAME_MISMATCH = "frame_mismatch"
    OBJECT_SET_MISMATCH = "object_set_mismatch"
    LABEL_MISMATCH = "label_mismatch"
    GEOMETRY_DEVIATION = "geometry_deviation"
    RELATION_DEVIATION = "relation_deviation"


def _scene_descriptor(scene: SpatialScene2D) -> Dict:
    if not isinstance(scene, SpatialScene2D):
        raise SpatialExecutionError(
            "scene descriptor membutuhkan SpatialScene2D"
        )
    return {
        "scene_id": scene.scene_id,
        "namespace": scene.namespace,
        "belief_context_id": scene.belief_context_id,
        "frame_id": scene.frame_id,
        "observed_at": scene.observed_at,
        "objects": [
            {
                "object_id": item.object_id,
                "pose": {
                    "x": item.pose.x,
                    "y": item.pose.y,
                },
                "extent": {
                    "width": item.extent.width,
                    "height": item.extent.height,
                },
                "labels": list(item.labels),
            }
            for item in scene.objects
        ],
    }


def _json_safe(value: Dict) -> Dict:
    # Roundtrip through stdlib JSON as an explicit contract check.
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _finite_nonnegative(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise SpatialExecutionError(
            f"{name} harus finite dan non-negative"
        )
    if value == 0.0:
        return 0.0
    return value


@dataclass
class SpatialExecutionTicket:
    ticket_id: str
    plan_id: str
    plan_semantic_signature: str
    step_index: int
    step_count: int
    operator: SpatialManipulationOperator
    goal: SpatialRelationGoal
    source_scene_id: str
    source_scene_signature: str
    plan_source_scene_signature: str
    predicted_scene: SpatialScene2D
    predicted_scene_signature: str
    plan_predicted_scene_signature: str
    prepared_at: int
    status: SpatialExecutionTicketStatus = (
        SpatialExecutionTicketStatus.PREPARED
    )
    dispatched_at: Optional[int] = None
    external_receipt: Optional[str] = None
    cancelled_at: Optional[int] = None
    cancellation_reason: Optional[str] = None

    def __post_init__(self):
        for name in (
            "ticket_id",
            "plan_id",
            "plan_semantic_signature",
            "source_scene_id",
            "source_scene_signature",
            "plan_source_scene_signature",
            "predicted_scene_signature",
            "plan_predicted_scene_signature",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SpatialExecutionError(
                    f"{name} tidak boleh kosong"
                )

        if int(self.step_index) != self.step_index:
            raise SpatialExecutionError(
                "step_index harus integer"
            )
        if int(self.step_count) != self.step_count:
            raise SpatialExecutionError(
                "step_count harus integer"
            )
        if self.step_index < 1 or self.step_index > self.step_count:
            raise SpatialExecutionError(
                "step_index di luar plan"
            )
        if not isinstance(
            self.operator,
            SpatialManipulationOperator,
        ):
            raise SpatialExecutionError(
                "ticket operator tidak valid"
            )
        if not isinstance(self.goal, SpatialRelationGoal):
            raise SpatialExecutionError(
                "ticket goal tidak valid"
            )
        if not isinstance(
            self.predicted_scene,
            SpatialScene2D,
        ):
            raise SpatialExecutionError(
                "ticket predicted_scene tidak valid"
            )
        if not isinstance(
            self.status,
            SpatialExecutionTicketStatus,
        ):
            raise SpatialExecutionError(
                "ticket status tidak valid"
            )
        self.prepared_at = int(self.prepared_at)
        if self.dispatched_at is not None:
            self.dispatched_at = int(self.dispatched_at)
        if self.cancelled_at is not None:
            self.cancelled_at = int(self.cancelled_at)

    def dispatch_descriptor(self) -> Dict:
        return _json_safe({
            "schema": "agen-spatial-execution-ticket-v1",
            "ticket_id": self.ticket_id,
            "plan_id": self.plan_id,
            "plan_semantic_signature": self.plan_semantic_signature,
            "step_index": self.step_index,
            "step_count": self.step_count,
            "operator": self.operator.to_descriptor(),
            "goal": self.goal.to_descriptor(),
            "source_scene_id": self.source_scene_id,
            "source_scene_signature": self.source_scene_signature,
            "predicted_scene_signature": self.predicted_scene_signature,
            "predicted_scene": _scene_descriptor(
                self.predicted_scene
            ),
            "prepared_at": self.prepared_at,
            "status": self.status.value,
        })

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def was_executed(self) -> bool:
        return False

    @property
    def external_dispatch_acknowledged(self) -> bool:
        return self.status in (
            SpatialExecutionTicketStatus.DISPATCHED,
            SpatialExecutionTicketStatus.CLOSED,
        )


@dataclass(frozen=True)
class SpatialExecutionFeedback:
    feedback_id: str
    ticket_id: str
    plan_id: str
    step_index: int
    status: SpatialExecutionFeedbackStatus
    observed_scene: SpatialScene2D
    predicted_scene_signature: str
    observed_scene_signature: str
    exact_signature_equal: bool
    max_position_error: Optional[float]
    max_extent_error: Optional[float]
    relational_signature_equal: Optional[bool]
    goal_predicted_satisfied: bool
    goal_observed_satisfied: bool
    observed_at: int
    tolerance: float
    external_receipt: str

    def __post_init__(self):
        for name in (
            "feedback_id",
            "ticket_id",
            "plan_id",
            "predicted_scene_signature",
            "observed_scene_signature",
            "external_receipt",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SpatialExecutionError(
                    f"{name} tidak boleh kosong"
                )
        if not isinstance(
            self.status,
            SpatialExecutionFeedbackStatus,
        ):
            raise SpatialExecutionError(
                "feedback status tidak valid"
            )
        if not isinstance(
            self.observed_scene,
            SpatialScene2D,
        ):
            raise SpatialExecutionError(
                "observed_scene tidak valid"
            )
        if self.max_position_error is not None:
            object.__setattr__(
                self,
                "max_position_error",
                _finite_nonnegative(
                    self.max_position_error,
                    "max_position_error",
                ),
            )
        if self.max_extent_error is not None:
            object.__setattr__(
                self,
                "max_extent_error",
                _finite_nonnegative(
                    self.max_extent_error,
                    "max_extent_error",
                ),
            )
        object.__setattr__(
            self,
            "tolerance",
            _finite_nonnegative(
                self.tolerance,
                "tolerance",
            ),
        )
        object.__setattr__(
            self,
            "observed_at",
            int(self.observed_at),
        )

    def to_descriptor(self) -> Dict:
        return _json_safe({
            "schema": "agen-spatial-execution-feedback-v1",
            "feedback_id": self.feedback_id,
            "ticket_id": self.ticket_id,
            "plan_id": self.plan_id,
            "step_index": self.step_index,
            "status": self.status.value,
            "observed_scene": _scene_descriptor(
                self.observed_scene
            ),
            "predicted_scene_signature": self.predicted_scene_signature,
            "observed_scene_signature": self.observed_scene_signature,
            "exact_signature_equal": self.exact_signature_equal,
            "max_position_error": self.max_position_error,
            "max_extent_error": self.max_extent_error,
            "relational_signature_equal": self.relational_signature_equal,
            "goal_predicted_satisfied": self.goal_predicted_satisfied,
            "goal_observed_satisfied": self.goal_observed_satisfied,
            "observed_at": self.observed_at,
            "tolerance": self.tolerance,
            "external_receipt": self.external_receipt,
        })

    @property
    def matched(self) -> bool:
        return (
            self.status
            == SpatialExecutionFeedbackStatus.MATCH
        )

    @property
    def can_continue_plan(self) -> bool:
        return self.matched

    @property
    def is_actual_observation(self) -> bool:
        return True

    @property
    def is_evidence(self) -> bool:
        return False

    @property
    def is_q_experience(self) -> bool:
        return False


class SpatialExecutionComparator:

    @staticmethod
    def compare(
        ticket: SpatialExecutionTicket,
        observed_scene: SpatialScene2D,
        *,
        observed_at: int,
        tolerance: float = (
            DEFAULT_SPATIAL_EXECUTION_MATCH_TOLERANCE
        ),
    ) -> SpatialExecutionFeedback:
        if not isinstance(
            ticket,
            SpatialExecutionTicket,
        ):
            raise SpatialExecutionError(
                "compare membutuhkan SpatialExecutionTicket"
            )
        if not isinstance(
            observed_scene,
            SpatialScene2D,
        ):
            raise SpatialExecutionError(
                "compare membutuhkan SpatialScene2D actual"
            )
        if not ticket.external_receipt:
            raise SpatialExecutionError(
                "ticket belum mempunyai external dispatch receipt"
            )

        tolerance = _finite_nonnegative(
            tolerance,
            "tolerance",
        )

        predicted = ticket.predicted_scene
        predicted_signature = (
            SpatialSceneCanonicalizer.exact_signature(
                predicted
            )
        )
        observed_signature = (
            SpatialSceneCanonicalizer.exact_signature(
                observed_scene
            )
        )

        exact_equal = (
            predicted_signature
            == observed_signature
        )

        status = None
        max_position_error = None
        max_extent_error = None
        relational_equal = None

        if predicted.namespace != observed_scene.namespace:
            status = (
                SpatialExecutionFeedbackStatus
                .NAMESPACE_MISMATCH
            )
        elif (
            predicted.belief_context_id
            != observed_scene.belief_context_id
        ):
            status = (
                SpatialExecutionFeedbackStatus
                .BELIEF_CONTEXT_MISMATCH
            )
        elif predicted.frame_id != observed_scene.frame_id:
            status = (
                SpatialExecutionFeedbackStatus
                .FRAME_MISMATCH
            )
        else:
            predicted_map = predicted.object_map()
            observed_map = observed_scene.object_map()
            if set(predicted_map) != set(observed_map):
                status = (
                    SpatialExecutionFeedbackStatus
                    .OBJECT_SET_MISMATCH
                )
            else:
                label_mismatch = any(
                    predicted_map[object_id].labels
                    != observed_map[object_id].labels
                    for object_id in predicted_map
                )
                if label_mismatch:
                    status = (
                        SpatialExecutionFeedbackStatus
                        .LABEL_MISMATCH
                    )
                else:
                    max_position_error = 0.0
                    max_extent_error = 0.0
                    for object_id in predicted_map:
                        expected = predicted_map[object_id]
                        actual = observed_map[object_id]
                        max_position_error = max(
                            max_position_error,
                            abs(
                                expected.pose.x
                                - actual.pose.x
                            ),
                            abs(
                                expected.pose.y
                                - actual.pose.y
                            ),
                        )
                        max_extent_error = max(
                            max_extent_error,
                            abs(
                                expected.extent.width
                                - actual.extent.width
                            ),
                            abs(
                                expected.extent.height
                                - actual.extent.height
                            ),
                        )

                    predicted_rel = (
                        SpatialSceneCanonicalizer
                        .relational_signature(
                            predicted
                        )
                    )
                    observed_rel = (
                        SpatialSceneCanonicalizer
                        .relational_signature(
                            observed_scene
                        )
                    )
                    relational_equal = (
                        predicted_rel == observed_rel
                    )

                    if (
                        max_position_error > tolerance
                        or max_extent_error > tolerance
                    ):
                        status = (
                            SpatialExecutionFeedbackStatus
                            .GEOMETRY_DEVIATION
                        )
                    elif not relational_equal:
                        status = (
                            SpatialExecutionFeedbackStatus
                            .RELATION_DEVIATION
                        )
                    else:
                        status = (
                            SpatialExecutionFeedbackStatus
                            .MATCH
                        )

        return SpatialExecutionFeedback(
            feedback_id=(
                "spatial-feedback:"
                + ticket.ticket_id
            ),
            ticket_id=ticket.ticket_id,
            plan_id=ticket.plan_id,
            step_index=ticket.step_index,
            status=status,
            observed_scene=observed_scene,
            predicted_scene_signature=(
                predicted_signature
            ),
            observed_scene_signature=(
                observed_signature
            ),
            exact_signature_equal=exact_equal,
            max_position_error=max_position_error,
            max_extent_error=max_extent_error,
            relational_signature_equal=(
                relational_equal
            ),
            goal_predicted_satisfied=(
                ticket.goal.satisfied_by(
                    predicted
                )
            ),
            goal_observed_satisfied=(
                ticket.goal.satisfied_by(
                    observed_scene
                )
            ),
            observed_at=observed_at,
            tolerance=tolerance,
            external_receipt=(
                ticket.external_receipt
            ),
        )


class SpatialExecutionStore:
    """Bounded durable execution-ticket/feedback journal.

    Pending PREPARED/DISPATCHED tickets are never silently evicted.
    Old terminal tickets may be evicted with their feedback when capacity is
    needed.
    """

    def __init__(
        self,
        *,
        limit: int = (
            DEFAULT_SPATIAL_EXECUTION_TICKET_LIMIT
        ),
    ):
        limit = int(limit)
        if limit <= 0:
            raise SpatialExecutionError(
                "execution store limit harus positif"
            )
        self.limit = limit
        self.tickets: "OrderedDict[str, SpatialExecutionTicket]" = (
            OrderedDict()
        )
        self.feedback_by_ticket: "OrderedDict[str, SpatialExecutionFeedback]" = (
            OrderedDict()
        )
        self._ticket_counter = 0

    def _ensure_capacity(self):
        if len(self.tickets) < self.limit:
            return

        removable = next(
            (
                ticket_id
                for ticket_id, ticket
                in self.tickets.items()
                if ticket.status in (
                    SpatialExecutionTicketStatus.CLOSED,
                    SpatialExecutionTicketStatus.CANCELLED,
                )
            ),
            None,
        )
        if removable is None:
            raise SpatialExecutionError(
                "execution store penuh oleh pending tickets"
            )

        self.tickets.pop(removable, None)
        self.feedback_by_ticket.pop(
            removable,
            None,
        )

    def issue(
        self,
        *,
        plan: SpatialManipulationPlan,
        step_index: int,
        source_scene: SpatialScene2D,
        predicted_scene: SpatialScene2D,
        prepared_at: int,
    ) -> SpatialExecutionTicket:
        if not isinstance(plan, SpatialManipulationPlan):
            raise SpatialExecutionError(
                "issue membutuhkan SpatialManipulationPlan"
            )
        step_index = int(step_index)
        if (
            step_index < 1
            or step_index > plan.step_count
        ):
            raise SpatialExecutionError(
                "step_index di luar plan"
            )
        step = plan.steps[step_index - 1]

        self._ensure_capacity()
        self._ticket_counter += 1
        ticket_id = (
            f"spatial-exec-{self._ticket_counter}"
        )

        ticket = SpatialExecutionTicket(
            ticket_id=ticket_id,
            plan_id=plan.plan_id,
            plan_semantic_signature=(
                plan.semantic_signature
            ),
            step_index=step_index,
            step_count=plan.step_count,
            operator=step.operator,
            goal=plan.goal,
            source_scene_id=(
                source_scene.scene_id
            ),
            source_scene_signature=(
                SpatialSceneCanonicalizer
                .exact_signature(source_scene)
            ),
            plan_source_scene_signature=(
                step.source_scene_signature
            ),
            predicted_scene=predicted_scene,
            predicted_scene_signature=(
                SpatialSceneCanonicalizer
                .exact_signature(
                    predicted_scene
                )
            ),
            plan_predicted_scene_signature=(
                step.predicted_scene_signature
            ),
            prepared_at=int(prepared_at),
        )
        self.tickets[ticket_id] = ticket
        return ticket

    def get_ticket(
        self,
        ticket_id: str,
    ) -> SpatialExecutionTicket:
        try:
            return self.tickets[ticket_id]
        except KeyError as exc:
            raise KeyError(
                f"Spatial execution ticket {ticket_id} tidak ditemukan"
            ) from exc

    def dispatch(
        self,
        ticket_id: str,
        *,
        external_receipt: str,
        dispatched_at: int,
    ) -> SpatialExecutionTicket:
        ticket = self.get_ticket(ticket_id)
        if (
            not isinstance(external_receipt, str)
            or not external_receipt
        ):
            raise SpatialExecutionError(
                "external_receipt tidak boleh kosong"
            )

        if ticket.status == SpatialExecutionTicketStatus.PREPARED:
            ticket.status = (
                SpatialExecutionTicketStatus.DISPATCHED
            )
            ticket.external_receipt = (
                external_receipt
            )
            ticket.dispatched_at = int(
                dispatched_at
            )
            return ticket

        if ticket.status == SpatialExecutionTicketStatus.DISPATCHED:
            if ticket.external_receipt != external_receipt:
                raise SpatialExecutionConflict(
                    "ticket sudah dispatched dengan receipt berbeda"
                )
            return ticket

        raise SpatialExecutionConflict(
            f"ticket status {ticket.status.value} tidak dapat dispatched"
        )

    def cancel(
        self,
        ticket_id: str,
        *,
        reason: str,
        cancelled_at: int,
    ) -> SpatialExecutionTicket:
        ticket = self.get_ticket(ticket_id)
        if (
            not isinstance(reason, str)
            or not reason
        ):
            raise SpatialExecutionError(
                "cancellation reason tidak boleh kosong"
            )

        if ticket.status == SpatialExecutionTicketStatus.CANCELLED:
            if ticket.cancellation_reason != reason:
                raise SpatialExecutionConflict(
                    "ticket sudah cancelled dengan reason berbeda"
                )
            return ticket

        if ticket.status == SpatialExecutionTicketStatus.CLOSED:
            raise SpatialExecutionConflict(
                "closed ticket tidak dapat cancelled"
            )

        ticket.status = (
            SpatialExecutionTicketStatus.CANCELLED
        )
        ticket.cancelled_at = int(
            cancelled_at
        )
        ticket.cancellation_reason = reason
        return ticket

    def close_with_feedback(
        self,
        ticket_id: str,
        feedback: SpatialExecutionFeedback,
    ) -> SpatialExecutionFeedback:
        ticket = self.get_ticket(ticket_id)
        if not isinstance(
            feedback,
            SpatialExecutionFeedback,
        ):
            raise SpatialExecutionError(
                "feedback tidak valid"
            )
        if feedback.ticket_id != ticket_id:
            raise SpatialExecutionConflict(
                "feedback ticket_id mismatch"
            )
        if ticket.status != SpatialExecutionTicketStatus.DISPATCHED:
            if ticket.status == SpatialExecutionTicketStatus.CLOSED:
                existing = self.feedback_by_ticket.get(
                    ticket_id
                )
                if existing == feedback:
                    return existing
            raise SpatialExecutionConflict(
                "hanya dispatched ticket dapat menerima actual feedback"
            )

        existing = self.feedback_by_ticket.get(
            ticket_id
        )
        if existing is not None:
            if existing == feedback:
                return existing
            raise SpatialExecutionConflict(
                "ticket sudah memiliki feedback berbeda"
            )

        self.feedback_by_ticket[
            ticket_id
        ] = feedback
        ticket.status = (
            SpatialExecutionTicketStatus.CLOSED
        )
        return feedback

    def feedback(
        self,
        ticket_id: str,
    ) -> Optional[SpatialExecutionFeedback]:
        return self.feedback_by_ticket.get(
            ticket_id
        )

    def latest_plan_feedback(
        self,
        plan_id: str,
        step_index: int,
    ) -> Optional[SpatialExecutionFeedback]:
        matches = [
            feedback
            for feedback in self.feedback_by_ticket.values()
            if (
                feedback.plan_id == plan_id
                and feedback.step_index
                == int(step_index)
            )
        ]
        if not matches:
            return None
        return matches[-1]

    def state(self) -> Dict:
        statuses = {
            status.value: 0
            for status
            in SpatialExecutionTicketStatus
        }
        for ticket in self.tickets.values():
            statuses[ticket.status.value] += 1
        return {
            "limit": self.limit,
            "retained_tickets": len(
                self.tickets
            ),
            "retained_feedback": len(
                self.feedback_by_ticket
            ),
            "ticket_counter": (
                self._ticket_counter
            ),
            "status_counts": statuses,
            "physical_execution_performed_by_core": False,
            "q_world_evidence_learning": False,
        }


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialExecutionError,
    SpatialExecutionConflict,
    SpatialExecutionStaleSource,
    SpatialExecutionContinuationBlocked,
    SpatialExecutionTicketStatus,
    SpatialExecutionFeedbackStatus,
    SpatialExecutionTicket,
    SpatialExecutionFeedback,
    SpatialExecutionComparator,
    SpatialExecutionStore,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_SPATIAL_EXECUTION_TICKET_LIMIT",
    "DEFAULT_SPATIAL_EXECUTION_MATCH_TOLERANCE",
    "SpatialExecutionError",
    "SpatialExecutionConflict",
    "SpatialExecutionStaleSource",
    "SpatialExecutionContinuationBlocked",
    "SpatialExecutionTicketStatus",
    "SpatialExecutionFeedbackStatus",
    "SpatialExecutionTicket",
    "SpatialExecutionFeedback",
    "SpatialExecutionComparator",
    "SpatialExecutionStore",
]
