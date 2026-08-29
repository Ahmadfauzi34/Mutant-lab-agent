"""Empirical spatial manipulation reliability — V2.40.

Reliability is learned only from CLOSED V2.37 execution feedback backed by an
actual submitted SpatialScene2D observation.

Counted feedback:
- MATCH -> positive model-fidelity sample;
- GEOMETRY_DEVIATION / RELATION_DEVIATION -> negative model-fidelity sample.

Scope/context/frame/object/label mismatches are journaled but excluded from
operator reliability because they do not isolate manipulation-model fidelity.

The subsystem maintains two non-independent aggregate views over the SAME
actual feedback events:
- exact operator signature;
- operator-kind fallback.

Recovery confidence is a read-only assessment over these aggregates. It never
creates Evidence, Q/world-model samples, physical execution, or automatic
replanning/dispatch.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .spatial import SpatialError, SpatialScene2D
from .spatial_execution import (
    SpatialExecutionFeedback,
    SpatialExecutionFeedbackStatus,
    SpatialExecutionTicket,
    SpatialExecutionTicketStatus,
)
from .spatial_manipulation import SpatialManipulationKind, SpatialManipulationOperator
from .spatial_planning import SpatialManipulationPlan
from .spatial_recovery import (
    SpatialRecoveryAction,
    SpatialRecoveryDecisionRecord,
)


DEFAULT_SPATIAL_RELIABILITY_STAT_LIMIT = 2048
DEFAULT_SPATIAL_RELIABILITY_UPDATE_LIMIT = 2048
DEFAULT_SPATIAL_RELIABILITY_ASSESSMENT_LIMIT = 512
DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES = 4
DEFAULT_SPATIAL_RELIABILITY_WILSON_Z = 1.96
DEFAULT_SPATIAL_RECOVERY_RELIABILITY_THRESHOLD = 0.50


class SpatialReliabilityError(SpatialError):
    pass


class SpatialReliabilityConflict(SpatialReliabilityError):
    pass


class SpatialReliabilityGateBlocked(SpatialReliabilityError):
    pass


class SpatialReliabilityAggregationLevel(Enum):
    EXACT_OPERATOR = "exact_operator"
    OPERATOR_KIND = "operator_kind"
    UNKNOWN = "unknown"


class SpatialReliabilityUpdateDisposition(Enum):
    COUNTED_MATCH = "counted_match"
    COUNTED_DEVIATION = "counted_deviation"
    EXCLUDED_NONCOMPARABLE = "excluded_noncomparable"


class SpatialRecoveryReliabilityStatus(Enum):
    TRUSTED = "trusted"
    INSUFFICIENT_DATA = "insufficient_data"
    PARTIAL_COVERAGE = "partial_coverage"
    BELOW_THRESHOLD = "below_threshold"
    NOT_APPLICABLE = "not_applicable"


_COMPARABLE_DEVIATIONS = frozenset({
    SpatialExecutionFeedbackStatus.GEOMETRY_DEVIATION,
    SpatialExecutionFeedbackStatus.RELATION_DEVIATION,
})


def _finite_nonnegative(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SpatialReliabilityError(f"{name} harus finite dan non-negative")
    return 0.0 if result == 0.0 else result


def _wilson_lower_bound(matches: int, total: int, z: float) -> Optional[float]:
    if total <= 0:
        return None
    z = _finite_nonnegative(z, "wilson_z")
    p = matches / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    margin = z * math.sqrt(
        (p * (1.0 - p) / total) + (z2 / (4.0 * total * total))
    )
    value = (centre - margin) / denom
    return min(1.0, max(0.0, value))


@dataclass
class SpatialReliabilityAccumulator:
    aggregation_level: SpatialReliabilityAggregationLevel
    namespace: str
    belief_context_id: str
    frame_id: str
    operator_key: str
    operator_kind: SpatialManipulationKind
    sample_count: int = 0
    match_count: int = 0
    deviation_count: int = 0
    first_observed_at: Optional[int] = None
    last_observed_at: Optional[int] = None

    def __post_init__(self):
        if self.aggregation_level not in (
            SpatialReliabilityAggregationLevel.EXACT_OPERATOR,
            SpatialReliabilityAggregationLevel.OPERATOR_KIND,
        ):
            raise SpatialReliabilityError("accumulator aggregation level tidak valid")
        for name in ("namespace", "belief_context_id", "frame_id", "operator_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SpatialReliabilityError(f"{name} tidak boleh kosong")
        if not isinstance(self.operator_kind, SpatialManipulationKind):
            raise SpatialReliabilityError("operator_kind tidak valid")
        self.sample_count = int(self.sample_count)
        self.match_count = int(self.match_count)
        self.deviation_count = int(self.deviation_count)
        if min(self.sample_count, self.match_count, self.deviation_count) < 0:
            raise SpatialReliabilityError("reliability counts tidak boleh negatif")
        if self.match_count + self.deviation_count != self.sample_count:
            raise SpatialReliabilityError("reliability counts tidak konsisten")

    def observe(self, *, matched: bool, observed_at: int) -> None:
        observed_at = int(observed_at)
        self.sample_count += 1
        if matched:
            self.match_count += 1
        else:
            self.deviation_count += 1
        if self.first_observed_at is None:
            self.first_observed_at = observed_at
        self.last_observed_at = observed_at


@dataclass(frozen=True)
class SpatialReliabilityUpdate:
    feedback_id: str
    ticket_id: str
    operator_semantic_signature: str
    operator_kind: SpatialManipulationKind
    namespace: str
    belief_context_id: str
    frame_id: str
    feedback_status: SpatialExecutionFeedbackStatus
    disposition: SpatialReliabilityUpdateDisposition
    observed_at: int
    reliability_revision: int

    def __post_init__(self):
        for name in (
            "feedback_id",
            "ticket_id",
            "operator_semantic_signature",
            "namespace",
            "belief_context_id",
            "frame_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SpatialReliabilityError(f"{name} tidak boleh kosong")
        if not isinstance(self.operator_kind, SpatialManipulationKind):
            raise SpatialReliabilityError("operator_kind tidak valid")
        if not isinstance(self.feedback_status, SpatialExecutionFeedbackStatus):
            raise SpatialReliabilityError("feedback_status tidak valid")
        if not isinstance(self.disposition, SpatialReliabilityUpdateDisposition):
            raise SpatialReliabilityError("disposition tidak valid")
        object.__setattr__(self, "observed_at", int(self.observed_at))
        object.__setattr__(self, "reliability_revision", int(self.reliability_revision))

    @property
    def counted(self) -> bool:
        return self.disposition in (
            SpatialReliabilityUpdateDisposition.COUNTED_MATCH,
            SpatialReliabilityUpdateDisposition.COUNTED_DEVIATION,
        )

    @property
    def matched(self) -> Optional[bool]:
        if self.disposition == SpatialReliabilityUpdateDisposition.COUNTED_MATCH:
            return True
        if self.disposition == SpatialReliabilityUpdateDisposition.COUNTED_DEVIATION:
            return False
        return None

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False


@dataclass(frozen=True)
class SpatialReliabilityEstimate:
    aggregation_level: SpatialReliabilityAggregationLevel
    namespace: str
    belief_context_id: str
    frame_id: str
    operator_key: str
    operator_kind: SpatialManipulationKind
    sample_count: int
    match_count: int
    deviation_count: int
    empirical_match_rate: Optional[float]
    posterior_mean: Optional[float]
    posterior_stddev: Optional[float]
    wilson_lower_bound: Optional[float]
    min_samples: int
    scorable: bool

    @classmethod
    def unknown(
        cls,
        *,
        scene: SpatialScene2D,
        operator: SpatialManipulationOperator,
        min_samples: int,
    ) -> "SpatialReliabilityEstimate":
        return cls(
            aggregation_level=SpatialReliabilityAggregationLevel.UNKNOWN,
            namespace=scene.namespace,
            belief_context_id=scene.belief_context_id,
            frame_id=scene.frame_id,
            operator_key=operator.semantic_signature,
            operator_kind=operator.kind,
            sample_count=0,
            match_count=0,
            deviation_count=0,
            empirical_match_rate=None,
            posterior_mean=None,
            posterior_stddev=None,
            wilson_lower_bound=None,
            min_samples=int(min_samples),
            scorable=False,
        )

    @classmethod
    def from_accumulator(
        cls,
        accumulator: SpatialReliabilityAccumulator,
        *,
        min_samples: int,
        wilson_z: float,
    ) -> "SpatialReliabilityEstimate":
        n = accumulator.sample_count
        matches = accumulator.match_count
        if n > 0:
            rate = matches / n
            alpha = 1.0 + matches
            beta = 1.0 + accumulator.deviation_count
            denom = alpha + beta
            posterior_mean = alpha / denom
            posterior_var = (
                alpha * beta
                / (denom * denom * (denom + 1.0))
            )
            posterior_stddev = math.sqrt(posterior_var)
            lower = _wilson_lower_bound(matches, n, wilson_z)
        else:
            rate = None
            posterior_mean = None
            posterior_stddev = None
            lower = None
        return cls(
            aggregation_level=accumulator.aggregation_level,
            namespace=accumulator.namespace,
            belief_context_id=accumulator.belief_context_id,
            frame_id=accumulator.frame_id,
            operator_key=accumulator.operator_key,
            operator_kind=accumulator.operator_kind,
            sample_count=n,
            match_count=matches,
            deviation_count=accumulator.deviation_count,
            empirical_match_rate=rate,
            posterior_mean=posterior_mean,
            posterior_stddev=posterior_stddev,
            wilson_lower_bound=lower,
            min_samples=int(min_samples),
            scorable=(n >= int(min_samples)),
        )


@dataclass(frozen=True)
class SpatialPlanReliabilityStep:
    step_index: int
    operator_semantic_signature: str
    operator_kind: SpatialManipulationKind
    estimate: SpatialReliabilityEstimate

    def __post_init__(self):
        if int(self.step_index) != self.step_index or self.step_index <= 0:
            raise SpatialReliabilityError("step_index harus integer positif")
        if not isinstance(self.operator_semantic_signature, str) or not self.operator_semantic_signature:
            raise SpatialReliabilityError("operator_semantic_signature kosong")
        if not isinstance(self.operator_kind, SpatialManipulationKind):
            raise SpatialReliabilityError("operator_kind tidak valid")
        if not isinstance(self.estimate, SpatialReliabilityEstimate):
            raise SpatialReliabilityError("estimate tidak valid")


@dataclass(frozen=True)
class SpatialRecoveryReliabilityAssessment:
    assessment_id: str
    recovery_id: str
    replacement_plan_id: str
    reliability_revision: int
    status: SpatialRecoveryReliabilityStatus
    steps: Tuple[SpatialPlanReliabilityStep, ...]
    coverage: float
    conservative_score: Optional[float]
    mean_posterior_reliability: Optional[float]
    min_samples: int
    minimum_wilson_lower_bound: float
    evaluated_at: int

    def __post_init__(self):
        for name in ("assessment_id", "recovery_id", "replacement_plan_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SpatialReliabilityError(f"{name} tidak boleh kosong")
        if not isinstance(self.status, SpatialRecoveryReliabilityStatus):
            raise SpatialReliabilityError("assessment status tidak valid")
        steps = tuple(self.steps)
        if not steps:
            raise SpatialReliabilityError("assessment harus memiliki plan steps")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "reliability_revision", int(self.reliability_revision))
        coverage = float(self.coverage)
        if not 0.0 <= coverage <= 1.0:
            raise SpatialReliabilityError("coverage harus dalam [0,1]")
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "min_samples", int(self.min_samples))
        if self.min_samples <= 0:
            raise SpatialReliabilityError("min_samples harus positif")
        threshold = float(self.minimum_wilson_lower_bound)
        if not 0.0 <= threshold <= 1.0:
            raise SpatialReliabilityError("minimum lower bound harus dalam [0,1]")
        object.__setattr__(self, "minimum_wilson_lower_bound", threshold)
        object.__setattr__(self, "evaluated_at", int(self.evaluated_at))

    @property
    def trusted(self) -> bool:
        return self.status == SpatialRecoveryReliabilityStatus.TRUSTED

    @property
    def is_joint_success_probability(self) -> bool:
        return False

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False

    @property
    def request_key(self) -> Tuple:
        return (
            self.recovery_id,
            self.reliability_revision,
            self.min_samples,
            self.minimum_wilson_lower_bound,
        )

    def to_descriptor(self) -> Dict:
        return {
            "schema": "agen-spatial-recovery-reliability-v1",
            "assessment_id": self.assessment_id,
            "recovery_id": self.recovery_id,
            "replacement_plan_id": self.replacement_plan_id,
            "reliability_revision": self.reliability_revision,
            "status": self.status.value,
            "coverage": self.coverage,
            "conservative_score": self.conservative_score,
            "mean_posterior_reliability": self.mean_posterior_reliability,
            "min_samples": self.min_samples,
            "minimum_wilson_lower_bound": self.minimum_wilson_lower_bound,
            "steps": [
                {
                    "step_index": step.step_index,
                    "operator_semantic_signature": step.operator_semantic_signature,
                    "operator_kind": step.operator_kind.value,
                    "aggregation_level": step.estimate.aggregation_level.value,
                    "sample_count": step.estimate.sample_count,
                    "match_count": step.estimate.match_count,
                    "deviation_count": step.estimate.deviation_count,
                    "empirical_match_rate": step.estimate.empirical_match_rate,
                    "posterior_mean": step.estimate.posterior_mean,
                    "posterior_stddev": step.estimate.posterior_stddev,
                    "wilson_lower_bound": step.estimate.wilson_lower_bound,
                    "scorable": step.estimate.scorable,
                }
                for step in self.steps
            ],
            "is_joint_success_probability": False,
            "is_experience": False,
            "is_evidence": False,
        }


class SpatialManipulationReliabilityStore:
    """Bounded durable empirical reliability aggregates and assessments."""

    def __init__(
        self,
        *,
        stat_limit: int = DEFAULT_SPATIAL_RELIABILITY_STAT_LIMIT,
        update_limit: int = DEFAULT_SPATIAL_RELIABILITY_UPDATE_LIMIT,
        assessment_limit: int = DEFAULT_SPATIAL_RELIABILITY_ASSESSMENT_LIMIT,
    ):
        self.stat_limit = int(stat_limit)
        self.update_limit = int(update_limit)
        self.assessment_limit = int(assessment_limit)
        if min(self.stat_limit, self.update_limit, self.assessment_limit) <= 0:
            raise SpatialReliabilityError("reliability limits harus positif")
        self.exact_stats: "OrderedDict[Tuple, SpatialReliabilityAccumulator]" = OrderedDict()
        self.kind_stats: "OrderedDict[Tuple, SpatialReliabilityAccumulator]" = OrderedDict()
        self.updates_by_feedback: "OrderedDict[str, SpatialReliabilityUpdate]" = OrderedDict()
        self.assessments: "OrderedDict[str, SpatialRecoveryReliabilityAssessment]" = OrderedDict()
        self._reliability_revision = 0
        self._assessment_counter = 0

    @staticmethod
    def _scope(scene: SpatialScene2D) -> Tuple[str, str, str]:
        return (
            scene.namespace,
            scene.belief_context_id,
            scene.frame_id,
        )

    @staticmethod
    def _exact_key(scene: SpatialScene2D, operator: SpatialManipulationOperator) -> Tuple:
        return SpatialManipulationReliabilityStore._scope(scene) + (
            operator.semantic_signature,
        )

    @staticmethod
    def _kind_key(scene: SpatialScene2D, operator: SpatialManipulationOperator) -> Tuple:
        return SpatialManipulationReliabilityStore._scope(scene) + (
            operator.kind.value,
        )

    def _bounded_put(self, mapping: OrderedDict, key, value, limit: int) -> None:
        if key in mapping:
            mapping[key] = value
            mapping.move_to_end(key)
        else:
            mapping[key] = value
        while len(mapping) > limit:
            mapping.popitem(last=False)

    def _accumulator(
        self,
        *,
        scene: SpatialScene2D,
        operator: SpatialManipulationOperator,
        level: SpatialReliabilityAggregationLevel,
    ) -> SpatialReliabilityAccumulator:
        namespace, context, frame = self._scope(scene)
        if level == SpatialReliabilityAggregationLevel.EXACT_OPERATOR:
            key = self._exact_key(scene, operator)
            mapping = self.exact_stats
            operator_key = operator.semantic_signature
        elif level == SpatialReliabilityAggregationLevel.OPERATOR_KIND:
            key = self._kind_key(scene, operator)
            mapping = self.kind_stats
            operator_key = operator.kind.value
        else:
            raise SpatialReliabilityError("aggregation level tidak valid")
        existing = mapping.get(key)
        if existing is None:
            existing = SpatialReliabilityAccumulator(
                aggregation_level=level,
                namespace=namespace,
                belief_context_id=context,
                frame_id=frame,
                operator_key=operator_key,
                operator_kind=operator.kind,
            )
            self._bounded_put(mapping, key, existing, self.stat_limit)
        else:
            mapping.move_to_end(key)
        return existing

    def observe_closed_feedback(
        self,
        ticket: SpatialExecutionTicket,
        feedback: SpatialExecutionFeedback,
    ) -> SpatialReliabilityUpdate:
        if not isinstance(ticket, SpatialExecutionTicket):
            raise SpatialReliabilityError("ticket tidak valid")
        if not isinstance(feedback, SpatialExecutionFeedback):
            raise SpatialReliabilityError("feedback tidak valid")
        if ticket.status != SpatialExecutionTicketStatus.CLOSED:
            raise SpatialReliabilityConflict("reliability update membutuhkan CLOSED ticket")
        if feedback.ticket_id != ticket.ticket_id or feedback.plan_id != ticket.plan_id:
            raise SpatialReliabilityConflict("feedback/ticket provenance mismatch")

        existing = self.updates_by_feedback.get(feedback.feedback_id)
        if existing is not None:
            if (
                existing.ticket_id != ticket.ticket_id
                or existing.feedback_status != feedback.status
                or existing.operator_semantic_signature != ticket.operator.semantic_signature
            ):
                raise SpatialReliabilityConflict("feedback reliability idempotence conflict")
            return existing

        if feedback.status == SpatialExecutionFeedbackStatus.MATCH:
            disposition = SpatialReliabilityUpdateDisposition.COUNTED_MATCH
            matched = True
        elif feedback.status in _COMPARABLE_DEVIATIONS:
            disposition = SpatialReliabilityUpdateDisposition.COUNTED_DEVIATION
            matched = False
        else:
            disposition = SpatialReliabilityUpdateDisposition.EXCLUDED_NONCOMPARABLE
            matched = None

        # Scope from the prediction contract. Comparable feedback necessarily
        # agrees on namespace/context/frame; non-comparable feedback is excluded.
        scene = ticket.predicted_scene
        if matched is not None:
            self._accumulator(
                scene=scene,
                operator=ticket.operator,
                level=SpatialReliabilityAggregationLevel.EXACT_OPERATOR,
            ).observe(matched=matched, observed_at=feedback.observed_at)
            self._accumulator(
                scene=scene,
                operator=ticket.operator,
                level=SpatialReliabilityAggregationLevel.OPERATOR_KIND,
            ).observe(matched=matched, observed_at=feedback.observed_at)
            self._reliability_revision += 1

        namespace, context, frame = self._scope(scene)
        update = SpatialReliabilityUpdate(
            feedback_id=feedback.feedback_id,
            ticket_id=ticket.ticket_id,
            operator_semantic_signature=ticket.operator.semantic_signature,
            operator_kind=ticket.operator.kind,
            namespace=namespace,
            belief_context_id=context,
            frame_id=frame,
            feedback_status=feedback.status,
            disposition=disposition,
            observed_at=feedback.observed_at,
            reliability_revision=self._reliability_revision,
        )
        self._bounded_put(
            self.updates_by_feedback,
            feedback.feedback_id,
            update,
            self.update_limit,
        )
        return update

    def estimate_operator(
        self,
        scene: SpatialScene2D,
        operator: SpatialManipulationOperator,
        *,
        min_samples: int = DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES,
        wilson_z: float = DEFAULT_SPATIAL_RELIABILITY_WILSON_Z,
    ) -> SpatialReliabilityEstimate:
        if not isinstance(scene, SpatialScene2D):
            raise SpatialReliabilityError("scene tidak valid")
        if not isinstance(operator, SpatialManipulationOperator):
            raise SpatialReliabilityError("operator tidak valid")
        min_samples = int(min_samples)
        if min_samples <= 0:
            raise SpatialReliabilityError("min_samples harus positif")

        exact = self.exact_stats.get(self._exact_key(scene, operator))
        # Any exact history takes precedence, even when still below the scoring
        # threshold. A sparse known exact mismatch must not be hidden by a much
        # larger operator-kind aggregate.
        if exact is not None:
            return SpatialReliabilityEstimate.from_accumulator(
                exact,
                min_samples=min_samples,
                wilson_z=wilson_z,
            )

        kind = self.kind_stats.get(self._kind_key(scene, operator))
        if kind is not None:
            return SpatialReliabilityEstimate.from_accumulator(
                kind,
                min_samples=min_samples,
                wilson_z=wilson_z,
            )
        return SpatialReliabilityEstimate.unknown(
            scene=scene,
            operator=operator,
            min_samples=min_samples,
        )

    def assess_recovery(
        self,
        decision: SpatialRecoveryDecisionRecord,
        replacement_plan: SpatialManipulationPlan,
        *,
        min_samples: int = DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES,
        minimum_wilson_lower_bound: float = (
            DEFAULT_SPATIAL_RECOVERY_RELIABILITY_THRESHOLD
        ),
        evaluated_at: int = 0,
    ) -> SpatialRecoveryReliabilityAssessment:
        if not isinstance(decision, SpatialRecoveryDecisionRecord):
            raise SpatialReliabilityError("recovery decision tidak valid")
        if not isinstance(replacement_plan, SpatialManipulationPlan):
            raise SpatialReliabilityError("replacement_plan tidak valid")
        if decision.action != SpatialRecoveryAction.HANDOFF_REPLACEMENT:
            raise SpatialReliabilityError(
                "reliability assessment hanya berlaku untuk HANDOFF_REPLACEMENT"
            )
        if decision.replacement_plan_id != replacement_plan.plan_id:
            raise SpatialReliabilityConflict("replacement plan identity mismatch")
        min_samples = int(min_samples)
        if min_samples <= 0:
            raise SpatialReliabilityError("min_samples harus positif")
        threshold = float(minimum_wilson_lower_bound)
        if not 0.0 <= threshold <= 1.0:
            raise SpatialReliabilityError("reliability threshold harus dalam [0,1]")

        request_key = (
            decision.recovery_id,
            self._reliability_revision,
            min_samples,
            threshold,
        )
        for existing in self.assessments.values():
            if existing.request_key == request_key:
                if existing.replacement_plan_id != replacement_plan.plan_id:
                    raise SpatialReliabilityConflict("assessment idempotence conflict")
                return existing

        scene = replacement_plan.steps[0].simulation.predicted_scene
        # The step simulation's predicted scene has the same namespace/context/
        # frame as its source; use plan final scene for a stable scope carrier.
        scope_scene = replacement_plan.final_scene
        steps = []
        for step in replacement_plan.steps:
            estimate = self.estimate_operator(
                scope_scene,
                step.operator,
                min_samples=min_samples,
            )
            steps.append(
                SpatialPlanReliabilityStep(
                    step_index=step.step_index,
                    operator_semantic_signature=step.operator.semantic_signature,
                    operator_kind=step.operator.kind,
                    estimate=estimate,
                )
            )

        scorable = [step for step in steps if step.estimate.scorable]
        coverage = len(scorable) / len(steps)
        if not scorable:
            status = SpatialRecoveryReliabilityStatus.INSUFFICIENT_DATA
            conservative = None
            mean_posterior = None
        else:
            lowers = [
                step.estimate.wilson_lower_bound
                for step in scorable
                if step.estimate.wilson_lower_bound is not None
            ]
            means = [
                step.estimate.posterior_mean
                for step in scorable
                if step.estimate.posterior_mean is not None
            ]
            conservative = min(lowers) if lowers else None
            mean_posterior = (
                sum(means) / len(means)
                if means else None
            )
            if coverage < 1.0:
                status = SpatialRecoveryReliabilityStatus.PARTIAL_COVERAGE
            elif conservative is None or conservative < threshold:
                status = SpatialRecoveryReliabilityStatus.BELOW_THRESHOLD
            else:
                status = SpatialRecoveryReliabilityStatus.TRUSTED

        self._assessment_counter += 1
        assessment = SpatialRecoveryReliabilityAssessment(
            assessment_id=f"spatial-reliability-assessment-{self._assessment_counter}",
            recovery_id=decision.recovery_id,
            replacement_plan_id=replacement_plan.plan_id,
            reliability_revision=self._reliability_revision,
            status=status,
            steps=tuple(steps),
            coverage=coverage,
            conservative_score=conservative,
            mean_posterior_reliability=mean_posterior,
            min_samples=min_samples,
            minimum_wilson_lower_bound=threshold,
            evaluated_at=int(evaluated_at),
        )
        self._bounded_put(
            self.assessments,
            assessment.assessment_id,
            assessment,
            self.assessment_limit,
        )
        return assessment

    def get_assessment(self, assessment_id: str) -> SpatialRecoveryReliabilityAssessment:
        try:
            return self.assessments[assessment_id]
        except KeyError as exc:
            raise KeyError(
                f"Spatial reliability assessment {assessment_id} tidak ditemukan"
            ) from exc

    @property
    def reliability_revision(self) -> int:
        return self._reliability_revision

    def update_for_feedback(self, feedback_id: str) -> Optional[SpatialReliabilityUpdate]:
        return self.updates_by_feedback.get(feedback_id)

    def state(self) -> Dict:
        counted = sum(1 for update in self.updates_by_feedback.values() if update.counted)
        excluded = len(self.updates_by_feedback) - counted
        return {
            "stat_limit": self.stat_limit,
            "update_limit": self.update_limit,
            "assessment_limit": self.assessment_limit,
            "exact_stat_keys": len(self.exact_stats),
            "kind_stat_keys": len(self.kind_stats),
            "retained_updates": len(self.updates_by_feedback),
            "counted_updates": counted,
            "excluded_noncomparable_updates": excluded,
            "retained_assessments": len(self.assessments),
            "reliability_revision": self._reliability_revision,
            "actual_closed_feedback_only": True,
            "simulation_training": False,
            "planning_training": False,
            "q_world_evidence_learning": False,
            "automatic_dispatch": False,
        }


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialReliabilityError,
    SpatialReliabilityConflict,
    SpatialReliabilityGateBlocked,
    SpatialReliabilityAggregationLevel,
    SpatialReliabilityUpdateDisposition,
    SpatialRecoveryReliabilityStatus,
    SpatialReliabilityAccumulator,
    SpatialReliabilityUpdate,
    SpatialReliabilityEstimate,
    SpatialPlanReliabilityStep,
    SpatialRecoveryReliabilityAssessment,
    SpatialManipulationReliabilityStore,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_SPATIAL_RELIABILITY_STAT_LIMIT",
    "DEFAULT_SPATIAL_RELIABILITY_UPDATE_LIMIT",
    "DEFAULT_SPATIAL_RELIABILITY_ASSESSMENT_LIMIT",
    "DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES",
    "DEFAULT_SPATIAL_RELIABILITY_WILSON_Z",
    "DEFAULT_SPATIAL_RECOVERY_RELIABILITY_THRESHOLD",
    "SpatialReliabilityError",
    "SpatialReliabilityConflict",
    "SpatialReliabilityGateBlocked",
    "SpatialReliabilityAggregationLevel",
    "SpatialReliabilityUpdateDisposition",
    "SpatialRecoveryReliabilityStatus",
    "SpatialReliabilityAccumulator",
    "SpatialReliabilityUpdate",
    "SpatialReliabilityEstimate",
    "SpatialPlanReliabilityStep",
    "SpatialRecoveryReliabilityAssessment",
    "SpatialManipulationReliabilityStore",
]
