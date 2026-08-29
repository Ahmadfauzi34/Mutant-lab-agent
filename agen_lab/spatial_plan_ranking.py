"""Reliability-aware ranking of equal-depth spatial plans — V2.41.

This module is intentionally read-only over:
- V2.36 bounded planning results;
- V2.40 empirical manipulation reliability.

It never changes search feasibility/depth, never learns reliability, never
creates Evidence/Q/world-model samples, and never dispatches execution.

Conservative V2.41 policy:
- reliability may reorder only the shortest-depth candidate set already
  produced by V2.36;
- ALL candidates must have full empirical coverage at the same min-sample
  policy before ranking may change;
- incomplete coverage preserves the original deterministic planner order;
- among fully covered equal-depth candidates, higher bottleneck Wilson lower
  bound wins, then higher mean posterior reliability, then original planner
  order as deterministic tie-break;
- bottleneck score is not a joint plan-success probability.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .spatial import SpatialError
from .spatial_planning import (
    SpatialManipulationPlan,
    SpatialManipulationPlanningResult,
    SpatialPlanningStatus,
)
from .spatial_reliability import (
    DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES,
    DEFAULT_SPATIAL_RELIABILITY_WILSON_Z,
    SpatialManipulationReliabilityStore,
    SpatialPlanReliabilityStep,
    SpatialReliabilityError,
)


DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES = (
    DEFAULT_SPATIAL_RELIABILITY_MIN_SAMPLES
)
DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z = (
    DEFAULT_SPATIAL_RELIABILITY_WILSON_Z
)


class SpatialPlanRankingError(SpatialError):
    pass


class SpatialPlanRankingConflict(SpatialPlanRankingError):
    pass


class SpatialPlanReliabilityRankingStatus(Enum):
    RANKED = "ranked"
    SINGLE_CANDIDATE = "single_candidate"
    PRESERVED_INCOMPLETE_COVERAGE = "preserved_incomplete_coverage"
    NOT_APPLICABLE = "not_applicable"


def _finite_nonnegative(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SpatialPlanRankingError(
            f"{name} harus finite dan non-negative"
        )
    return 0.0 if result == 0.0 else result


def _json_safe(value: Dict) -> Dict:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


@dataclass(frozen=True)
class SpatialPlanReliabilityCandidate:
    plan: SpatialManipulationPlan
    original_rank: int
    reliability_revision: int
    steps: Tuple[SpatialPlanReliabilityStep, ...]
    coverage: float
    conservative_score: Optional[float]
    mean_posterior_reliability: Optional[float]
    minimum_sample_count: int

    def __post_init__(self):
        if not isinstance(self.plan, SpatialManipulationPlan):
            raise SpatialPlanRankingError("candidate plan tidak valid")
        rank = int(self.original_rank)
        if rank <= 0:
            raise SpatialPlanRankingError("original_rank harus positif")
        object.__setattr__(self, "original_rank", rank)
        object.__setattr__(
            self,
            "reliability_revision",
            int(self.reliability_revision),
        )
        steps = tuple(self.steps)
        if len(steps) != self.plan.step_count:
            raise SpatialPlanRankingError(
                "candidate reliability steps tidak cocok dengan plan"
            )
        object.__setattr__(self, "steps", steps)
        coverage = float(self.coverage)
        if not 0.0 <= coverage <= 1.0:
            raise SpatialPlanRankingError("coverage harus dalam [0,1]")
        object.__setattr__(self, "coverage", coverage)
        if self.conservative_score is not None:
            score = float(self.conservative_score)
            if not 0.0 <= score <= 1.0:
                raise SpatialPlanRankingError(
                    "conservative_score harus dalam [0,1]"
                )
            object.__setattr__(self, "conservative_score", score)
        if self.mean_posterior_reliability is not None:
            mean = float(self.mean_posterior_reliability)
            if not 0.0 <= mean <= 1.0:
                raise SpatialPlanRankingError(
                    "mean posterior harus dalam [0,1]"
                )
            object.__setattr__(
                self,
                "mean_posterior_reliability",
                mean,
            )
        minimum = int(self.minimum_sample_count)
        if minimum < 0:
            raise SpatialPlanRankingError(
                "minimum_sample_count tidak boleh negatif"
            )
        object.__setattr__(self, "minimum_sample_count", minimum)

    @property
    def fully_scorable(self) -> bool:
        return self.coverage == 1.0

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id

    @property
    def is_joint_success_probability(self) -> bool:
        return False

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False

    def to_descriptor(self) -> Dict:
        return _json_safe({
            "plan_id": self.plan.plan_id,
            "plan_semantic_signature": self.plan.semantic_signature,
            "original_rank": self.original_rank,
            "step_count": self.plan.step_count,
            "coverage": self.coverage,
            "conservative_score": self.conservative_score,
            "mean_posterior_reliability": self.mean_posterior_reliability,
            "minimum_sample_count": self.minimum_sample_count,
            "reliability_revision": self.reliability_revision,
            "steps": [
                {
                    "step_index": step.step_index,
                    "operator_semantic_signature": (
                        step.operator_semantic_signature
                    ),
                    "operator_kind": step.operator_kind.value,
                    "aggregation_level": (
                        step.estimate.aggregation_level.value
                    ),
                    "sample_count": step.estimate.sample_count,
                    "match_count": step.estimate.match_count,
                    "deviation_count": step.estimate.deviation_count,
                    "wilson_lower_bound": (
                        step.estimate.wilson_lower_bound
                    ),
                    "posterior_mean": step.estimate.posterior_mean,
                    "scorable": step.estimate.scorable,
                }
                for step in self.steps
            ],
            "is_joint_success_probability": False,
            "is_experience": False,
            "is_evidence": False,
        })


@dataclass(frozen=True)
class SpatialReliabilityRankedPlanningResult:
    planning_result: SpatialManipulationPlanningResult
    status: SpatialPlanReliabilityRankingStatus
    candidates: Tuple[SpatialPlanReliabilityCandidate, ...]
    original_plan_ids: Tuple[str, ...]
    ranked_plan_ids: Tuple[str, ...]
    reliability_revision: int
    min_samples: int
    wilson_z: float
    ranked_at: int

    def __post_init__(self):
        if not isinstance(
            self.planning_result,
            SpatialManipulationPlanningResult,
        ):
            raise SpatialPlanRankingError(
                "planning_result tidak valid"
            )
        if not isinstance(
            self.status,
            SpatialPlanReliabilityRankingStatus,
        ):
            raise SpatialPlanRankingError("ranking status tidak valid")
        candidates = tuple(self.candidates)
        object.__setattr__(self, "candidates", candidates)
        original = tuple(self.original_plan_ids)
        ranked = tuple(self.ranked_plan_ids)
        object.__setattr__(self, "original_plan_ids", original)
        object.__setattr__(self, "ranked_plan_ids", ranked)
        object.__setattr__(
            self,
            "reliability_revision",
            int(self.reliability_revision),
        )
        min_samples = int(self.min_samples)
        if min_samples <= 0:
            raise SpatialPlanRankingError("min_samples harus positif")
        object.__setattr__(self, "min_samples", min_samples)
        object.__setattr__(
            self,
            "wilson_z",
            _finite_nonnegative(self.wilson_z, "wilson_z"),
        )
        object.__setattr__(self, "ranked_at", int(self.ranked_at))

        if self.planning_result.status == SpatialPlanningStatus.FOUND:
            expected = tuple(
                plan.plan_id
                for plan in self.planning_result.solutions
            )
            if original != expected:
                raise SpatialPlanRankingConflict(
                    "original plan identity tidak cocok planning result"
                )
            if set(ranked) != set(original) or len(ranked) != len(original):
                raise SpatialPlanRankingConflict(
                    "ranked plan identity harus permutation kandidat asli"
                )
            if tuple(c.plan_id for c in candidates) != ranked:
                raise SpatialPlanRankingConflict(
                    "candidate order tidak cocok ranked_plan_ids"
                )
        elif candidates or original or ranked:
            raise SpatialPlanRankingConflict(
                "non-FOUND planning result tidak boleh memiliki ranked candidates"
            )

    @property
    def best_plan(self) -> Optional[SpatialManipulationPlan]:
        if not self.candidates:
            return None
        return self.candidates[0].plan

    @property
    def ranking_changed(self) -> bool:
        return self.ranked_plan_ids != self.original_plan_ids

    @property
    def all_candidates_fully_scorable(self) -> bool:
        return bool(self.candidates) and all(
            candidate.fully_scorable
            for candidate in self.candidates
        )

    def is_stale_against(
        self,
        reliability_store: SpatialManipulationReliabilityStore,
    ) -> bool:
        if not isinstance(
            reliability_store,
            SpatialManipulationReliabilityStore,
        ):
            raise SpatialPlanRankingError(
                "reliability_store tidak valid"
            )
        return (
            self.reliability_revision
            != reliability_store.reliability_revision
        )

    @property
    def is_joint_success_probability(self) -> bool:
        return False

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False

    def to_descriptor(self) -> Dict:
        return _json_safe({
            "schema": "agen-spatial-plan-reliability-ranking-v1",
            "planning_status": self.planning_result.status.value,
            "ranking_status": self.status.value,
            "source_scene_id": self.planning_result.source_scene_id,
            "goal": self.planning_result.goal.to_descriptor(),
            "reliability_revision": self.reliability_revision,
            "min_samples": self.min_samples,
            "wilson_z": self.wilson_z,
            "ranked_at": self.ranked_at,
            "original_plan_ids": list(self.original_plan_ids),
            "ranked_plan_ids": list(self.ranked_plan_ids),
            "ranking_changed": self.ranking_changed,
            "candidates": [
                candidate.to_descriptor()
                for candidate in self.candidates
            ],
            "is_joint_success_probability": False,
            "is_experience": False,
            "is_evidence": False,
        })


class SpatialPlanReliabilityRanker:
    """Read-only reliability tie-ranker for V2.36 shortest plans."""

    @classmethod
    def assess_candidate(
        cls,
        plan: SpatialManipulationPlan,
        *,
        original_rank: int,
        reliability_store: SpatialManipulationReliabilityStore,
        min_samples: int,
        wilson_z: float,
    ) -> SpatialPlanReliabilityCandidate:
        if not isinstance(plan, SpatialManipulationPlan):
            raise SpatialPlanRankingError("plan tidak valid")
        if not isinstance(
            reliability_store,
            SpatialManipulationReliabilityStore,
        ):
            raise SpatialPlanRankingError(
                "reliability_store tidak valid"
            )

        # V2.35 manipulations preserve namespace/context/frame, so final scene
        # is a stable scope carrier for every operator in the same plan.
        scope_scene = plan.final_scene
        steps = []
        for step in plan.steps:
            estimate = reliability_store.estimate_operator(
                scope_scene,
                step.operator,
                min_samples=min_samples,
                wilson_z=wilson_z,
            )
            steps.append(
                SpatialPlanReliabilityStep(
                    step_index=step.step_index,
                    operator_semantic_signature=(
                        step.operator.semantic_signature
                    ),
                    operator_kind=step.operator.kind,
                    estimate=estimate,
                )
            )

        scorable = [step for step in steps if step.estimate.scorable]
        coverage = len(scorable) / len(steps)
        if len(scorable) == len(steps):
            lowers = [
                step.estimate.wilson_lower_bound
                for step in steps
            ]
            means = [
                step.estimate.posterior_mean
                for step in steps
            ]
            if any(value is None for value in lowers):
                raise SpatialPlanRankingConflict(
                    "scorable step kehilangan Wilson lower bound"
                )
            if any(value is None for value in means):
                raise SpatialPlanRankingConflict(
                    "scorable step kehilangan posterior mean"
                )
            conservative = min(lowers)
            mean_posterior = sum(means) / len(means)
            minimum_sample_count = min(
                step.estimate.sample_count
                for step in steps
            )
        else:
            conservative = None
            mean_posterior = None
            minimum_sample_count = (
                min(
                    step.estimate.sample_count
                    for step in steps
                )
                if steps else 0
            )

        return SpatialPlanReliabilityCandidate(
            plan=plan,
            original_rank=int(original_rank),
            reliability_revision=(
                reliability_store.reliability_revision
            ),
            steps=tuple(steps),
            coverage=coverage,
            conservative_score=conservative,
            mean_posterior_reliability=mean_posterior,
            minimum_sample_count=minimum_sample_count,
        )

    @classmethod
    def rank(
        cls,
        planning_result: SpatialManipulationPlanningResult,
        reliability_store: SpatialManipulationReliabilityStore,
        *,
        min_samples: int = DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES,
        wilson_z: float = DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z,
        ranked_at: int = 0,
    ) -> SpatialReliabilityRankedPlanningResult:
        if not isinstance(
            planning_result,
            SpatialManipulationPlanningResult,
        ):
            raise SpatialPlanRankingError(
                "planning_result tidak valid"
            )
        if not isinstance(
            reliability_store,
            SpatialManipulationReliabilityStore,
        ):
            raise SpatialPlanRankingError(
                "reliability_store tidak valid"
            )
        min_samples = int(min_samples)
        if min_samples <= 0:
            raise SpatialPlanRankingError("min_samples harus positif")
        wilson_z = _finite_nonnegative(wilson_z, "wilson_z")

        if planning_result.status != SpatialPlanningStatus.FOUND:
            return SpatialReliabilityRankedPlanningResult(
                planning_result=planning_result,
                status=SpatialPlanReliabilityRankingStatus.NOT_APPLICABLE,
                candidates=(),
                original_plan_ids=(),
                ranked_plan_ids=(),
                reliability_revision=(
                    reliability_store.reliability_revision
                ),
                min_samples=min_samples,
                wilson_z=wilson_z,
                ranked_at=int(ranked_at),
            )

        solutions = tuple(planning_result.solutions)
        if not solutions:
            raise SpatialPlanRankingConflict(
                "FOUND planning result tanpa solutions"
            )
        depths = {plan.step_count for plan in solutions}
        if len(depths) != 1:
            raise SpatialPlanRankingConflict(
                "V2.41 hanya meranking equal-depth planner candidates"
            )

        candidates = tuple(
            cls.assess_candidate(
                plan,
                original_rank=index,
                reliability_store=reliability_store,
                min_samples=min_samples,
                wilson_z=wilson_z,
            )
            for index, plan in enumerate(solutions, start=1)
        )
        original_ids = tuple(plan.plan_id for plan in solutions)

        if len(candidates) == 1:
            status = SpatialPlanReliabilityRankingStatus.SINGLE_CANDIDATE
            ranked = candidates
        elif not all(candidate.fully_scorable for candidate in candidates):
            # Unknown-vs-known is an exploration policy question. V2.41 does
            # not invent one; preserve the V2.36 deterministic order exactly.
            status = (
                SpatialPlanReliabilityRankingStatus
                .PRESERVED_INCOMPLETE_COVERAGE
            )
            ranked = candidates
        else:
            status = SpatialPlanReliabilityRankingStatus.RANKED
            ranked = tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        -candidate.conservative_score,
                        -candidate.mean_posterior_reliability,
                        candidate.original_rank,
                    ),
                )
            )

        return SpatialReliabilityRankedPlanningResult(
            planning_result=planning_result,
            status=status,
            candidates=ranked,
            original_plan_ids=original_ids,
            ranked_plan_ids=tuple(
                candidate.plan_id
                for candidate in ranked
            ),
            reliability_revision=(
                reliability_store.reliability_revision
            ),
            min_samples=min_samples,
            wilson_z=wilson_z,
            ranked_at=int(ranked_at),
        )


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialPlanRankingError,
    SpatialPlanRankingConflict,
    SpatialPlanReliabilityRankingStatus,
    SpatialPlanReliabilityCandidate,
    SpatialReliabilityRankedPlanningResult,
    SpatialPlanReliabilityRanker,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES",
    "DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z",
    "SpatialPlanRankingError",
    "SpatialPlanRankingConflict",
    "SpatialPlanReliabilityRankingStatus",
    "SpatialPlanReliabilityCandidate",
    "SpatialReliabilityRankedPlanningResult",
    "SpatialPlanReliabilityRanker",
]
