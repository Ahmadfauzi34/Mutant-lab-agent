"""Read-only reliability-ranked view of completed spatial replans.

This module reuses the existing equal-depth plan reliability ranker over the
immutable planning result already stored in ``SpatialReplanningRecord``.
Replanning search, provenance, recovery, execution, and reliability learning
remain separate owners.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .spatial import SpatialError
from .spatial_planning import (
    SpatialManipulationPlan,
    SpatialPlanningStatus,
)
from .spatial_replanning import SpatialReplanningRecord
from .spatial_reliability import SpatialManipulationReliabilityStore
from .spatial_plan_ranking import (
    DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES,
    DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z,
    SpatialPlanRankingConflict,
    SpatialPlanReliabilityCandidate,
    SpatialPlanReliabilityRanker,
    SpatialPlanReliabilityRankingStatus,
    SpatialReliabilityRankedPlanningResult,
)

DEFAULT_SPATIAL_REPLAN_RANKING_MIN_SAMPLES = (
    DEFAULT_SPATIAL_PLAN_RANKING_MIN_SAMPLES
)
DEFAULT_SPATIAL_REPLAN_RANKING_WILSON_Z = (
    DEFAULT_SPATIAL_PLAN_RANKING_WILSON_Z
)


class SpatialReplanRankingError(SpatialError):
    pass


class SpatialReplanRankingConflict(SpatialReplanRankingError):
    pass


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
class SpatialReliabilityRankedReplanView:
    replan_record: SpatialReplanningRecord
    planning_ranking: SpatialReliabilityRankedPlanningResult

    def __post_init__(self):
        if not isinstance(self.replan_record, SpatialReplanningRecord):
            raise SpatialReplanRankingError("replan_record tidak valid")
        if not isinstance(
            self.planning_ranking,
            SpatialReliabilityRankedPlanningResult,
        ):
            raise SpatialReplanRankingError("planning_ranking tidak valid")

        base = self.replan_record.planning_result
        ranked_base = self.planning_ranking.planning_result
        if ranked_base is not base and ranked_base != base:
            raise SpatialReplanRankingConflict(
                "ranked planning result tidak cocok replan provenance"
            )

        if self.replan_record.status == SpatialPlanningStatus.FOUND:
            solutions = tuple(base.solutions)
            if not solutions:
                raise SpatialReplanRankingConflict(
                    "FOUND replan tanpa candidate solutions"
                )

            for plan in solutions:
                if not plan.steps:
                    raise SpatialReplanRankingConflict(
                        "FOUND replacement candidate tanpa step"
                    )
                if (
                    plan.steps[0].source_scene_signature
                    != self.replan_record.actual_scene_signature
                ):
                    raise SpatialReplanRankingConflict(
                        "replacement candidate detached dari actual trigger scene"
                    )

            if self.planning_ranking.best_plan not in solutions:
                raise SpatialReplanRankingConflict(
                    "ranked replacement bukan candidate replan"
                )
        else:
            if self.planning_ranking.best_plan is not None:
                raise SpatialReplanRankingConflict(
                    "non-FOUND replan tidak boleh memiliki ranked replacement"
                )
            if (
                self.planning_ranking.status
                != SpatialPlanReliabilityRankingStatus.NOT_APPLICABLE
            ):
                raise SpatialReplanRankingConflict(
                    "non-FOUND replan harus NOT_APPLICABLE"
                )

    @property
    def replan_id(self) -> str:
        return self.replan_record.replan_id

    @property
    def trigger_ticket_id(self) -> str:
        return self.replan_record.trigger_ticket_id

    @property
    def trigger_feedback_id(self) -> str:
        return self.replan_record.trigger_feedback_id

    @property
    def actual_scene_signature(self) -> str:
        return self.replan_record.actual_scene_signature

    @property
    def status(self):
        return self.planning_ranking.status

    @property
    def candidates(self) -> Tuple[SpatialPlanReliabilityCandidate, ...]:
        return self.planning_ranking.candidates

    @property
    def original_replacement_plan(self) -> Optional[SpatialManipulationPlan]:
        return self.replan_record.replacement_plan

    @property
    def ranked_replacement_plan(self) -> Optional[SpatialManipulationPlan]:
        return self.planning_ranking.best_plan

    @property
    def ranking_changed(self) -> bool:
        old = self.original_replacement_plan
        new = self.ranked_replacement_plan
        old_id = None if old is None else old.plan_id
        new_id = None if new is None else new.plan_id
        return old_id != new_id

    @property
    def reliability_revision(self) -> int:
        return self.planning_ranking.reliability_revision

    @property
    def all_candidates_fully_scorable(self) -> bool:
        return self.planning_ranking.all_candidates_fully_scorable

    def is_stale_against(
        self,
        reliability_store: SpatialManipulationReliabilityStore,
    ) -> bool:
        return self.planning_ranking.is_stale_against(reliability_store)

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
    def was_executed(self) -> bool:
        return False

    def to_descriptor(self) -> Dict:
        original = self.original_replacement_plan
        ranked = self.ranked_replacement_plan
        return _json_safe({
            "schema": "agen-spatial-replan-reliability-view-v1",
            "replan_id": self.replan_id,
            "trigger_ticket_id": self.trigger_ticket_id,
            "trigger_feedback_id": self.trigger_feedback_id,
            "actual_scene_signature": self.actual_scene_signature,
            "planning_status": self.replan_record.status.value,
            "ranking_status": self.status.value,
            "original_replacement_plan_id": (
                None if original is None else original.plan_id
            ),
            "ranked_replacement_plan_id": (
                None if ranked is None else ranked.plan_id
            ),
            "ranking_changed": self.ranking_changed,
            "reliability_revision": self.reliability_revision,
            "min_samples": self.planning_ranking.min_samples,
            "wilson_z": self.planning_ranking.wilson_z,
            "ranked_at": self.planning_ranking.ranked_at,
            "all_candidates_fully_scorable": (
                self.all_candidates_fully_scorable
            ),
            "candidates": [
                candidate.to_descriptor()
                for candidate in self.candidates
            ],
            "is_joint_success_probability": False,
            "is_experience": False,
            "is_evidence": False,
            "was_executed": False,
            "rewrites_replan_record": False,
            "creates_execution_ticket": False,
            "automatic_dispatch": False,
        })


class SpatialReplanReliabilityRanker:
    """Read-only adapter over V2.41 plan reliability ranking."""

    @classmethod
    def rank(
        cls,
        replan_record: SpatialReplanningRecord,
        reliability_store: SpatialManipulationReliabilityStore,
        *,
        min_samples: int = DEFAULT_SPATIAL_REPLAN_RANKING_MIN_SAMPLES,
        wilson_z: float = DEFAULT_SPATIAL_REPLAN_RANKING_WILSON_Z,
        ranked_at: int = 0,
    ) -> SpatialReliabilityRankedReplanView:
        if not isinstance(replan_record, SpatialReplanningRecord):
            raise SpatialReplanRankingError("replan_record tidak valid")
        if not isinstance(
            reliability_store,
            SpatialManipulationReliabilityStore,
        ):
            raise SpatialReplanRankingError("reliability_store tidak valid")

        if replan_record.status == SpatialPlanningStatus.FOUND:
            for plan in replan_record.planning_result.solutions:
                if not plan.steps:
                    raise SpatialReplanRankingConflict(
                        "FOUND replacement candidate tanpa step"
                    )
                if (
                    plan.steps[0].source_scene_signature
                    != replan_record.actual_scene_signature
                ):
                    raise SpatialReplanRankingConflict(
                        "replacement candidate detached dari actual trigger scene"
                    )

        try:
            ranked = SpatialPlanReliabilityRanker.rank(
                replan_record.planning_result,
                reliability_store,
                min_samples=min_samples,
                wilson_z=wilson_z,
                ranked_at=ranked_at,
            )
        except SpatialPlanRankingConflict as exc:
            raise SpatialReplanRankingConflict(str(exc)) from exc

        return SpatialReliabilityRankedReplanView(
            replan_record=replan_record,
            planning_ranking=ranked,
        )


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialReplanRankingError,
    SpatialReplanRankingConflict,
    SpatialReliabilityRankedReplanView,
    SpatialReplanReliabilityRanker,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_SPATIAL_REPLAN_RANKING_MIN_SAMPLES",
    "DEFAULT_SPATIAL_REPLAN_RANKING_WILSON_Z",
    "SpatialReplanRankingError",
    "SpatialReplanRankingConflict",
    "SpatialReliabilityRankedReplanView",
    "SpatialReplanReliabilityRanker",
]
