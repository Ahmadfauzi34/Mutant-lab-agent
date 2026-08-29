"""Memory subsystem — physically extracted in modularization M4.

Physical ownership:
- bounded operational memory containers;
- HOT/WARM retention policy and compaction summaries;
- lifecycle compaction orchestration;
- exact SQLite COLD Evidence/Episode archive.

Dependency rule:
No module-level dependency on the compatibility kernel. Record annotations are
postponed. COLD JSON deserializers resolve canonical epistemic types lazily at
call time after the V2.28 kernel has completed loading.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .objectives import OBJECTIVE_COMPONENTS, ObjectiveOutcome


def _canonical_type(name: str):
    import sys
    module = sys.modules.get("agen_kognitif_v2_28")
    if module is None:
        raise RuntimeError(
            "Canonical module agen_kognitif_v2_28 belum tersedia"
        )
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Canonical V2.28 type '{name}' tidak tersedia"
        ) from exc


class BeliefShiftDecisionMemory:
    """
    Bounded audit memory.

    V2.23 intentionally does not add another append-only history. Old detailed
    shift probes are evicted after `limit`; total_seen/evicted remain auditable.
    """
    def __init__(self, limit: int = 2048):
        if limit < 1:
            raise ValueError("belief shift memory limit harus >= 1")
        self.limit = limit
        self._records: List[BeliefShiftDecisionRecord] = []
        self.total_seen = 0
        self.evicted = 0

    def append(self, record: BeliefShiftDecisionRecord):
        self.total_seen += 1
        self._records.append(record)
        overflow = len(self._records) - self.limit
        if overflow > 0:
            del self._records[:overflow]
            self.evicted += overflow

    def all(self) -> List[BeliefShiftDecisionRecord]:
        return list(self._records)

    def state(self) -> Dict:
        return {
            "hot_records": len(self._records),
            "limit": self.limit,
            "total_seen": self.total_seen,
            "evicted": self.evicted,
        }


class EpisodeMemory:
    def __init__(self):
        self._episodes: List[Episode] = []
        # V2.19: set by IntegratedCognitiveAgent.
        self.archive_manager = None

    def append(self, episode: Episode):
        self._episodes.append(episode)

    def hot_all(self) -> List[Episode]:
        return list(self._episodes)

    def all(self) -> List[Episode]:
        archived = (
            self.archive_manager.all_episodes()
            if self.archive_manager is not None
            else []
        )
        return sorted(
            archived + list(self._episodes),
            key=lambda e: e.episode_id,
        )

    def for_claim(self, claim_id: str) -> List[Episode]:
        archived = (
            self.archive_manager.episodes_for_claim(
                claim_id
            )
            if self.archive_manager is not None
            else []
        )
        hot = [
            e for e in self._episodes
            if e.claim_id == claim_id
        ]
        return sorted(
            archived + hot,
            key=lambda e: e.episode_id,
        )

    def get(self, episode_id: int) -> Optional[Episode]:
        for episode in self._episodes:
            if episode.episode_id == episode_id:
                return episode

        if self.archive_manager is not None:
            return self.archive_manager.get_episode(
                episode_id
            )

        return None

    def set_outcome(
        self,
        episode_id: int,
        outcome: bool,
    ):
        for episode in self._episodes:
            if episode.episode_id == episode_id:
                episode.outcome = outcome
                return

        if self.archive_manager is not None:
            if self.archive_manager.update_episode_outcome(
                episode_id,
                outcome,
            ):
                return

        raise KeyError(
            f"Episode {episode_id} tidak ditemukan"
        )

    def latest(self, claim_id: str) -> Optional[Episode]:
        items = self.for_claim(claim_id)
        return items[-1] if items else None


class TransitionMemory:
    def __init__(self):
        self._records: List[TransitionRecord] = []

    def append(self, record: TransitionRecord):
        self._records.append(record)

    def all(self) -> List[TransitionRecord]:
        return list(self._records)


class DecisionMemory:
    def __init__(self):
        self._records: List[DecisionRecord] = []

    def append(self, record: DecisionRecord):
        self._records.append(record)

    def all(self) -> List[DecisionRecord]:
        return list(self._records)

    def get(self, decision_id: int) -> Optional[DecisionRecord]:
        for record in self._records:
            if record.decision_id == decision_id:
                return record
        return None

    def for_context(
        self,
        context: str,
        belief_context_id: Optional[str] = None,
    ) -> List[DecisionRecord]:
        return [
            r for r in self._records
            if r.context == context
            and (
                belief_context_id is None
                or r.belief_context_id == belief_context_id
            )
        ]

    def for_belief_context(
        self,
        belief_context_id: str,
    ) -> List[DecisionRecord]:
        return [
            r for r in self._records
            if r.belief_context_id == belief_context_id
        ]


class TrajectoryDecisionMemory:
    def __init__(self):
        self._records: List[TrajectoryDecisionRecord] = []

    def append(self, record: TrajectoryDecisionRecord):
        self._records.append(record)

    def all(self) -> List[TrajectoryDecisionRecord]:
        return list(self._records)

    def get(
        self,
        trajectory_decision_id: int,
    ) -> Optional[TrajectoryDecisionRecord]:
        for record in self._records:
            if record.trajectory_decision_id == trajectory_decision_id:
                return record
        return None


class CounterfactualMemory:
    """
    Menyimpan hasil 'what-if'. Catatan di sini BUKAN episode aktual
    dan tidak boleh langsung memperbarui Q-value.
    """
    def __init__(self):
        self._records: List[CounterfactualEstimate] = []

    def append(self, record: CounterfactualEstimate):
        self._records.append(record)

    def all(self) -> List[CounterfactualEstimate]:
        return list(self._records)

    def for_context(
        self,
        context: str,
        belief_context_id: Optional[str] = None,
    ) -> List[CounterfactualEstimate]:
        return [
            r for r in self._records
            if r.context == context
            and (
                belief_context_id is None
                or r.belief_context_id == belief_context_id
            )
        ]

    def for_belief_context(
        self,
        belief_context_id: str,
    ) -> List[CounterfactualEstimate]:
        return [
            r for r in self._records
            if r.belief_context_id == belief_context_id
        ]


class MetaRiskDecisionMemory:
    def __init__(self):
        self._records: List[MetaRiskDecision] = []

    def append(self, record: MetaRiskDecision):
        self._records.append(record)

    def all(self) -> List[MetaRiskDecision]:
        return list(self._records)

    def get(
        self,
        meta_decision_id: int,
    ) -> Optional[MetaRiskDecision]:
        for record in self._records:
            if record.meta_decision_id == meta_decision_id:
                return record
        return None

    def for_belief_context(
        self,
        belief_context_id: str,
    ) -> List[MetaRiskDecision]:
        return [
            record
            for record in self._records
            if record.belief_context_id == belief_context_id
        ]


class PredictionMemory:
    """
    Stores forecasts, not experiences.

    A prediction never updates the world model merely because it exists.
    """
    def __init__(self):
        self._records: List[OutcomePrediction] = []

    def append(self, record: OutcomePrediction):
        self._records.append(record)

    def all(self) -> List[OutcomePrediction]:
        return list(self._records)

    def get(self, prediction_id: int) -> Optional[OutcomePrediction]:
        for record in self._records:
            if record.prediction_id == prediction_id:
                return record
        return None

    def for_context(
        self,
        context: str,
        belief_context_id: Optional[str] = None,
    ) -> List[OutcomePrediction]:
        return [
            r for r in self._records
            if r.context == context
            and (
                belief_context_id is None
                or r.belief_context_id == belief_context_id
            )
        ]

    def for_belief_context(
        self,
        belief_context_id: str,
    ) -> List[OutcomePrediction]:
        return [
            r for r in self._records
            if r.belief_context_id == belief_context_id
        ]


class PredictionErrorMemory:
    def __init__(self):
        self._records: List[PredictionErrorRecord] = []

    def append(self, record: PredictionErrorRecord):
        self._records.append(record)

    def all(self) -> List[PredictionErrorRecord]:
        return list(self._records)

    def for_context(
        self,
        context: str,
        belief_context_id: Optional[str] = None,
    ) -> List[PredictionErrorRecord]:
        return [
            record
            for record in self._records
            if record.context == context
            and (
                belief_context_id is None
                or record.belief_context_id == belief_context_id
            )
        ]

    def for_belief_context(
        self,
        belief_context_id: str,
    ) -> List[PredictionErrorRecord]:
        return [
            record
            for record in self._records
            if record.belief_context_id == belief_context_id
        ]


@dataclass(frozen=True)
class MemoryRetentionPolicy:
    """
    RAM retention limits for operational histories.

    These memories are safe to compact because the learned state they feed
    already lives separately in Q tables, world-model statistics, and model
    calibration.

    Evidence / grounding / justifications are intentionally NOT compacted by
    this policy because their exact provenance can affect epistemic truth and
    historical reconstruction.
    """
    enabled: bool = True

    decision_hot_limit: int = 2048
    transition_hot_limit: int = 2048
    prediction_hot_limit: int = 8192
    prediction_error_hot_limit: int = 2048
    meta_risk_hot_limit: int = 2048
    counterfactual_hot_limit: int = 1024
    world_decision_hot_limit: int = 1024

    compact_batch: int = 256

    def __post_init__(self):
        values = {
            "decision_hot_limit": self.decision_hot_limit,
            "transition_hot_limit": self.transition_hot_limit,
            "prediction_hot_limit": self.prediction_hot_limit,
            "prediction_error_hot_limit": self.prediction_error_hot_limit,
            "meta_risk_hot_limit": self.meta_risk_hot_limit,
            "counterfactual_hot_limit": self.counterfactual_hot_limit,
            "world_decision_hot_limit": self.world_decision_hot_limit,
            "compact_batch": self.compact_batch,
        }
        for name, value in values.items():
            if value < 1:
                raise ValueError(
                    f"{name} harus >= 1"
                )


@dataclass
class MemoryCompactionSummary:
    """
    Warm-memory aggregate for compacted full-detail operational records.

    This is deliberately NOT used as training data. It exists for audit,
    counts, coarse historical statistics, and integrity tracking only.
    """
    memory_name: str
    compacted_records: int = 0
    first_record_id: Optional[int] = None
    last_record_id: Optional[int] = None

    by_belief_context: Dict[str, int] = field(
        default_factory=dict
    )
    by_action: Dict[str, int] = field(
        default_factory=dict
    )
    by_action_instance: Dict[str, int] = field(
        default_factory=dict
    )
    mode_counts: Dict[str, int] = field(
        default_factory=dict
    )

    reward_count: int = 0
    reward_sum: float = 0.0

    # V2.26 — preserve structured objective meaning across WARM compaction.
    objective_component_counts: Dict[str, int] = field(
        default_factory=dict
    )
    objective_component_sums: Dict[str, float] = field(
        default_factory=dict
    )
    objective_profile_counts: Dict[str, int] = field(
        default_factory=dict
    )
    objective_profile_instance_counts: Dict[str, int] = field(
        default_factory=dict
    )

    error_count: int = 0
    error_sum: float = 0.0
    state_drift_count: int = 0

    rolling_digest: str = "0" * 64

    @property
    def average_reward(self) -> Optional[float]:
        if self.reward_count == 0:
            return None
        return self.reward_sum / self.reward_count

    def average_objectives(self) -> Dict[str, float]:
        output = {}
        for (
            component,
            count,
        ) in self.objective_component_counts.items():
            if count <= 0:
                continue
            output[component] = (
                self.objective_component_sums.get(
                    component,
                    0.0,
                )
                / count
            )
        return output

    @property
    def average_error(self) -> Optional[float]:
        if self.error_count == 0:
            return None
        return self.error_sum / self.error_count

    def absorb(
        self,
        record,
        record_id: Optional[int],
    ):
        self.compacted_records += 1

        if record_id is not None:
            if self.first_record_id is None:
                self.first_record_id = record_id
            self.last_record_id = record_id

        scope = getattr(
            record,
            "belief_context_id",
            None,
        )
        if scope is not None:
            scope = str(scope)
            self.by_belief_context[scope] = (
                self.by_belief_context.get(
                    scope,
                    0,
                )
                + 1
            )

        action = (
            getattr(
                record,
                "selected_action",
                None,
            )
            or getattr(
                record,
                "action_name",
                None,
            )
            or getattr(
                record,
                "action",
                None,
            )
        )
        if action is not None:
            action = str(action)
            self.by_action[action] = (
                self.by_action.get(
                    action,
                    0,
                )
                + 1
            )

        action_instance = (
            getattr(
                record,
                "selected_action_instance_id",
                None,
            )
            or getattr(
                record,
                "action_instance_id",
                None,
            )
        )
        if action_instance is not None:
            action_instance = str(
                action_instance
            )
            self.by_action_instance[
                action_instance
            ] = (
                self.by_action_instance.get(
                    action_instance,
                    0,
                )
                + 1
            )

        mode = getattr(
            record,
            "risk_mode",
            None,
        )
        if mode is None:
            selected_mode = getattr(
                record,
                "selected_mode",
                None,
            )
            if selected_mode is not None:
                mode = getattr(
                    selected_mode,
                    "value",
                    str(selected_mode),
                )
        if mode is not None:
            mode = str(mode)
            self.mode_counts[mode] = (
                self.mode_counts.get(
                    mode,
                    0,
                )
                + 1
            )

        reward = getattr(
            record,
            "reward",
            None,
        )
        if reward is None:
            outcome = getattr(
                record,
                "outcome",
                None,
            )
            if outcome is not None:
                reward = getattr(
                    outcome,
                    "reward",
                    None,
                )
        if reward is not None:
            self.reward_count += 1
            self.reward_sum += float(reward)

        objective_outcome = getattr(
            record,
            "objective_outcome",
            None,
        )
        if isinstance(
            objective_outcome,
            dict,
        ):
            for (
                component,
                value,
            ) in objective_outcome.items():
                if value is None:
                    continue
                self.objective_component_counts[
                    component
                ] = (
                    self.objective_component_counts.get(
                        component,
                        0,
                    )
                    + 1
                )
                self.objective_component_sums[
                    component
                ] = (
                    self.objective_component_sums.get(
                        component,
                        0.0,
                    )
                    + float(value)
                )

        profile_instance = getattr(
            record,
            "objective_profile_instance_id",
            None,
        )
        if profile_instance is not None:
            profile_instance = str(
                profile_instance
            )
            self.objective_profile_instance_counts[
                profile_instance
            ] = (
                self.objective_profile_instance_counts.get(
                    profile_instance,
                    0,
                )
                + 1
            )

        profile_signature = getattr(
            record,
            "objective_profile_signature",
            None,
        )
        if profile_signature is not None:
            profile_signature = str(
                profile_signature
            )
            self.objective_profile_counts[
                profile_signature
            ] = (
                self.objective_profile_counts.get(
                    profile_signature,
                    0,
                )
                + 1
            )

        error = getattr(
            record,
            "aggregate_error",
            None,
        )
        if error is not None:
            self.error_count += 1
            self.error_sum += float(error)

        if bool(
            getattr(
                record,
                "state_drift",
                False,
            )
        ):
            self.state_drift_count += 1

        payload = (
            self.rolling_digest
            + "|"
            + repr(record)
        ).encode(
            "utf-8",
            errors="replace",
        )
        self.rolling_digest = hashlib.sha256(
            payload
        ).hexdigest()

    def state(self) -> Dict:
        return {
            "memory_name": self.memory_name,
            "compacted_records": self.compacted_records,
            "first_record_id": self.first_record_id,
            "last_record_id": self.last_record_id,
            "by_belief_context": dict(
                self.by_belief_context
            ),
            "by_action": dict(
                self.by_action
            ),
            "by_action_instance": dict(
                self.by_action_instance
            ),
            "mode_counts": dict(
                self.mode_counts
            ),
            "reward_count": self.reward_count,
            "average_reward": self.average_reward,
            "objective_component_counts": dict(
                self.objective_component_counts
            ),
            "average_objectives": (
                self.average_objectives()
            ),
            "objective_profile_counts": dict(
                self.objective_profile_counts
            ),
            "objective_profile_instance_counts": dict(
                self.objective_profile_instance_counts
            ),
            "error_count": self.error_count,
            "average_error": self.average_error,
            "state_drift_count": self.state_drift_count,
            "rolling_digest": self.rolling_digest,
        }


class MemoryLifecycleManager:
    """
    HOT:
        recent full-detail operational records in RAM.

    WARM:
        compacted aggregate + rolling integrity digest.

    PROTECTED:
        exact epistemic histories not compacted here:
        EpisodeMemory, Evidence pool, GroundingStore, Justifications.

    V2.17 deliberately does not solve disk persistence/cold archival yet.
    """

    def __init__(
        self,
        agent,
        policy: MemoryRetentionPolicy,
    ):
        self.agent = agent
        self.policy = policy
        self.summaries: Dict[
            str,
            MemoryCompactionSummary,
        ] = {
            name: MemoryCompactionSummary(name)
            for name in (
                "decision",
                "transition",
                "prediction",
                "prediction_error",
                "meta_risk",
                "counterfactual",
                "world_decision",
            )
        }

    def _compact(
        self,
        memory_name: str,
        records: List,
        hot_limit: int,
        id_attr: str,
        eligible=None,
    ) -> int:
        if not self.policy.enabled:
            return 0

        trigger = (
            hot_limit
            + self.policy.compact_batch
        )
        if len(records) <= trigger:
            return 0

        target = len(records) - hot_limit
        summary = self.summaries[
            memory_name
        ]

        keep = []
        compacted = 0

        for record in records:
            can_compact = (
                eligible(record)
                if eligible is not None
                else True
            )

            if (
                compacted < target
                and can_compact
            ):
                record_id = getattr(
                    record,
                    id_attr,
                    None,
                )
                summary.absorb(
                    record,
                    record_id,
                )
                compacted += 1
            else:
                keep.append(record)

        if compacted:
            records[:] = keep

        return compacted

    def maintain(
        self,
        memory_names: Optional[
            Tuple[str, ...]
        ] = None,
    ) -> Dict[str, int]:
        selected = set(
            memory_names
            or self.summaries.keys()
        )

        results: Dict[str, int] = {}

        if "decision" in selected:
            results["decision"] = self._compact(
                "decision",
                self.agent.decision_memory._records,
                self.policy.decision_hot_limit,
                "decision_id",
                eligible=lambda r: (
                    r.reward is not None
                ),
            )

        if "transition" in selected:
            results["transition"] = self._compact(
                "transition",
                self.agent.transition_memory._records,
                self.policy.transition_hot_limit,
                "transition_id",
            )

        if "prediction" in selected:
            active_pins = getattr(
                self.agent,
                "_active_prediction_pins",
                set(),
            )
            results["prediction"] = self._compact(
                "prediction",
                self.agent.prediction_memory._records,
                self.policy.prediction_hot_limit,
                "prediction_id",
                eligible=lambda r: (
                    r.prediction_id
                    not in active_pins
                ),
            )

        if "prediction_error" in selected:
            results["prediction_error"] = self._compact(
                "prediction_error",
                self.agent.prediction_error_memory._records,
                self.policy.prediction_error_hot_limit,
                "prediction_error_id",
            )

        if "meta_risk" in selected:
            results["meta_risk"] = self._compact(
                "meta_risk",
                self.agent.meta_risk_memory._records,
                self.policy.meta_risk_hot_limit,
                "meta_decision_id",
            )

        if "counterfactual" in selected:
            results["counterfactual"] = self._compact(
                "counterfactual",
                self.agent.counterfactual_memory._records,
                self.policy.counterfactual_hot_limit,
                "counterfactual_id",
            )

        if "world_decision" in selected:
            results["world_decision"] = self._compact(
                "world_decision",
                self.agent.world_decision_history,
                self.policy.world_decision_hot_limit,
                "decision_id",
            )

        return results

    def state(self) -> Dict:
        hot = {
            "decision": len(
                self.agent.decision_memory._records
            ),
            "transition": len(
                self.agent.transition_memory._records
            ),
            "prediction": len(
                self.agent.prediction_memory._records
            ),
            "prediction_error": len(
                self.agent.prediction_error_memory._records
            ),
            "meta_risk": len(
                self.agent.meta_risk_memory._records
            ),
            "counterfactual": len(
                self.agent.counterfactual_memory._records
            ),
            "world_decision": len(
                self.agent.world_decision_history
            ),
        }

        compacted = {
            name: summary.compacted_records
            for name, summary
            in self.summaries.items()
        }

        total_seen = {
            name: (
                hot[name]
                + compacted[name]
            )
            for name in hot
        }

        return {
            "enabled": self.policy.enabled,
            "hot": hot,
            "compacted": compacted,
            "total_seen": total_seen,
            "summaries": {
                name: summary.state()
                for name, summary
                in self.summaries.items()
            },
            "active_causal_pins": {
                "prediction_ids": sorted(
                    getattr(
                        self.agent,
                        "_active_prediction_pins",
                        set(),
                    )
                ),
                "count": len(
                    getattr(
                        self.agent,
                        "_active_prediction_pins",
                        set(),
                    )
                ),
            },
            "protected_exact_histories": {
                "episodes_hot": len(
                    self.agent.memory._episodes
                ),
                "evidence_hot": len(
                    self.agent.evidence_pool
                ),
                "groundings": len(
                    self.agent.grounding_store.records
                ),
                "justifications": len(
                    self.agent.justifications
                ),
            },
            "note": (
                "Operational histories use HOT/WARM compaction. "
                "V2.19 may move exact Evidence/Episodes to SQLite COLD; "
                "Groundings/Justifications remain exact in RAM."
            ),
        }


@dataclass(frozen=True)
class EpistemicArchivePolicy:
    """
    Exact cold archival for high-volume epistemic observations.

    V2.19 intentionally archives only:
    - Evidence
    - EpisodeMemory

    GroundingStore and Justifications remain exact in RAM because they are
    lower-volume and structurally involved in proof/TMS mutation. Their
    lifecycle is a separate technical-debt stage.
    """
    enabled: bool = True
    evidence_hot_limit: int = 256
    episode_hot_limit: int = 256
    archive_batch: int = 64

    def __post_init__(self):
        for name, value in {
            "evidence_hot_limit":
                self.evidence_hot_limit,
            "episode_hot_limit":
                self.episode_hot_limit,
            "archive_batch":
                self.archive_batch,
        }.items():
            if value < 1:
                raise ValueError(
                    f"{name} harus >= 1"
                )


def _justification_to_dict(
    value: Justification,
) -> Dict:
    return {
        "conclusion": value.conclusion,
        "premises": list(value.premises),
        "rule_id": value.rule_id,
        "rule_version": value.rule_version,
        "context_id": value.context_id,
        "valid_from": value.valid_from,
        "valid_until": value.valid_until,
    }


def _justification_from_dict(
    value: Dict,
) -> Justification:
    Justification = _canonical_type("Justification")
    return Justification(
        conclusion=value["conclusion"],
        premises=tuple(value["premises"]),
        rule_id=value["rule_id"],
        rule_version=value.get("rule_version"),
        context_id=value.get("context_id"),
        valid_from=value.get("valid_from"),
        valid_until=value.get("valid_until"),
    )


def _evidence_to_json(
    evidence: Evidence,
) -> str:
    return json.dumps(
        {
            "evidence_id": evidence.evidence_id,
            "source": evidence.source,
            "origin_id": evidence.origin_id,
            "claim_id": evidence.claim_id,
            "polarity": evidence.polarity,
            "strength": evidence.strength,
            "observed_at": evidence.observed_at,
            "valid_from": evidence.valid_from,
            "valid_until": evidence.valid_until,
            "context_id": evidence.context_id,
            "observation_quality": evidence.observation_quality,
            "retry_group_id": evidence.retry_group_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _evidence_from_json(
    payload: str,
) -> Evidence:
    Evidence = _canonical_type("Evidence")
    value = json.loads(payload)
    return Evidence(
        evidence_id=value["evidence_id"],
        source=value["source"],
        origin_id=value["origin_id"],
        claim_id=value["claim_id"],
        polarity=int(value["polarity"]),
        strength=float(value["strength"]),
        observed_at=value.get("observed_at"),
        valid_from=value.get("valid_from"),
        valid_until=value.get("valid_until"),
        context_id=value.get("context_id"),
        observation_quality=float(
            value.get("observation_quality", 1.0)
        ),
        retry_group_id=value.get("retry_group_id"),
    )


def _episode_to_json(
    episode: Episode,
) -> str:
    return json.dumps(
        {
            "episode_id": episode.episode_id,
            "claim_id": episode.claim_id,
            "verdict": episode.verdict.value,
            "truth_status":
                episode.truth_status,
            "evidence_status":
                episode.evidence_status,
            "support_score":
                episode.support_score,
            "oppose_score":
                episode.oppose_score,
            "selected_proof": [
                _justification_to_dict(j)
                for j in episode.selected_proof
            ],
            "used_axioms": sorted(
                episode.used_axioms
            ),
            "admission_status":
                episode.admission_status.value,
            "outcome": episode.outcome,
            "notes": episode.notes,
            "belief_context_id":
                episode.belief_context_id,
            "observed_at":
                episode.observed_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _episode_from_json(
    payload: str,
) -> Episode:
    Episode = _canonical_type("Episode")
    AdmissionStatus = _canonical_type("AdmissionStatus")
    EpistemicVerdict = _canonical_type("EpistemicVerdict")
    value = json.loads(payload)
    return Episode(
        episode_id=int(
            value["episode_id"]
        ),
        claim_id=value["claim_id"],
        verdict=EpistemicVerdict(
            value["verdict"]
        ),
        truth_status=value["truth_status"],
        evidence_status=value[
            "evidence_status"
        ],
        support_score=float(
            value["support_score"]
        ),
        oppose_score=float(
            value["oppose_score"]
        ),
        selected_proof=[
            _justification_from_dict(j)
            for j in value[
                "selected_proof"
            ]
        ],
        used_axioms=set(
            value["used_axioms"]
        ),
        admission_status=AdmissionStatus(
            value["admission_status"]
        ),
        outcome=value.get("outcome"),
        notes=value.get(
            "notes",
            "",
        ),
        belief_context_id=value.get(
            "belief_context_id"
        ),
        observed_at=value.get(
            "observed_at"
        ),
    )


@dataclass(frozen=True)
class ObjectiveExperienceRecord:
    """
    Exact COLD record of one ACTUAL structured objective outcome.

    The objective vector is profile-independent historical fact.
    `scalarization_profile_instance_id` and `derived_scalar_utility` are
    provenance describing the preference semantics used at decision time;
    they never scope or rewrite the raw objective vector.
    """
    experience_id: str
    context: str
    belief_context_id: Optional[str]
    state_key: str
    action_name: str
    action_family: str
    action_instance_id: str
    objective_outcome: Dict[str, Optional[float]]
    source_event: str
    decision_id: Optional[int] = None
    transition_id: Optional[int] = None
    observed_at: Optional[int] = None
    success: Optional[bool] = None
    scalarization_profile_instance_id: Optional[str] = None
    derived_scalar_utility: Optional[float] = None

    def __post_init__(self):
        for name, value in (
            ("experience_id", self.experience_id),
            ("context", self.context),
            ("state_key", self.state_key),
            ("action_name", self.action_name),
            ("action_family", self.action_family),
            ("action_instance_id", self.action_instance_id),
            ("source_event", self.source_event),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{name} objective experience tidak boleh kosong"
                )

        canonical = ObjectiveOutcome.coerce(
            self.objective_outcome
        ).as_dict()
        object.__setattr__(
            self,
            "objective_outcome",
            canonical,
        )

        if (
            self.success is not None
            and not isinstance(
                self.success,
                bool,
            )
        ):
            raise TypeError(
                "success objective experience harus bool atau None"
            )

        if (
            self.derived_scalar_utility is not None
            and not (
                0.0
                <= float(
                    self.derived_scalar_utility
                )
                <= 1.0
            )
        ):
            raise ValueError(
                "derived_scalar_utility harus 0..1 atau None"
            )

    @property
    def component_mask(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name in OBJECTIVE_COMPONENTS
            if self.objective_outcome.get(
                name
            ) is not None
        )

    def observed_components(self) -> Dict[str, float]:
        return {
            name: float(
                self.objective_outcome[
                    name
                ]
            )
            for name in self.component_mask
        }

    def as_dict(self) -> Dict:
        return {
            "experience_id":
                self.experience_id,
            "context":
                self.context,
            "belief_context_id":
                self.belief_context_id,
            "state_key":
                self.state_key,
            "action_name":
                self.action_name,
            "action_family":
                self.action_family,
            "action_instance_id":
                self.action_instance_id,
            "objective_outcome":
                dict(
                    self.objective_outcome
                ),
            "component_mask":
                self.component_mask,
            "source_event":
                self.source_event,
            "decision_id":
                self.decision_id,
            "transition_id":
                self.transition_id,
            "observed_at":
                self.observed_at,
            "success":
                self.success,
            "scalarization_profile_instance_id":
                self.scalarization_profile_instance_id,
            "derived_scalar_utility":
                self.derived_scalar_utility,
        }


class ObjectiveExperienceConflict(ValueError):
    pass


def _objective_experience_to_json(
    record: ObjectiveExperienceRecord,
) -> str:
    return json.dumps(
        record.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def _objective_experience_from_json(
    payload: str,
) -> ObjectiveExperienceRecord:
    value = json.loads(
        payload
    )
    return ObjectiveExperienceRecord(
        experience_id=value[
            "experience_id"
        ],
        context=value[
            "context"
        ],
        belief_context_id=value.get(
            "belief_context_id"
        ),
        state_key=value[
            "state_key"
        ],
        action_name=value[
            "action_name"
        ],
        action_family=value[
            "action_family"
        ],
        action_instance_id=value[
            "action_instance_id"
        ],
        objective_outcome=value[
            "objective_outcome"
        ],
        source_event=value[
            "source_event"
        ],
        decision_id=value.get(
            "decision_id"
        ),
        transition_id=value.get(
            "transition_id"
        ),
        observed_at=value.get(
            "observed_at"
        ),
        success=value.get(
            "success"
        ),
        scalarization_profile_instance_id=value.get(
            "scalarization_profile_instance_id"
        ),
        derived_scalar_utility=value.get(
            "derived_scalar_utility"
        ),
    )


class EpistemicArchiveManager:
    """
    Exact SQLite cold store.

    Records are stored as JSON, not pickle. SQLite rows are never treated as
    learned statistics: lazy recall reconstructs the original Evidence /
    Episode objects before the normal epistemic logic runs.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path=None,
    ):
        if path is None:
            fd, generated = tempfile.mkstemp(
                prefix="agen_kognitif_epistemic_",
                suffix=".sqlite3",
            )
            os.close(fd)
            self.path = str(
                Path(generated)
            )
            self._owned_path = True
        else:
            self.path = str(Path(path))
            self._owned_path = False

        self._initialize_schema()

    def _connect(self):
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
        )
        connection.execute(
            "PRAGMA foreign_keys=ON"
        )
        return connection

    @contextmanager
    def _connection(self):
        db = self._connect()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def close(
        self,
        delete_owned: bool = True,
    ):
        if (
            delete_owned
            and getattr(
                self,
                "_owned_path",
                False,
            )
        ):
            path = Path(self.path)
            try:
                path.unlink(
                    missing_ok=True
                )
            except Exception:
                # Destructors must never mask application errors.
                pass

    def __del__(self):
        try:
            self.close(
                delete_owned=True
            )
        except Exception:
            pass

    def _initialize_schema(self):
        Path(self.path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence (
                    archive_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id TEXT NOT NULL,
                    claim_id TEXT NOT NULL,
                    context_id TEXT,
                    observed_at INTEGER,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_claim
                ON evidence(claim_id, archive_seq);

                CREATE TABLE IF NOT EXISTS observation_groups (
                    source TEXT NOT NULL,
                    retry_group_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY(source, retry_group_id)
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id INTEGER PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    belief_context_id TEXT,
                    observed_at INTEGER,
                    outcome INTEGER,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_episode_claim
                ON episodes(claim_id, episode_id);

                CREATE TABLE IF NOT EXISTS objective_experiences (
                    archive_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience_id TEXT NOT NULL UNIQUE,
                    context TEXT NOT NULL,
                    belief_context_id TEXT,
                    state_key TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    action_family TEXT NOT NULL,
                    action_instance_id TEXT NOT NULL,
                    decision_id INTEGER,
                    transition_id INTEGER,
                    observed_at INTEGER,
                    source_event TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_objective_experience_scope
                ON objective_experiences(
                    belief_context_id,
                    state_key,
                    action_instance_id,
                    archive_seq
                );

                CREATE INDEX IF NOT EXISTS idx_objective_experience_decision
                ON objective_experiences(
                    decision_id,
                    archive_seq
                );
                """
            )

            row = db.execute(
                """
                SELECT value
                FROM archive_meta
                WHERE key='schema_version'
                """
            ).fetchone()

            if row is None:
                db.execute(
                    """
                    INSERT INTO archive_meta(key, value)
                    VALUES('schema_version', ?)
                    """,
                    (str(self.SCHEMA_VERSION),),
                )
            elif int(row[0]) != self.SCHEMA_VERSION:
                raise RuntimeError(
                    "Epistemic archive schema tidak didukung"
                )

    def rebind(
        self,
        path,
        owned_path: bool = True,
    ):
        # Do not delete the old path here: an unpickled checkpoint may still
        # temporarily point at the source agent's live archive.
        self.path = str(Path(path))
        self._owned_path = bool(
            owned_path
        )
        self._initialize_schema()

    def evidence_for_retry_group(
        self,
        source: str,
        retry_group_id: str,
    ) -> List[Evidence]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT payload_json
                FROM evidence
                WHERE json_extract(payload_json, '$.source')=?
                  AND json_extract(payload_json, '$.retry_group_id')=?
                ORDER BY archive_seq
                """,
                (source, retry_group_id),
            ).fetchall()

        return [
            _evidence_from_json(row[0])
            for row in rows
        ]

    def observation_group_status(
        self,
        source: str,
        retry_group_id: str,
    ) -> Optional[str]:
        with self._connection() as db:
            row = db.execute(
                """
                SELECT status
                FROM observation_groups
                WHERE source=? AND retry_group_id=?
                """,
                (source, retry_group_id),
            ).fetchone()

        return None if row is None else str(row[0])

    def set_observation_group_status(
        self,
        source: str,
        retry_group_id: str,
        status: str,
    ):
        if status not in {
            "pending",
            "consistent",
            "conflicting",
        }:
            raise ValueError(
                "status observation group tidak valid"
            )

        with self._connection() as db:
            db.execute(
                """
                INSERT INTO observation_groups(
                    source,
                    retry_group_id,
                    status
                ) VALUES(?,?,?)
                ON CONFLICT(source, retry_group_id)
                DO UPDATE SET status=excluded.status
                """,
                (source, retry_group_id, status),
            )

    def observation_group_count(self) -> int:
        with self._connection() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM observation_groups"
                ).fetchone()[0]
            )

    def archive_evidence_batch(
        self,
        records: List[Evidence],
    ) -> int:
        if not records:
            return 0

        rows = [
            (
                e.evidence_id,
                e.claim_id,
                e.context_id,
                e.observed_at,
                _evidence_to_json(e),
            )
            for e in records
        ]

        with self._connection() as db:
            db.executemany(
                """
                INSERT INTO evidence(
                    evidence_id,
                    claim_id,
                    context_id,
                    observed_at,
                    payload_json
                )
                VALUES(?,?,?,?,?)
                """,
                rows,
            )

        return len(rows)

    def reduced_evidence_candidates(
        self,
        claim_id: str,
        context_id: Optional[str],
        as_of: Optional[int],
    ) -> Dict:
        """
        Return exact sufficient candidates for EvidenceAggregator semantics.

        Within one (origin, polarity, source), reliability is constant for a
        query, so only the strongest Evidence from that source can ever win.
        Ties preserve original archive order.

        The query filters temporal/context scope in SQLite before Python object
        reconstruction. It returns scalar rows, not Evidence objects.
        """
        params = [
            claim_id,
            context_id,
            as_of,
            as_of,
            as_of,
        ]

        with self._connection() as db:
            total = int(
                db.execute(
                    """
                    SELECT COUNT(*)
                    FROM evidence
                    WHERE claim_id=?
                    """,
                    (claim_id,),
                ).fetchone()[0]
            )

            in_scope = int(
                db.execute(
                    """
                    SELECT COUNT(*)
                    FROM evidence
                    WHERE claim_id=?
                      AND (
                        context_id IS NULL
                        OR context_id=?
                      )
                      AND (
                        observed_at IS NULL
                        OR observed_at<=?
                      )
                      AND (
                        json_extract(
                            payload_json,
                            '$.valid_from'
                        ) IS NULL
                        OR CAST(
                            json_extract(
                                payload_json,
                                '$.valid_from'
                            ) AS INTEGER
                        )<=?
                      )
                      AND (
                        json_extract(
                            payload_json,
                            '$.valid_until'
                        ) IS NULL
                        OR CAST(
                            json_extract(
                                payload_json,
                                '$.valid_until'
                            ) AS INTEGER
                        )>?
                      )
                    """,
                    params,
                ).fetchone()[0]
            )

            source_rows = db.execute(
                """
                SELECT
                    json_extract(
                        payload_json,
                        '$.source'
                    ) AS source,
                    COUNT(*)
                FROM evidence
                WHERE claim_id=?
                  AND (
                    context_id IS NULL
                    OR context_id=?
                  )
                  AND (
                    observed_at IS NULL
                    OR observed_at<=?
                  )
                  AND (
                    json_extract(
                        payload_json,
                        '$.valid_from'
                    ) IS NULL
                    OR CAST(
                        json_extract(
                            payload_json,
                            '$.valid_from'
                        ) AS INTEGER
                    )<=?
                  )
                  AND (
                    json_extract(
                        payload_json,
                        '$.valid_until'
                    ) IS NULL
                    OR CAST(
                        json_extract(
                            payload_json,
                            '$.valid_until'
                        ) AS INTEGER
                    )>?
                  )
                GROUP BY source
                """,
                params,
            ).fetchall()

            rows = db.execute(
                """
                WITH scoped AS (
                    SELECT
                        archive_seq,
                        evidence_id,
                        json_extract(
                            payload_json,
                            '$.source'
                        ) AS source,
                        json_extract(
                            payload_json,
                            '$.origin_id'
                        ) AS origin_id,
                        CAST(
                            json_extract(
                                payload_json,
                                '$.polarity'
                            ) AS INTEGER
                        ) AS polarity,
                        CAST(
                            json_extract(
                                payload_json,
                                '$.strength'
                            ) AS REAL
                        ) AS strength,
                        COALESCE(
                            CAST(
                                json_extract(
                                    payload_json,
                                    '$.observation_quality'
                                ) AS REAL
                            ),
                            1.0
                        ) AS observation_quality,
                        json_extract(
                            payload_json,
                            '$.retry_group_id'
                        ) AS retry_group_id
                    FROM evidence
                    WHERE claim_id=?
                      AND (
                        context_id IS NULL
                        OR context_id=?
                      )
                      AND (
                        observed_at IS NULL
                        OR observed_at<=?
                      )
                      AND (
                        json_extract(
                            payload_json,
                            '$.valid_from'
                        ) IS NULL
                        OR CAST(
                            json_extract(
                                payload_json,
                                '$.valid_from'
                            ) AS INTEGER
                        )<=?
                      )
                      AND (
                        json_extract(
                            payload_json,
                            '$.valid_until'
                        ) IS NULL
                        OR CAST(
                            json_extract(
                                payload_json,
                                '$.valid_until'
                            ) AS INTEGER
                        )>?
                      )
                ),
                ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY
                                CASE
                                    WHEN retry_group_id IS NOT NULL
                                    THEN 'retry:' || retry_group_id
                                    ELSE 'origin:' || origin_id
                                END,
                                polarity,
                                source
                            ORDER BY
                                (strength * observation_quality) DESC,
                                archive_seq ASC
                        ) AS rn
                    FROM scoped
                )
                SELECT
                    archive_seq,
                    evidence_id,
                    source,
                    origin_id,
                    polarity,
                    strength,
                    observation_quality,
                    retry_group_id
                FROM ranked
                WHERE rn=1
                ORDER BY archive_seq ASC
                """,
                params,
            ).fetchall()

            max_seq = int(
                db.execute(
                    """
                    SELECT COALESCE(
                        MAX(archive_seq),
                        0
                    )
                    FROM evidence
                    """
                ).fetchone()[0]
            )

        return {
            "total_records": total,
            "in_scope_records": in_scope,
            "source_counts": {
                str(source): int(count)
                for source, count
                in source_rows
            },
            "max_archive_seq": max_seq,
            "candidates": [
                {
                    "order": int(row[0]),
                    "evidence_id": row[1],
                    "source": row[2],
                    "origin_id": row[3],
                    "polarity": int(row[4]),
                    "strength": float(row[5]),
                    "observation_quality": float(row[6]),
                    "retry_group_id": row[7],
                }
                for row in rows
            ],
        }

    def evidence_for_claim(
        self,
        claim_id: str,
    ) -> List[Evidence]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT payload_json
                FROM evidence
                WHERE claim_id=?
                ORDER BY archive_seq
                """,
                (claim_id,),
            ).fetchall()

        return [
            _evidence_from_json(row[0])
            for row in rows
        ]

    def all_evidence(
        self,
    ) -> List[Evidence]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT payload_json
                FROM evidence
                ORDER BY archive_seq
                """
            ).fetchall()

        return [
            _evidence_from_json(row[0])
            for row in rows
        ]

    def archive_episode_batch(
        self,
        records: List[Episode],
    ) -> int:
        if not records:
            return 0

        rows = [
            (
                e.episode_id,
                e.claim_id,
                e.belief_context_id,
                e.observed_at,
                (
                    None
                    if e.outcome is None
                    else int(bool(e.outcome))
                ),
                _episode_to_json(e),
            )
            for e in records
        ]

        with self._connection() as db:
            db.executemany(
                """
                INSERT OR REPLACE INTO episodes(
                    episode_id,
                    claim_id,
                    belief_context_id,
                    observed_at,
                    outcome,
                    payload_json
                )
                VALUES(?,?,?,?,?,?)
                """,
                rows,
            )

        return len(rows)

    def get_episode(
        self,
        episode_id: int,
    ) -> Optional[Episode]:
        with self._connection() as db:
            row = db.execute(
                """
                SELECT payload_json
                FROM episodes
                WHERE episode_id=?
                """,
                (episode_id,),
            ).fetchone()

        if row is None:
            return None

        return _episode_from_json(
            row[0]
        )

    def episodes_for_claim(
        self,
        claim_id: str,
    ) -> List[Episode]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT payload_json
                FROM episodes
                WHERE claim_id=?
                ORDER BY episode_id
                """,
                (claim_id,),
            ).fetchall()

        return [
            _episode_from_json(row[0])
            for row in rows
        ]

    def all_episodes(
        self,
    ) -> List[Episode]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT payload_json
                FROM episodes
                ORDER BY episode_id
                """
            ).fetchall()

        return [
            _episode_from_json(row[0])
            for row in rows
        ]

    def update_episode_outcome(
        self,
        episode_id: int,
        outcome: bool,
    ) -> bool:
        episode = self.get_episode(
            episode_id
        )
        if episode is None:
            return False

        episode.outcome = bool(outcome)
        payload = _episode_to_json(
            episode
        )

        with self._connection() as db:
            db.execute(
                """
                UPDATE episodes
                SET outcome=?, payload_json=?
                WHERE episode_id=?
                """,
                (
                    int(bool(outcome)),
                    payload,
                    episode_id,
                ),
            )

        return True


    def archive_objective_experience(
        self,
        record: ObjectiveExperienceRecord,
    ) -> int:
        """
        Append one exact actual-objective record.

        experience_id is immutable. Re-inserting the byte-equivalent payload
        is idempotent; reusing the identity for different historical content
        is rejected.
        """
        if not isinstance(
            record,
            ObjectiveExperienceRecord,
        ):
            raise TypeError(
                "record harus ObjectiveExperienceRecord"
            )

        payload = (
            _objective_experience_to_json(
                record
            )
        )

        with self._connection() as db:
            existing = db.execute(
                """
                SELECT payload_json
                FROM objective_experiences
                WHERE experience_id=?
                """,
                (
                    record.experience_id,
                ),
            ).fetchone()

            if existing is not None:
                if existing[0] == payload:
                    return 0
                raise ObjectiveExperienceConflict(
                    "Objective experience identity collision: "
                    f"{record.experience_id}"
                )

            db.execute(
                """
                INSERT INTO objective_experiences(
                    experience_id,
                    context,
                    belief_context_id,
                    state_key,
                    action_name,
                    action_family,
                    action_instance_id,
                    decision_id,
                    transition_id,
                    observed_at,
                    source_event,
                    payload_json
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.experience_id,
                    record.context,
                    record.belief_context_id,
                    record.state_key,
                    record.action_name,
                    record.action_family,
                    record.action_instance_id,
                    record.decision_id,
                    record.transition_id,
                    record.observed_at,
                    record.source_event,
                    payload,
                ),
            )

        return 1

    def get_objective_experience(
        self,
        experience_id: str,
    ) -> Optional[ObjectiveExperienceRecord]:
        with self._connection() as db:
            row = db.execute(
                """
                SELECT payload_json
                FROM objective_experiences
                WHERE experience_id=?
                """,
                (
                    experience_id,
                ),
            ).fetchone()

        if row is None:
            return None

        return (
            _objective_experience_from_json(
                row[0]
            )
        )

    def objective_experiences(
        self,
        belief_context_id: Optional[
            str
        ] = None,
        state_key: Optional[str] = None,
        action_instance_id: Optional[
            str
        ] = None,
        action_family: Optional[
            str
        ] = None,
        decision_id: Optional[
            int
        ] = None,
        source_event: Optional[
            str
        ] = None,
    ) -> List[ObjectiveExperienceRecord]:
        clauses = []
        params = []

        filters = (
            (
                "belief_context_id",
                belief_context_id,
            ),
            (
                "state_key",
                state_key,
            ),
            (
                "action_instance_id",
                action_instance_id,
            ),
            (
                "action_family",
                action_family,
            ),
            (
                "decision_id",
                decision_id,
            ),
            (
                "source_event",
                source_event,
            ),
        )

        for column, value in filters:
            if value is None:
                continue
            clauses.append(
                f"{column}=?"
            )
            params.append(
                value
            )

        where = (
            ""
            if not clauses
            else (
                " WHERE "
                + " AND ".join(
                    clauses
                )
            )
        )

        with self._connection() as db:
            rows = db.execute(
                """
                SELECT payload_json
                FROM objective_experiences
                """
                + where
                + """
                ORDER BY archive_seq
                """,
                params,
            ).fetchall()

        return [
            _objective_experience_from_json(
                row[0]
            )
            for row in rows
        ]

    def objective_experience_count(
        self,
        belief_context_id: Optional[
            str
        ] = None,
        state_key: Optional[str] = None,
        action_instance_id: Optional[
            str
        ] = None,
    ) -> int:
        clauses = []
        params = []

        for column, value in (
            (
                "belief_context_id",
                belief_context_id,
            ),
            (
                "state_key",
                state_key,
            ),
            (
                "action_instance_id",
                action_instance_id,
            ),
        ):
            if value is None:
                continue
            clauses.append(
                f"{column}=?"
            )
            params.append(
                value
            )

        where = (
            ""
            if not clauses
            else (
                " WHERE "
                + " AND ".join(
                    clauses
                )
            )
        )

        with self._connection() as db:
            return int(
                db.execute(
                    """
                    SELECT COUNT(*)
                    FROM objective_experiences
                    """
                    + where,
                    params,
                ).fetchone()[0]
            )

    def count(
        self,
        table: str,
    ) -> int:
        if table not in {
            "evidence",
            "episodes",
            "objective_experiences",
        }:
            raise ValueError(
                "archive table tidak dikenal"
            )

        with self._connection() as db:
            return int(
                db.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )

    def state(self) -> Dict:
        path = Path(self.path)
        return {
            "schema_version":
                self.SCHEMA_VERSION,
            "path": str(path),
            "owned_path": bool(
                getattr(
                    self,
                    "_owned_path",
                    False,
                )
            ),
            "file_bytes": (
                path.stat().st_size
                if path.exists()
                else 0
            ),
            "evidence_records":
                self.count("evidence"),
            "episode_records":
                self.count("episodes"),
            "objective_experience_records":
                self.count(
                    "objective_experiences"
                ),
            "observation_group_records":
                self.observation_group_count(),
        }

    def snapshot_bytes(self) -> bytes:
        # Every operation uses short-lived committed SQLite connections, so
        # reading after connect-close produces a coherent local snapshot.
        return Path(self.path).read_bytes()


# Trusted-local checkpoint compatibility.
_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"

_PICKLE_COMPAT_CLASSES = (
    BeliefShiftDecisionMemory,
    EpisodeMemory,
    TransitionMemory,
    DecisionMemory,
    TrajectoryDecisionMemory,
    CounterfactualMemory,
    MetaRiskDecisionMemory,
    PredictionMemory,
    PredictionErrorMemory,
    MemoryRetentionPolicy,
    MemoryCompactionSummary,
    MemoryLifecycleManager,
    EpistemicArchivePolicy,
    EpistemicArchiveManager,
    ObjectiveExperienceRecord,
    ObjectiveExperienceConflict,
)

for _cls in _PICKLE_COMPAT_CLASSES:
    _cls.__module__ = _CANONICAL_PICKLE_MODULE

del _cls

__all__ = [
    "BeliefShiftDecisionMemory",
    "EpisodeMemory",
    "TransitionMemory",
    "DecisionMemory",
    "TrajectoryDecisionMemory",
    "CounterfactualMemory",
    "MetaRiskDecisionMemory",
    "PredictionMemory",
    "PredictionErrorMemory",
    "MemoryRetentionPolicy",
    "MemoryCompactionSummary",
    "MemoryLifecycleManager",
    "EpistemicArchivePolicy",
    "EpistemicArchiveManager",
    "ObjectiveExperienceRecord",
    "ObjectiveExperienceConflict",
]
