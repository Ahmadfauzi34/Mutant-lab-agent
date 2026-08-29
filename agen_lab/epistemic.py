"""Epistemic core — physical extraction M6B.

Owns Evidence, source reliability, BeliefContext, robust context shift,
grounding, versioned rules/proofs, TMS, admission, and exact Evidence queries.

No module-level dependency on the compatibility kernel.
"""
from __future__ import annotations

import hashlib
import json
import math

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Hashable

from .planning import Domain
from .memory import BeliefShiftDecisionMemory, EpisodeMemory

@dataclass
class SourceProfile:
    name: str
    alpha: float = 10.0  # Prior keberhasilan / factual accuracy
    beta: float = 1.0    # Prior kegagalan / factual inaccuracy
    parents: frozenset = frozenset()

    # V2.24 — repeatability is NOT factual accuracy.
    # A source can be usually correct but operationally flaky, or stable but
    # systematically wrong. Keep both dimensions separate.
    observation_consistent_groups: float = 0.0
    observation_conflicting_groups: float = 0.0
    observation_prior_consistency: float = 9.0

    @property
    def reliability(self) -> float:
        """Ekspektasi factual/source accuracy Beta(alpha, beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def observation_reliability(self) -> float:
        """
        Repeatability/stability of observations from this source.

        Initial value is 1.0 for backward compatibility. Independent retry
        groups that disagree lower this factor; factual feedback does not.
        """
        denominator = (
            self.observation_prior_consistency
            + self.observation_consistent_groups
            + self.observation_conflicting_groups
        )
        if denominator <= 0:
            return 1.0
        return (
            self.observation_prior_consistency
            + self.observation_consistent_groups
        ) / denominator

    def record_feedback(self, accurate: bool, weight: float = 1.0):
        if accurate:
            self.alpha += weight
        else:
            self.beta += weight

    def record_observation_group_transition(
        self,
        previous_status: Optional[str],
        new_status: str,
    ):
        """Update repeatability counts exactly once per retry-group status."""
        if previous_status == new_status:
            return

        if previous_status == "consistent":
            self.observation_consistent_groups = max(
                0.0,
                self.observation_consistent_groups - 1.0,
            )
        elif previous_status == "conflicting":
            self.observation_conflicting_groups = max(
                0.0,
                self.observation_conflicting_groups - 1.0,
            )

        if new_status == "consistent":
            self.observation_consistent_groups += 1.0
        elif new_status == "conflicting":
            self.observation_conflicting_groups += 1.0
        elif new_status != "pending":
            raise ValueError(
                "observation group status harus pending/consistent/conflicting"
            )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: str
    origin_id: str
    claim_id: str
    polarity: int  # +1 (dukungan) atau -1 (penolakan)
    strength: float

    # V2.9 — temporal/contextual scope.
    # context_id=None berarti evidence global lintas konteks.
    observed_at: Optional[int] = None
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    context_id: Optional[str] = None

    # V2.24 — per-observation quality is distinct from source accuracy and
    # semantic strength. retry_group_id groups technical retries/copies that
    # must not be treated as independent votes.
    observation_quality: float = 1.0
    retry_group_id: Optional[str] = None

    def __post_init__(self):
        if self.polarity not in (-1, 1):
            raise ValueError("polarity harus -1 atau +1")
        if not 0 <= self.strength <= 1:
            raise ValueError("strength harus di rentang 0..1")
        if not 0 <= self.observation_quality <= 1:
            raise ValueError(
                "observation_quality harus di rentang 0..1"
            )
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError("valid_until harus lebih besar dari valid_from")

    def applies_to(
        self,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Scope semantics:
        - context_id=None pada evidence => global
        - query context None => tidak melakukan context filtering
        - temporal interval menggunakan [from, until)
        - evidence tidak berlaku sebelum observed_at
        """
        if (
            context_id is not None
            and self.context_id is not None
            and self.context_id != context_id
        ):
            return False, (
                f"context '{self.context_id}' != query '{context_id}'"
            )

        if as_of is not None:
            starts = [
                t
                for t in (self.observed_at, self.valid_from)
                if t is not None
            ]
            effective_from = max(starts) if starts else None

            if effective_from is not None and as_of < effective_from:
                return False, (
                    f"query t={as_of} sebelum evidence berlaku t={effective_from}"
                )

            if self.valid_until is not None and as_of >= self.valid_until:
                return False, (
                    f"query t={as_of} setelah evidence berakhir "
                    f"t={self.valid_until}"
                )

        return True, "in_scope"


@dataclass
class BeliefContext:
    context_id: str
    valid_from: int
    valid_until: Optional[int] = None
    parent_id: Optional[str] = None
    reason: str = "initial"


class BeliefContextManager:
    """
    Maintains explicit belief epochs.

    Context shift does not delete old evidence. Historical adjudication can
    still query the previous context/time.
    """
    def __init__(self):
        self._counter = 0
        self.now = 0
        self.contexts: List[BeliefContext] = [
            BeliefContext(
                context_id="ctx-0",
                valid_from=0,
                reason="initial",
            )
        ]

    @property
    def current(self) -> BeliefContext:
        return self.contexts[-1]

    @property
    def current_id(self) -> str:
        return self.current.context_id

    def touch(self, observed_at: Optional[int]):
        if observed_at is None:
            return
        if observed_at < self.now:
            # Historical observations are allowed but do not move "now" back.
            return
        self.now = observed_at

    def advance(
        self,
        observed_at: int,
        reason: str,
    ) -> BeliefContext:
        if observed_at < self.now:
            raise ValueError(
                "Context baru tidak boleh dimulai sebelum current time"
            )

        self.touch(observed_at)
        previous = self.current

        if previous.valid_until is None:
            previous.valid_until = observed_at

        self._counter += 1
        context = BeliefContext(
            context_id=f"ctx-{self._counter}",
            valid_from=observed_at,
            parent_id=previous.context_id,
            reason=reason,
        )
        self.contexts.append(context)
        return context

    def context_at(self, as_of: int) -> Optional[BeliefContext]:
        for context in reversed(self.contexts):
            if (
                context.valid_from <= as_of
                and (
                    context.valid_until is None
                    or as_of < context.valid_until
                )
            ):
                return context
        return None

    def get(self, context_id: str) -> Optional[BeliefContext]:
        for context in self.contexts:
            if context.context_id == context_id:
                return context
        return None

    def state(self) -> List[Dict]:
        return [
            {
                "context_id": c.context_id,
                "valid_from": c.valid_from,
                "valid_until": c.valid_until,
                "parent_id": c.parent_id,
                "reason": c.reason,
            }
            for c in self.contexts
        ]


@dataclass
class BeliefShiftCandidate:
    """
    Pending hypothesis that the environment regime may have changed.

    This is NOT evidence and is NOT a belief context. It is bounded
    meta-control state used only to decide when a new context is warranted.
    """
    context_id: str
    claim_id: str
    baseline_evidence_status: str
    expected_polarity: int
    contradiction_polarity: int
    first_observed_at: int
    last_observed_at: int
    contradiction_count: int = 0
    cumulative_strength: float = 0.0
    origin_tokens: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)

    def snapshot(self) -> Dict:
        return {
            "context_id": self.context_id,
            "claim_id": self.claim_id,
            "baseline_evidence_status":
                self.baseline_evidence_status,
            "expected_polarity": self.expected_polarity,
            "contradiction_polarity":
                self.contradiction_polarity,
            "first_observed_at":
                self.first_observed_at,
            "last_observed_at":
                self.last_observed_at,
            "contradiction_count":
                self.contradiction_count,
            "cumulative_strength":
                self.cumulative_strength,
            "independent_signal_count":
                len(self.origin_tokens),
            "sources": tuple(sorted(self.sources)),
        }


@dataclass(frozen=True)
class BeliefShiftDecisionRecord:
    shift_decision_id: int
    claim_id: str
    previous_context: str
    current_context: str
    observed_at: int
    incoming_polarity: int
    incoming_strength: float
    source: Optional[str]
    origin_id: Optional[str]
    previous_evidence_status: str
    baseline_evidence_status: Optional[str]
    expected_polarity: Optional[int]
    contradiction_count: int
    cumulative_strength: float
    independent_signal_count: int
    first_observed_at: Optional[int]
    shifted: bool
    pending: bool
    decision: str
    reason: str


class ContextualBeliefRevisionPolicy:
    """
    V2.23 robust context-shift detector.

    A direct contradiction is only a CHANGE CANDIDATE. The policy requires
    persistent independent contradictory observations before opening a new
    belief context.

    Default hysteresis:
      - at least 2 independent contradictory observations
      - cumulative contradiction strength >= 1.5
      - confirmations must arrive within max_observation_gap=5 interactions
      - a baseline-confirming observation clears the pending candidate
      - repeated explicit origin_id is de-duplicated

    Important:
      pending shift state is meta-control state, NOT empirical Evidence.
    """

    def __init__(
        self,
        min_contradictions: int = 2,
        min_cumulative_strength: float = 1.5,
        max_observation_gap: int = 5,
        minimum_signal_strength: float = 0.25,
    ):
        if min_contradictions < 1:
            raise ValueError(
                "min_contradictions harus >= 1"
            )
        if not 0.0 <= min_cumulative_strength:
            raise ValueError(
                "min_cumulative_strength harus >= 0"
            )
        if max_observation_gap < 1:
            raise ValueError(
                "max_observation_gap harus >= 1"
            )
        if not 0.0 <= minimum_signal_strength <= 1.0:
            raise ValueError(
                "minimum_signal_strength harus 0..1"
            )

        self.min_contradictions = min_contradictions
        self.min_cumulative_strength = (
            min_cumulative_strength
        )
        self.max_observation_gap = (
            max_observation_gap
        )
        self.minimum_signal_strength = (
            minimum_signal_strength
        )

        self._pending: Dict[
            Tuple[str, str],
            BeliefShiftCandidate,
        ] = {}

        # Stable empirical baseline remembered for the lifetime of one belief
        # context. A single noisy Evidence may make the live aggregate
        # "conflicted", but it must not erase what regime the detector was
        # actually monitoring.
        self._baselines: Dict[
            Tuple[str, str],
            Dict,
        ] = {}

    @staticmethod
    def expected_polarity(
        current_evidence_status: str,
    ) -> Optional[int]:
        if current_evidence_status == "accepted":
            return +1
        if current_evidence_status == "rejected":
            return -1
        return None

    def should_shift(
        self,
        current_evidence_status: str,
        incoming_polarity: int,
    ) -> bool:
        """
        Backward-compatible low-level contradiction predicate.

        V2.23 agent-level context shifting uses `assess()` rather than treating
        this predicate as sufficient confirmation.
        """
        if incoming_polarity not in (-1, 1):
            raise ValueError(
                "incoming_polarity harus -1 atau +1"
            )

        expected = self.expected_polarity(
            current_evidence_status
        )
        return (
            expected is not None
            and incoming_polarity != expected
        )

    def _signal_token(
        self,
        origin_id: Optional[str],
        observed_at: int,
        candidate: BeliefShiftCandidate,
    ) -> str:
        if origin_id is not None:
            return f"origin:{origin_id}"

        # Legacy callers may not provide origin_id. Each actual interaction is
        # then considered an independent probe.
        return (
            f"anonymous:{observed_at}:"
            f"{candidate.contradiction_count + 1}"
        )

    def pending_state(self) -> List[Dict]:
        return [
            candidate.snapshot()
            for _, candidate in sorted(
                self._pending.items()
            )
        ]

    def baseline_state(self) -> List[Dict]:
        return [
            {
                "context_id": key[0],
                "claim_id": key[1],
                **dict(value),
            }
            for key, value in sorted(
                self._baselines.items()
            )
        ]

    def clear_context(
        self,
        context_id: str,
    ):
        stale = [
            key
            for key in self._pending
            if key[0] == context_id
        ]
        for key in stale:
            del self._pending[key]

        stale_baselines = [
            key
            for key in self._baselines
            if key[0] == context_id
        ]
        for key in stale_baselines:
            del self._baselines[key]

    def assess(
        self,
        context_id: str,
        claim_id: str,
        current_evidence_status: str,
        incoming_polarity: int,
        observed_at: int,
        incoming_strength: float = 1.0,
        source: Optional[str] = None,
        origin_id: Optional[str] = None,
    ) -> Dict:
        if incoming_polarity not in (-1, 1):
            raise ValueError(
                "incoming_polarity harus -1 atau +1"
            )
        if not 0.0 <= incoming_strength <= 1.0:
            raise ValueError(
                "incoming_strength harus 0..1"
            )

        key = (
            context_id,
            claim_id,
        )

        baseline = self._baselines.get(
            key
        )
        current_expected = self.expected_polarity(
            current_evidence_status
        )
        if (
            baseline is None
            and current_expected is not None
        ):
            baseline = {
                "evidence_status":
                    current_evidence_status,
                "expected_polarity":
                    current_expected,
                "established_at":
                    observed_at,
            }
            self._baselines[key] = baseline

        candidate = self._pending.get(
            key
        )

        expired = False
        if candidate is not None:
            if (
                observed_at
                < candidate.last_observed_at
            ):
                return {
                    "should_shift": False,
                    "pending": True,
                    "decision":
                        "historical_signal_ignored",
                    "reason":
                        "signal older than pending change candidate",
                    "candidate":
                        candidate.snapshot(),
                }

            gap = (
                observed_at
                - candidate.last_observed_at
            )
            if gap > self.max_observation_gap:
                del self._pending[key]
                candidate = None
                expired = True

        if candidate is not None:
            # The candidate's original baseline remains authoritative during
            # confirmation. Current aggregate status may already be conflicted
            # because the first contradictory Evidence was honestly retained.
            if (
                incoming_polarity
                == candidate.expected_polarity
            ):
                snapshot = (
                    candidate.snapshot()
                )
                del self._pending[key]
                return {
                    "should_shift": False,
                    "pending": False,
                    "decision":
                        "candidate_recovered",
                    "reason":
                        "baseline-confirming observation cleared pending change",
                    "candidate": snapshot,
                }

            if (
                incoming_strength
                < self.minimum_signal_strength
            ):
                return {
                    "should_shift": False,
                    "pending": True,
                    "decision":
                        "weak_contradiction_ignored",
                    "reason":
                        "contradiction below minimum signal strength",
                    "candidate":
                        candidate.snapshot(),
                }

            token = self._signal_token(
                origin_id,
                observed_at,
                candidate,
            )
            if token in candidate.origin_tokens:
                candidate.last_observed_at = max(
                    candidate.last_observed_at,
                    observed_at,
                )
                if source is not None:
                    candidate.sources.add(
                        source
                    )
                return {
                    "should_shift": False,
                    "pending": True,
                    "decision":
                        "duplicate_origin_ignored",
                    "reason":
                        "same explicit origin does not count twice",
                    "candidate":
                        candidate.snapshot(),
                }

            candidate.origin_tokens.add(
                token
            )
            if source is not None:
                candidate.sources.add(
                    source
                )
            candidate.contradiction_count += 1
            candidate.cumulative_strength += (
                incoming_strength
            )
            candidate.last_observed_at = (
                observed_at
            )

            confirmed = (
                candidate.contradiction_count
                    >= self.min_contradictions
                and candidate.cumulative_strength
                    >= self.min_cumulative_strength
            )

            return {
                "should_shift": confirmed,
                "pending": not confirmed,
                "decision": (
                    "persistent_change_confirmed"
                    if confirmed
                    else "change_candidate_accumulating"
                ),
                "reason": (
                    "persistent contradictory observations crossed shift threshold"
                    if confirmed
                    else "additional contradiction recorded; threshold not reached"
                ),
                "candidate":
                    candidate.snapshot(),
            }

        expected = (
            baseline["expected_polarity"]
            if baseline is not None
            else None
        )
        baseline_status = (
            baseline["evidence_status"]
            if baseline is not None
            else None
        )

        if expected is None:
            return {
                "should_shift": False,
                "pending": False,
                "decision":
                    "no_strong_baseline",
                "reason": (
                    "no accepted/rejected empirical baseline has yet been "
                    "established for this claim/context"
                ),
                "candidate": None,
                "expired_previous_candidate":
                    expired,
            }

        if incoming_polarity == expected:
            return {
                "should_shift": False,
                "pending": False,
                "decision":
                    "baseline_confirmed",
                "reason":
                    "incoming observation agrees with current strong baseline",
                "candidate": None,
                "expired_previous_candidate":
                    expired,
            }

        if (
            incoming_strength
            < self.minimum_signal_strength
        ):
            return {
                "should_shift": False,
                "pending": False,
                "decision":
                    "weak_contradiction_ignored",
                "reason":
                    "contradiction below minimum signal strength",
                "candidate": None,
                "expired_previous_candidate":
                    expired,
            }

        candidate = BeliefShiftCandidate(
            context_id=context_id,
            claim_id=claim_id,
            baseline_evidence_status=(
                baseline_status
            ),
            expected_polarity=expected,
            contradiction_polarity=(
                incoming_polarity
            ),
            first_observed_at=observed_at,
            last_observed_at=observed_at,
        )
        token = self._signal_token(
            origin_id,
            observed_at,
            candidate,
        )
        candidate.origin_tokens.add(
            token
        )
        if source is not None:
            candidate.sources.add(
                source
            )
        candidate.contradiction_count = 1
        candidate.cumulative_strength = (
            incoming_strength
        )
        self._pending[key] = candidate

        confirmed = (
            candidate.contradiction_count
                >= self.min_contradictions
            and candidate.cumulative_strength
                >= self.min_cumulative_strength
        )

        return {
            "should_shift": confirmed,
            "pending": not confirmed,
            "decision": (
                "persistent_change_confirmed"
                if confirmed
                else "change_candidate_started"
            ),
            "reason": (
                "single contradiction is only a pending change candidate"
                if not confirmed
                else "configured threshold permits immediate confirmation"
            ),
            "candidate":
                candidate.snapshot(),
            "expired_previous_candidate":
                expired,
        }

    def state(self) -> Dict:
        return {
            "min_contradictions":
                self.min_contradictions,
            "min_cumulative_strength":
                self.min_cumulative_strength,
            "max_observation_gap":
                self.max_observation_gap,
            "minimum_signal_strength":
                self.minimum_signal_strength,
            "pending_candidates":
                self.pending_state(),
            "stable_baselines":
                self.baseline_state(),
        }


class SourceLineage:
    def __init__(self, sources: Dict[str, SourceProfile]):
        self.sources = sources

    def root_weights(self, source: str) -> Dict[str, float]:
        """Atribusi fraksional ke akar independen; aman untuk DAG dan mendeteksi cycle."""
        memo: Dict[str, Dict[str, float]] = {}

        def get_roots(node: str, in_stack: Set[str]) -> Dict[str, float]:
            if node in in_stack:
                raise ValueError(f"Silsilah sumber melingkar terdeteksi di '{node}'")
            if node in memo:
                return dict(memo[node])

            profile = self.sources.get(node)
            if profile is None or not profile.parents:
                memo[node] = {node: 1.0}
                return {node: 1.0}

            next_stack = set(in_stack)
            next_stack.add(node)
            combined: Dict[str, float] = {}
            parent_share = 1.0 / len(profile.parents)

            for parent in profile.parents:
                for root, weight in get_roots(parent, next_stack).items():
                    combined[root] = combined.get(root, 0.0) + weight * parent_share

            total = sum(combined.values())
            if total > 0:
                combined = {root: weight / total for root, weight in combined.items()}

            memo[node] = dict(combined)
            return combined

        roots = get_roots(source, set())
        return roots if roots else {source: 1.0}


class EvidenceAggregator:
    def __init__(self, sources: Dict[str, SourceProfile], accept: float = 0.8, conflict: float = 0.5, min_evidence: float = 0.05):
        self.sources = sources
        self.lineage = SourceLineage(sources)
        self.accept = accept
        self.conflict = conflict
        self.min_evidence = min_evidence
        # PERBAIKAN #5: jejak audit untuk bukti yang dibuang
        self.last_dropped: List[Dict] = []
        # V2.9: evidence valid tetapi di luar scope query.
        self.last_out_of_scope: List[Dict] = []
        # V2.24 forensic counters.
        self.last_retry_quarantined_groups: List[str] = []

    def aggregate(
        self,
        claim_id: str,
        evidence: List[Evidence],
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Tuple[str, float, float]:
        support: Dict[str, float] = {}
        oppose: Dict[str, float] = {}
        self.last_dropped = []
        self.last_out_of_scope = []
        self.last_retry_quarantined_groups = []

        # Legacy evidence preserves V2.23 semantics exactly.
        best_by_origin: Dict[
            Tuple[str, int],
            Tuple[Evidence, float],
        ] = {}

        # Explicit retry groups are one technical observation family. Repeats
        # never become independent votes. Opposite outcomes in the same group
        # are quarantined rather than creating artificial epistemic conflict.
        retry_groups: Dict[
            str,
            List[Tuple[Evidence, float]],
        ] = {}

        for e in evidence:
            if e.claim_id != claim_id:
                continue

            in_scope, scope_reason = e.applies_to(
                context_id=context_id,
                as_of=as_of,
            )
            if not in_scope:
                self.last_out_of_scope.append({
                    "evidence_id": e.evidence_id,
                    "reason": scope_reason,
                    "evidence_context": e.context_id,
                    "query_context": context_id,
                    "observed_at": e.observed_at,
                    "valid_from": e.valid_from,
                    "valid_until": e.valid_until,
                    "as_of": as_of,
                })
                continue

            src = self.sources.get(e.source)
            if src is None:
                self.last_dropped.append({
                    "evidence_id": e.evidence_id,
                    "reason": f"sumber '{e.source}' tidak terdaftar",
                })
                continue

            value = (
                e.strength
                * e.observation_quality
                * src.reliability
                * src.observation_reliability
            )

            if e.retry_group_id is not None:
                retry_groups.setdefault(
                    e.retry_group_id,
                    [],
                ).append((e, value))
                continue

            key = (e.origin_id, e.polarity)
            current = best_by_origin.get(key)
            if current is None or value > current[1]:
                if current is not None:
                    self.last_dropped.append({
                        "evidence_id": current[0].evidence_id,
                        "reason": f"duplikat origin '{e.origin_id}' kalah kuat",
                    })
                best_by_origin[key] = (e, value)
            else:
                self.last_dropped.append({
                    "evidence_id": e.evidence_id,
                    "reason": f"duplikat origin '{e.origin_id}' tidak dihitung ulang",
                })

        resolved: List[Tuple[str, int, Evidence, float]] = [
            (origin_id, polarity, e, value)
            for (origin_id, polarity), (e, value)
            in best_by_origin.items()
        ]

        for retry_group_id, items in retry_groups.items():
            polarities = {
                e.polarity
                for e, _ in items
            }

            if len(polarities) > 1:
                self.last_retry_quarantined_groups.append(
                    retry_group_id
                )
                for e, _ in items:
                    self.last_dropped.append({
                        "evidence_id": e.evidence_id,
                        "reason": (
                            f"retry_group '{retry_group_id}' memiliki "
                            "outcome kontradiktif; seluruh group dikarantina"
                        ),
                    })
                continue

            # Same-polarity technical retries collapse to one strongest exact
            # contribution. Python max preserves first item on equal value.
            winner_e, winner_value = max(
                items,
                key=lambda item: item[1],
            )
            for e, _ in items:
                if e is winner_e:
                    continue
                self.last_dropped.append({
                    "evidence_id": e.evidence_id,
                    "reason": (
                        f"retry_group '{retry_group_id}' tidak dihitung ulang"
                    ),
                })

            resolved.append((
                f"retry:{retry_group_id}",
                winner_e.polarity,
                winner_e,
                winner_value,
            ))

        # After independence collapse, lineage attribution remains unchanged.
        for _origin_id, polarity, e, value in resolved:
            root_dist = self.lineage.root_weights(e.source)
            target = support if polarity == 1 else oppose
            for root, share in root_dist.items():
                effective_val = value * share
                target[root] = max(
                    target.get(root, 0.0),
                    effective_val,
                )

        s = 1.0 - math.prod(1.0 - x for x in support.values()) if support else 0.0
        o = 1.0 - math.prod(1.0 - x for x in oppose.values()) if oppose else 0.0

        if 0 < s < self.min_evidence:
            self.last_dropped.append({
                "evidence_id": "(agregat dukungan)",
                "reason": f"skor dukungan {s:.4f} di bawah min_evidence={self.min_evidence}",
            })
        if 0 < o < self.min_evidence:
            self.last_dropped.append({
                "evidence_id": "(agregat penolakan)",
                "reason": f"skor penolakan {o:.4f} di bawah min_evidence={self.min_evidence}",
            })

        if s < self.min_evidence and o < self.min_evidence:
            status = "unresolved"
        elif s >= self.conflict and o >= self.conflict:
            status = "conflicted"
        elif s >= self.min_evidence and o >= self.conflict and o >= s * 0.75:
            status = "conflicted"
        elif s >= self.accept and o < self.conflict:
            status = "accepted"
        elif o >= self.accept and s < self.conflict:
            status = "rejected"
        else:
            status = "unresolved"

        return status, s, o


@dataclass(frozen=True)
class GroundedFact:
    claim_id: str
    origin: str = "manual"
    note: str = ""
    context_id: Optional[str] = None
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None

    def __post_init__(self):
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError(
                "valid_until grounding harus lebih besar dari valid_from"
            )

    def applies_to(
        self,
        context_id: Optional[str],
        as_of: Optional[int],
    ) -> Tuple[bool, str]:
        if (
            context_id is not None
            and self.context_id is not None
            and self.context_id != context_id
        ):
            return False, (
                f"grounding context '{self.context_id}' "
                f"!= query '{context_id}'"
            )

        if as_of is not None:
            if self.valid_from is not None and as_of < self.valid_from:
                return False, (
                    f"query t={as_of} sebelum grounding berlaku "
                    f"t={self.valid_from}"
                )
            if self.valid_until is not None and as_of >= self.valid_until:
                return False, (
                    f"query t={as_of} setelah grounding berakhir "
                    f"t={self.valid_until}"
                )

        return True, "in_scope"


class GroundingStore:
    """
    Scoped grounding registry.

    V2.11:
    - contextual retraction closes validity intervals instead of deleting
      historical facts
    - restore creates a new grounding interval
    - legacy global records remain backward compatible
    """

    def __init__(self):
        self.records: List[GroundedFact] = []
        self.last_out_of_scope: List[Dict] = []

    def add(self, fact: GroundedFact):
        if fact not in self.records:
            self.records.append(fact)

    def close_active(
        self,
        claim_id: str,
        context_id: str,
        as_of: int,
    ) -> List[GroundedFact]:
        """
        Close active groundings that belong EXACTLY to context_id.

        Global groundings (context_id=None) are intentionally not closed by
        contextual retraction. Use the legacy global retract path for those.
        """
        closed: List[GroundedFact] = []
        updated: List[GroundedFact] = []

        for fact in self.records:
            if fact.claim_id != claim_id or fact.context_id != context_id:
                updated.append(fact)
                continue

            applies, _ = fact.applies_to(
                context_id=context_id,
                as_of=as_of,
            )
            if not applies:
                updated.append(fact)
                continue

            # If as_of equals/before valid_from, this interval never became
            # valid for the requested time; remove only that zero-history
            # interval rather than creating an invalid [t,t) interval.
            if fact.valid_from is not None and as_of <= fact.valid_from:
                closed.append(fact)
                continue

            replacement = GroundedFact(
                claim_id=fact.claim_id,
                origin=fact.origin,
                note=fact.note,
                context_id=fact.context_id,
                valid_from=fact.valid_from,
                valid_until=as_of,
            )
            updated.append(replacement)
            closed.append(fact)

        self.records = updated
        return closed

    def records_for(
        self,
        claim_id: str,
        context_id: Optional[str] = None,
    ) -> List[GroundedFact]:
        return [
            fact
            for fact in self.records
            if fact.claim_id == claim_id
            and (
                context_id is None
                or fact.context_id == context_id
            )
        ]

    def active_claims(
        self,
        context_id: Optional[str],
        as_of: Optional[int],
    ) -> Set[str]:
        self.last_out_of_scope = []
        active: Set[str] = set()

        for fact in self.records:
            ok, reason = fact.applies_to(context_id, as_of)
            if ok:
                active.add(fact.claim_id)
            else:
                self.last_out_of_scope.append({
                    "claim_id": fact.claim_id,
                    "origin": fact.origin,
                    "context_id": fact.context_id,
                    "valid_from": fact.valid_from,
                    "valid_until": fact.valid_until,
                    "reason": reason,
                })

        return active

    def active_provenance(
        self,
        claim_id: str,
        context_id: Optional[str],
        as_of: Optional[int],
    ) -> List[Dict]:
        output = []
        for fact in self.records:
            if fact.claim_id != claim_id:
                continue
            ok, _ = fact.applies_to(context_id, as_of)
            if ok:
                output.append({
                    "origin": fact.origin,
                    "note": fact.note,
                    "context_id": fact.context_id,
                    "valid_from": fact.valid_from,
                    "valid_until": fact.valid_until,
                })
        return output


@dataclass(frozen=True)
class Rule:
    """
    Logical rule with explicit version identity.

    `rule_id` is a stable logical family name.
    `rule_version` identifies one immutable semantic/scope version.
    """
    rule_id: str
    domain: str
    premises: tuple
    conclusion: str
    context_id: Optional[str] = None
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    rule_version: Optional[int] = None

    def __post_init__(self):
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError(
                "valid_until rule harus lebih besar dari valid_from"
            )

        if (
            self.rule_version is not None
            and self.rule_version < 1
        ):
            raise ValueError(
                "rule_version harus >= 1"
            )

    @property
    def instance_id(self) -> str:
        version = (
            self.rule_version
            if self.rule_version is not None
            else "unregistered"
        )
        return (
            f"{self.rule_id}@v{version}"
        )

    def applies_to(
        self,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Tuple[bool, str]:
        if (
            context_id is not None
            and self.context_id is not None
            and self.context_id != context_id
        ):
            return False, (
                f"rule context '{self.context_id}' != query '{context_id}'"
            )

        if as_of is not None:
            if (
                self.valid_from is not None
                and as_of < self.valid_from
            ):
                return False, (
                    f"query t={as_of} sebelum rule berlaku "
                    f"t={self.valid_from}"
                )
            if (
                self.valid_until is not None
                and as_of >= self.valid_until
            ):
                return False, (
                    f"query t={as_of} setelah rule berakhir "
                    f"t={self.valid_until}"
                )

        return True, "in_scope"


class RuleVersionConflict(ValueError):
    pass


class RuleValidator:
    """
    Version-aware rule registry.

    Backward compatibility:
    - `rules[rule_id]` remains a latest-version projection.
    - exact reasoning uses `versions` + `rule_version`.

    Safety:
    - same rule_id no longer silently overwrites history;
    - incompatible versions in the same exact scope may not overlap in time;
    - unversioned Justification references become invalid if more than one
      matching active version exists, rather than silently binding to one.
    """

    def __init__(self, domain: Domain):
        self.domain = domain

        # Legacy/latest projection only.
        self.rules: Dict[str, Rule] = {}

        # Canonical storage.
        self.versions: Dict[
            str,
            List[Rule],
        ] = {}

        self.semantic_validator = None

    def set_semantic_validator(
        self,
        validator,
    ):
        """Pasang validator domain: callable(rule) -> bool."""
        self.semantic_validator = validator

    @staticmethod
    def _intervals_overlap(
        left: Rule,
        right: Rule,
    ) -> bool:
        left_start = (
            float("-inf")
            if left.valid_from is None
            else left.valid_from
        )
        right_start = (
            float("-inf")
            if right.valid_from is None
            else right.valid_from
        )
        left_end = (
            float("inf")
            if left.valid_until is None
            else left.valid_until
        )
        right_end = (
            float("inf")
            if right.valid_until is None
            else right.valid_until
        )

        return max(
            left_start,
            right_start,
        ) < min(
            left_end,
            right_end,
        )

    @staticmethod
    def _same_semantics(
        left: Rule,
        right: Rule,
    ) -> bool:
        return (
            left.domain == right.domain
            and tuple(left.premises)
                == tuple(right.premises)
            and left.conclusion
                == right.conclusion
        )

    @staticmethod
    def _same_registration(
        left: Rule,
        right: Rule,
    ) -> bool:
        return (
            RuleValidator._same_semantics(
                left,
                right,
            )
            and left.context_id
                == right.context_id
            and left.valid_from
                == right.valid_from
            and left.valid_until
                == right.valid_until
        )

    def all_versions(
        self,
        rule_id: str,
    ) -> List[Rule]:
        return list(
            self.versions.get(
                rule_id,
                [],
            )
        )

    def get_version(
        self,
        rule_id: str,
        rule_version: int,
    ) -> Optional[Rule]:
        for rule in self.versions.get(
            rule_id,
            [],
        ):
            if (
                rule.rule_version
                == rule_version
            ):
                return rule

        return None

    def register(
        self,
        rule: Rule,
    ) -> Rule:
        if (
            rule.domain
            != self.domain.name
        ):
            raise ValueError(
                f"Domain mismatch: "
                f"{rule.domain} != {self.domain.name}"
            )

        existing = self.versions.setdefault(
            rule.rule_id,
            [],
        )

        # Idempotent exact registration.
        for old in existing:
            if (
                self._same_registration(
                    old,
                    rule,
                )
                and (
                    rule.rule_version is None
                    or old.rule_version
                        == rule.rule_version
                )
            ):
                return old

        if rule.rule_version is None:
            next_version = (
                max(
                    (
                        r.rule_version or 0
                        for r in existing
                    ),
                    default=0,
                )
                + 1
            )
        else:
            next_version = (
                rule.rule_version
            )

        candidate = replace(
            rule,
            rule_version=next_version,
        )

        for old in existing:
            if (
                old.rule_version
                == candidate.rule_version
            ):
                if old == candidate:
                    return old
                raise RuleVersionConflict(
                    "Rule version collision: "
                    f"{candidate.instance_id}"
                )

        # Same exact context scope is a sequential version family.
        # Any overlapping non-identical registration is identity-ambiguous,
        # even if its logical semantics happen to be the same.
        for old in existing:
            if (
                old.context_id
                == candidate.context_id
                and self._intervals_overlap(
                    old,
                    candidate,
                )
            ):
                raise RuleVersionConflict(
                    "Rule versions overlap in the same exact scope: "
                    f"{old.instance_id} vs {candidate.instance_id}. "
                    "Gunakan supersede_contextual_rule() atau valid_until."
                )

        if (
            self.semantic_validator
            is not None
            and not self.semantic_validator(
                candidate
            )
        ):
            raise ValueError(
                f"Rule '{candidate.instance_id}' "
                "gagal validasi semantik domain"
            )

        existing.append(
            candidate
        )
        existing.sort(
            key=lambda r: (
                r.rule_version or 0
            )
        )

        # Legacy projection, never canonical proof lookup.
        self.rules[
            candidate.rule_id
        ] = existing[-1]

        return candidate

    def close_version(
        self,
        rule_id: str,
        rule_version: int,
        valid_until: int,
    ) -> Rule:
        rules = self.versions.get(
            rule_id,
            [],
        )

        for index, old in enumerate(
            rules
        ):
            if (
                old.rule_version
                != rule_version
            ):
                continue

            if (
                old.valid_from
                is not None
                and valid_until
                    <= old.valid_from
            ):
                raise ValueError(
                    "valid_until harus setelah valid_from rule"
                )

            if (
                old.valid_until
                is not None
                and valid_until
                    > old.valid_until
            ):
                raise ValueError(
                    "close_version tidak boleh memperpanjang rule lama"
                )

            closed = replace(
                old,
                valid_until=valid_until,
            )
            rules[index] = closed

            if (
                self.rules.get(
                    rule_id
                ) is old
                or (
                    self.rules.get(
                        rule_id
                    ) is not None
                    and self.rules[
                        rule_id
                    ].rule_version
                        == rule_version
                )
            ):
                self.rules[
                    rule_id
                ] = max(
                    rules,
                    key=lambda r: (
                        r.rule_version or 0
                    ),
                )

            return closed

        raise KeyError(
            f"Rule {rule_id}@v{rule_version} tidak ditemukan"
        )

    def matching_versions(
        self,
        rule_id: str,
        premises: Optional[Set[str]] = None,
        conclusion: Optional[str] = None,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
        require_applicable: bool = True,
    ) -> List[Rule]:
        output = []

        for rule in self.versions.get(
            rule_id,
            [],
        ):
            if (
                conclusion is not None
                and rule.conclusion
                    != conclusion
            ):
                continue

            if (
                premises is not None
                and set(rule.premises)
                    != set(premises)
            ):
                continue

            if require_applicable:
                applicable, _ = (
                    rule.applies_to(
                        context_id=context_id,
                        as_of=as_of,
                    )
                )
                if not applicable:
                    continue

            output.append(rule)

        return output

    def reference_rule(
        self,
        rule_id: str,
        rule_version: Optional[int],
        premises: Set[str],
        conclusion: str,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Optional[Rule]:
        """
        Resolve a Justification reference without silent version rebinding.

        Exact version -> exact instance.
        Legacy unversioned reference -> only accepted if unambiguous.
        """
        if rule_version is not None:
            return self.get_version(
                rule_id,
                rule_version,
            )

        semantic = self.matching_versions(
            rule_id,
            premises=premises,
            conclusion=conclusion,
            require_applicable=False,
        )

        if len(semantic) == 1:
            return semantic[0]

        applicable = [
            rule
            for rule in semantic
            if rule.applies_to(
                context_id=context_id,
                as_of=as_of,
            )[0]
        ]

        if len(applicable) == 1:
            return applicable[0]

        # More than one possible version is an identity ambiguity.
        return None

    def valid(
        self,
        rule_id: str,
        premises: Set[str],
        conclusion: str,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
        rule_version: Optional[int] = None,
    ) -> bool:
        rule = self.reference_rule(
            rule_id,
            rule_version,
            premises,
            conclusion,
            context_id=context_id,
            as_of=as_of,
        )
        if rule is None:
            return False

        applicable, _ = rule.applies_to(
            context_id=context_id,
            as_of=as_of,
        )
        if not applicable:
            return False

        if (
            rule.domain
            != self.domain.name
            or rule.conclusion
                != conclusion
        ):
            return False

        if (
            set(rule.premises)
            != set(premises)
        ):
            return False

        if (
            self.semantic_validator
            is not None
            and not self.semantic_validator(
                rule
            )
        ):
            return False

        return True


@dataclass(frozen=True)
class Justification:
    conclusion: str
    premises: tuple
    rule_id: str
    context_id: Optional[str] = None
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    rule_version: Optional[int] = None

    def __post_init__(self):
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError(
                "valid_until justification harus lebih besar dari valid_from"
            )

        if (
            self.rule_version is not None
            and self.rule_version < 1
        ):
            raise ValueError(
                "justification rule_version harus >= 1"
            )

    @property
    def rule_instance_id(
        self,
    ) -> str:
        version = (
            self.rule_version
            if self.rule_version is not None
            else "legacy"
        )
        return (
            f"{self.rule_id}@v{version}"
        )

    def applies_to(
        self,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Tuple[bool, str]:
        if (
            context_id is not None
            and self.context_id is not None
            and self.context_id != context_id
        ):
            return False, (
                f"justification context '{self.context_id}' "
                f"!= query '{context_id}'"
            )

        if as_of is not None:
            if (
                self.valid_from is not None
                and as_of < self.valid_from
            ):
                return False, (
                    f"query t={as_of} sebelum justification berlaku "
                    f"t={self.valid_from}"
                )
            if (
                self.valid_until is not None
                and as_of >= self.valid_until
            ):
                return False, (
                    f"query t={as_of} setelah justification berakhir "
                    f"t={self.valid_until}"
                )

        return True, "in_scope"


class RelevanceSlicer:
    """Mengisolasi hanya subgraf kausal yang dibutuhkan oleh klaim target"""
    @staticmethod
    def slice(target_claim: str, justifications: List[Justification]) -> List[Justification]:
        just_by_conclusion: Dict[str, List[Justification]] = {}
        for j in justifications:
            just_by_conclusion.setdefault(j.conclusion, []).append(j)

        needed_claims = {target_claim}
        queue = [target_claim]
        relevant_justifications = []
        seen_just = set()

        while queue:
            curr = queue.pop(0)
            for j in just_by_conclusion.get(curr, []):
                if j not in seen_just:
                    seen_just.add(j)
                    relevant_justifications.append(j)
                    for p in j.premises:
                        if p not in needed_claims:
                            needed_claims.add(p)
                            queue.append(p)

        return relevant_justifications


class ProvenanceGraph:
    def __init__(self):
        self.edges: Dict[str, Set[str]] = {}

    def add(self, j: Justification):
        self.edges.setdefault(j.conclusion, set()).update(j.premises)

    def acyclic(self) -> bool:
        state = {}
        for n in self.edges:
            state[n] = 0
        for ps in self.edges.values():
            for p in ps:
                state.setdefault(p, 0)

        def visit(n):
            if state[n] == 1:
                return False
            if state[n] == 2:
                return True
            state[n] = 1
            for child in self.edges.get(n, ()):
                if not visit(child):
                    return False
            state[n] = 2
            return True

        return all(visit(n) for n in state if state[n] == 0)


class TruthEvaluator:
    def __init__(self, rule_validator: RuleValidator):
        self.rules = rule_validator
        self.last_selected_proof: List[Justification] = []
        self.last_out_of_scope_proof: List[Dict] = []

    def evaluate(
        self,
        claim_id: str,
        justifications: List[Justification],
        grounded: Set[str],
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> Tuple[str, Set[str]]:
        self.last_selected_proof = []
        self.last_out_of_scope_proof = []

        sliced = RelevanceSlicer.slice(
            claim_id,
            justifications,
        )

        if claim_id in grounded:
            return "supported", {claim_id}
        if not sliced:
            return "unknown", set()

        valid: List[Justification] = []
        invalid: List[Justification] = []

        for j in sliced:
            j_applicable, j_reason = j.applies_to(
                context_id=context_id,
                as_of=as_of,
            )
            if not j_applicable:
                self.last_out_of_scope_proof.append({
                    "type": "justification",
                    "rule_id": j.rule_id,
                    "rule_version": j.rule_version,
                    "rule_instance_id": j.rule_instance_id,
                    "conclusion": j.conclusion,
                    "reason": j_reason,
                })
                continue

            rule = self.rules.reference_rule(
                j.rule_id,
                j.rule_version,
                set(j.premises),
                j.conclusion,
                context_id=context_id,
                as_of=as_of,
            )
            if rule is not None:
                r_applicable, r_reason = rule.applies_to(
                    context_id=context_id,
                    as_of=as_of,
                )
                if not r_applicable:
                    self.last_out_of_scope_proof.append({
                        "type": "rule",
                        "rule_id": j.rule_id,
                        "rule_version": j.rule_version,
                        "rule_instance_id": j.rule_instance_id,
                        "conclusion": j.conclusion,
                        "reason": r_reason,
                    })
                    continue

            if self.rules.valid(
                j.rule_id,
                set(j.premises),
                j.conclusion,
                context_id=context_id,
                as_of=as_of,
                rule_version=j.rule_version,
            ):
                valid.append(j)
            else:
                invalid.append(j)

        by_conclusion: Dict[str, List[Justification]] = {}
        for j in valid:
            by_conclusion.setdefault(
                j.conclusion,
                [],
            ).append(j)

        cycle_seen = False

        def prove(node: str, stack: Set[str]):
            nonlocal cycle_seen

            if node in grounded:
                return True, {node}, []

            if node in stack:
                cycle_seen = True
                return False, set(), []

            next_stack = set(stack)
            next_stack.add(node)

            for j in by_conclusion.get(node, []):
                axioms: Set[str] = set()
                proof: List[Justification] = []
                ok = True

                for premise in j.premises:
                    p_ok, p_axioms, p_proof = prove(
                        premise,
                        next_stack,
                    )
                    if not p_ok:
                        ok = False
                        break

                    axioms.update(p_axioms)
                    proof.extend(p_proof)

                if ok:
                    proof.append(j)
                    return True, axioms, proof

            return False, set(), []

        ok, axioms, proof = prove(
            claim_id,
            set(),
        )

        if ok:
            selected_graph = ProvenanceGraph()
            for j in proof:
                selected_graph.add(j)

            if not selected_graph.acyclic():
                return "invalid", set()

            self.last_selected_proof = proof
            return "supported", axioms

        graph = ProvenanceGraph()
        for j in valid:
            graph.add(j)

        # Out-of-scope proof artifacts are not logical errors.
        if invalid or cycle_seen or not graph.acyclic():
            return "invalid", set()

        return "unknown", set()


class EpistemicVerdict(str, Enum):
    VERIFIED_FACT = "VERIFIED_FACT"
    THEORETICAL_CONSTRUCT = "THEORETICAL_CONSTRUCT"
    EMPIRICAL_ANOMALY = "EMPIRICAL_ANOMALY"
    EMPIRICAL_DISCOVERY = "EMPIRICAL_DISCOVERY"
    EMPIRICAL_REFUTATION = "EMPIRICAL_REFUTATION"
    EPISTEMIC_CRISIS = "EPISTEMIC_CRISIS"
    LOGICAL_FALLACY = "LOGICAL_FALLACY"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class AuditReport:
    target: str
    passed: bool
    risk_level: str
    challenge_name: str
    details: str


class AdmissionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    PENDING = "PENDING"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"


@dataclass
class Episode:
    episode_id: int
    claim_id: str
    verdict: EpistemicVerdict
    truth_status: str
    evidence_status: str
    support_score: float
    oppose_score: float
    selected_proof: List[Justification]
    used_axioms: Set[str]
    admission_status: AdmissionStatus
    outcome: Optional[bool] = None
    notes: str = ""
    belief_context_id: Optional[str] = None
    observed_at: Optional[int] = None


class KnowledgeAdmissionPolicy:
    """
    V2.2:
    - VERIFIED_FACT          -> ACCEPTED
    - EMPIRICAL_DISCOVERY    -> PENDING
    - THEORETICAL_CONSTRUCT  -> PENDING
    - EPISTEMIC_CRISIS       -> QUARANTINED
    - LOGICAL_FALLACY        -> REJECTED
    - EMPIRICAL_REFUTATION   -> REJECTED
    - EMPIRICAL_ANOMALY      -> QUARANTINED
    - sisanya                -> PENDING

    Catatan:
    ACCEPTED tidak otomatis menjadi grounded axiom.
    Ini mencegah derived claim berubah diam-diam menjadi premis dasar.
    """
    def decide(self, verdict: EpistemicVerdict) -> AdmissionStatus:
        mapping = {
            EpistemicVerdict.VERIFIED_FACT: AdmissionStatus.ACCEPTED,
            EpistemicVerdict.EMPIRICAL_DISCOVERY: AdmissionStatus.PENDING,
            EpistemicVerdict.THEORETICAL_CONSTRUCT: AdmissionStatus.PENDING,
            EpistemicVerdict.EPISTEMIC_CRISIS: AdmissionStatus.QUARANTINED,
            EpistemicVerdict.LOGICAL_FALLACY: AdmissionStatus.REJECTED,
            EpistemicVerdict.EMPIRICAL_REFUTATION: AdmissionStatus.REJECTED,
            EpistemicVerdict.EMPIRICAL_ANOMALY: AdmissionStatus.QUARANTINED,
            EpistemicVerdict.UNRESOLVED: AdmissionStatus.PENDING,
        }
        return mapping[verdict]


class DependencyGraph:
    """
    Dependency graph eksplisit:
        premise -> conclusions

    Dipakai untuk mencari semua claim yang terdampak saat
    sebuah grounded fact / premise dicabut atau dipulihkan.
    """
    def __init__(self):
        self.forward: Dict[str, Set[str]] = {}
        self.reverse: Dict[str, Set[str]] = {}

    def add_justification(self, justification: Justification):
        conclusion = justification.conclusion
        for premise in justification.premises:
            self.forward.setdefault(premise, set()).add(conclusion)
            self.reverse.setdefault(conclusion, set()).add(premise)

    def rebuild(self, justifications: List[Justification]):
        self.forward.clear()
        self.reverse.clear()
        for j in justifications:
            self.add_justification(j)

    def rebuild_scoped(
        self,
        justifications: List[Justification],
        rule_validator: RuleValidator,
        context_id: Optional[str],
        as_of: Optional[int],
    ):
        """
        Build only dependencies that can participate in the requested proof
        scope. Stale rules/justifications do not propagate TMS invalidation.
        """
        self.forward.clear()
        self.reverse.clear()

        for j in justifications:
            j_ok, _ = j.applies_to(
                context_id=context_id,
                as_of=as_of,
            )
            if not j_ok:
                continue

            if not rule_validator.valid(
                j.rule_id,
                set(j.premises),
                j.conclusion,
                context_id=context_id,
                as_of=as_of,
                rule_version=j.rule_version,
            ):
                continue

            self.add_justification(j)

    def descendants(self, node: str) -> Set[str]:
        result = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for child in self.forward.get(current, set()):
                if child not in result:
                    result.add(child)
                    queue.append(child)

        return result

    def ancestors(self, node: str) -> Set[str]:
        result = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            for parent in self.reverse.get(current, set()):
                if parent not in result:
                    result.add(parent)
                    queue.append(parent)

        return result


class TruthMaintenanceSystem:
    """
    Truth Maintenance System.

    Legacy path:
      retract()/restore_grounded() preserve V2.3 global behavior.

    V2.11 contextual path:
      - scoped dependency graph
      - temporal interval closure rather than history deletion
      - historical queries remain valid
      - admissions are revised only for the target belief context
    """

    def __init__(self, agent):
        self.agent = agent
        self.dependencies = DependencyGraph()
        self.history: List[Dict] = []

    # -----------------------------
    # Legacy/global compatibility
    # -----------------------------

    def rebuild(self):
        self.dependencies.rebuild(
            self.agent.justifications
        )

    def affected_by(self, premise: str) -> Set[str]:
        self.rebuild()
        return self.dependencies.descendants(premise)

    def reevaluate_claims(
        self,
        claims: Set[str],
    ) -> Dict[str, Dict]:
        reports = {}

        for claim_id in sorted(claims):
            report = self.agent.adjudicate_claim(
                claim_id
            )
            reports[claim_id] = report

            status = self.agent.admission_policy.decide(
                report["verdict"]
            )
            self.agent._apply_admission(
                claim_id,
                status,
            )

        return reports

    def retract(
        self,
        claim: str,
        reason: str = "",
    ) -> Dict:
        """
        Legacy GLOBAL retract.

        Kept for backward compatibility. It should not be used for scoped
        temporal facts.
        """
        existed = claim in self.agent.grounded
        previous_provenance = (
            self.agent.grounded_provenance.pop(
                claim,
                None,
            )
        )
        self.agent.grounded.discard(claim)

        # Remove legacy global GroundingStore records too, otherwise active
        # grounding would resurrect the fact through active_grounded().
        self.agent.grounding_store.records = [
            fact
            for fact in self.agent.grounding_store.records
            if not (
                fact.claim_id == claim
                and fact.context_id is None
                and fact.valid_from is None
                and fact.valid_until is None
            )
        ]

        affected = self.affected_by(claim)
        reports = self.reevaluate_claims(
            affected
        )

        record = {
            "operation": "global_retract",
            "claim": claim,
            "existed": existed,
            "reason": reason,
            "previous_provenance": previous_provenance,
            "affected_claims": sorted(affected),
            "reevaluated": reports,
        }
        self.history.append(record)
        return record

    def restore_grounded(
        self,
        claim: str,
        origin: str = "manual_restore",
        note: str = "",
    ) -> Dict:
        self.agent.add_grounded(
            claim,
            origin=origin,
            note=note,
        )

        affected = self.affected_by(claim)
        reports = self.reevaluate_claims(
            affected
        )

        record = {
            "operation": "global_restore",
            "claim": claim,
            "affected_claims": sorted(affected),
            "reevaluated": reports,
        }
        self.history.append(record)
        return record

    # -----------------------------
    # V2.11 contextual/temporal TMS
    # -----------------------------

    def rebuild_scoped(
        self,
        context_id: str,
        as_of: int,
    ):
        self.dependencies.rebuild_scoped(
            self.agent.justifications,
            self.agent.rule_validator,
            context_id=context_id,
            as_of=as_of,
        )

    def affected_by_scoped(
        self,
        premise: str,
        context_id: str,
        as_of: int,
    ) -> Set[str]:
        self.rebuild_scoped(
            context_id=context_id,
            as_of=as_of,
        )
        return self.dependencies.descendants(
            premise
        )

    def reevaluate_claims_scoped(
        self,
        claims: Set[str],
        context_id: str,
        as_of: int,
    ) -> Dict[str, Dict]:
        reports: Dict[str, Dict] = {}

        for claim_id in sorted(claims):
            report = self.agent.adjudicate_claim(
                claim_id,
                context_id=context_id,
                as_of=as_of,
            )
            reports[claim_id] = report

            status = self.agent.admission_policy.decide(
                report["verdict"]
            )
            self.agent._apply_admission(
                claim_id,
                status,
                context_id=context_id,
            )

        return reports

    def retract_contextual(
        self,
        claim: str,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
        reason: str = "",
    ) -> Dict:
        if as_of is not None:
            self.agent.touch_interaction_time(as_of)

        context_id, as_of = (
            self.agent._resolve_belief_scope(
                context_id=context_id,
                as_of=as_of,
            )
        )

        before = self.agent.adjudicate_claim(
            claim,
            context_id=context_id,
            as_of=as_of,
        )

        closed = (
            self.agent.grounding_store.close_active(
                claim_id=claim,
                context_id=context_id,
                as_of=as_of,
            )
        )

        affected = self.affected_by_scoped(
            claim,
            context_id=context_id,
            as_of=as_of,
        )
        reports = self.reevaluate_claims_scoped(
            affected,
            context_id=context_id,
            as_of=as_of,
        )

        after = self.agent.adjudicate_claim(
            claim,
            context_id=context_id,
            as_of=as_of,
        )

        record = {
            "operation": "contextual_retract",
            "claim": claim,
            "context_id": context_id,
            "as_of": as_of,
            "reason": reason,
            "closed_groundings": [
                {
                    "claim_id": fact.claim_id,
                    "origin": fact.origin,
                    "context_id": fact.context_id,
                    "valid_from": fact.valid_from,
                    "valid_until": fact.valid_until,
                }
                for fact in closed
            ],
            "claim_truth_before": before["truth_status"],
            "claim_truth_after": after["truth_status"],
            "affected_claims": sorted(affected),
            "reevaluated": reports,
        }
        self.history.append(record)
        return record

    def restore_contextual(
        self,
        claim: str,
        context_id: Optional[str] = None,
        as_of: Optional[int] = None,
        origin: str = "contextual_restore",
        note: str = "",
    ) -> Dict:
        if as_of is not None:
            self.agent.touch_interaction_time(as_of)

        context_id, as_of = (
            self.agent._resolve_belief_scope(
                context_id=context_id,
                as_of=as_of,
            )
        )

        fact = self.agent.add_contextual_grounded(
            claim=claim,
            origin=origin,
            note=note,
            context_id=context_id,
            valid_from=as_of,
        )

        affected = self.affected_by_scoped(
            claim,
            context_id=context_id,
            as_of=as_of,
        )
        reports = self.reevaluate_claims_scoped(
            affected,
            context_id=context_id,
            as_of=as_of,
        )

        record = {
            "operation": "contextual_restore",
            "claim": claim,
            "context_id": context_id,
            "as_of": as_of,
            "restored_grounding": {
                "claim_id": fact.claim_id,
                "origin": fact.origin,
                "context_id": fact.context_id,
                "valid_from": fact.valid_from,
                "valid_until": fact.valid_until,
            },
            "affected_claims": sorted(affected),
            "reevaluated": reports,
        }
        self.history.append(record)
        return record


class EvidenceAuditMode(str, Enum):
    FULL = "full"
    COMPACT = "compact"


@dataclass(frozen=True)
class IndexedEvidenceAggregate:
    status: str
    support_score: float
    oppose_score: float

    total_records: int
    in_scope_records: int
    out_of_scope_count: int
    unknown_source_count: int
    duplicate_drop_count: int
    source_candidate_count: int
    origin_winner_count: int

    cold_candidate_rows: int
    hot_records_scanned: int

    aggregate_below_min_count: int = 0
    retry_quarantined_group_count: int = 0


class ExactEvidenceQueryEngine:
    """
    Exact EvidenceAggregator-compatible fast path.

    It does NOT approximate evidence truth.

    Reduction proof:
    - source reliability is constant within one query;
    - therefore for fixed (origin, polarity, source), only maximum strength can
      win that origin;
    - after that reduction, exact current source reliability chooses the
      original best (origin, polarity) winner;
    - root-lineage max and final noisy-OR are identical to EvidenceAggregator.

    Cache is bounded and disposable. It is performance state, not cognitive
    knowledge.
    """

    def __init__(
        self,
        agent,
        cache_limit: int = 256,
    ):
        self.agent = agent
        self.cache_limit = cache_limit
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def clear(self):
        self._cache.clear()

    def _source_fingerprint(self) -> str:
        payload = repr(
            [
                (
                    name,
                    profile.alpha,
                    profile.beta,
                    profile.observation_consistent_groups,
                    profile.observation_conflicting_groups,
                    profile.observation_prior_consistency,
                    tuple(
                        sorted(
                            profile.parents
                        )
                    ),
                )
                for name, profile
                in sorted(
                    self.agent.sources.items()
                )
            ]
        ).encode("utf-8")
        return hashlib.sha256(
            payload
        ).hexdigest()

    def _cache_key(
        self,
        claim_id: str,
        context_id: str,
        as_of: int,
    ):
        return (
            claim_id,
            context_id,
            as_of,
            self.agent._evidence_revision,
            self._source_fingerprint(),
            self.agent.evidence_aggregator.accept,
            self.agent.evidence_aggregator.conflict,
            self.agent.evidence_aggregator.min_evidence,
        )

    def _cache_put(
        self,
        key,
        value: IndexedEvidenceAggregate,
    ):
        self._cache[key] = value
        self._cache.move_to_end(key)

        while len(
            self._cache
        ) > self.cache_limit:
            self._cache.popitem(
                last=False
            )

    def aggregate(
        self,
        claim_id: str,
        context_id: str,
        as_of: int,
    ) -> Tuple[
        IndexedEvidenceAggregate,
        bool,
    ]:
        key = self._cache_key(
            claim_id,
            context_id,
            as_of,
        )

        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return cached, True

        self.cache_misses += 1

        cold = (
            self.agent.epistemic_archive
            .reduced_evidence_candidates(
                claim_id,
                context_id,
                as_of,
            )
        )

        # Pre-reduction is exact because source accuracy/stability are constant
        # per source within this query.
        per_source = {}
        source_counts = dict(
            cold["source_counts"]
        )

        for row in cold["candidates"]:
            key_source = (
                (
                    f"retry:{row['retry_group_id']}"
                    if row.get("retry_group_id") is not None
                    else f"origin:{row['origin_id']}"
                ),
                row["polarity"],
                row["source"],
            )
            per_source[key_source] = row

        hot_total = 0
        hot_in_scope = 0
        hot_records_scanned = 0
        hot_order_base = cold["max_archive_seq"]

        for index, evidence in enumerate(
            self.agent.evidence_pool
        ):
            if evidence.claim_id != claim_id:
                continue

            hot_total += 1
            hot_records_scanned += 1

            in_scope, _ = evidence.applies_to(
                context_id=context_id,
                as_of=as_of,
            )
            if not in_scope:
                continue

            hot_in_scope += 1
            source_counts[evidence.source] = (
                source_counts.get(
                    evidence.source,
                    0,
                )
                + 1
            )

            row = {
                "order": hot_order_base + index + 1,
                "evidence_id": evidence.evidence_id,
                "source": evidence.source,
                "origin_id": evidence.origin_id,
                "polarity": evidence.polarity,
                "strength": evidence.strength,
                "observation_quality": evidence.observation_quality,
                "retry_group_id": evidence.retry_group_id,
            }

            key_source = (
                (
                    f"retry:{evidence.retry_group_id}"
                    if evidence.retry_group_id is not None
                    else f"origin:{evidence.origin_id}"
                ),
                evidence.polarity,
                evidence.source,
            )
            current = per_source.get(key_source)
            base_value = (
                row["strength"]
                * row["observation_quality"]
            )
            current_base = (
                None
                if current is None
                else (
                    current["strength"]
                    * current.get(
                        "observation_quality",
                        1.0,
                    )
                )
            )

            if (
                current is None
                or base_value > current_base
                or (
                    base_value == current_base
                    and row["order"] < current["order"]
                )
            ):
                per_source[key_source] = row

        total_records = cold["total_records"] + hot_total
        in_scope_records = cold["in_scope_records"] + hot_in_scope

        unknown_source_count = sum(
            count
            for source, count
            in source_counts.items()
            if source not in self.agent.sources
        )

        # Compute exact current effective values first.
        known_rows = []
        for row in per_source.values():
            profile = self.agent.sources.get(
                row["source"]
            )
            if profile is None:
                continue

            value = (
                row["strength"]
                * row.get(
                    "observation_quality",
                    1.0,
                )
                * profile.reliability
                * profile.observation_reliability
            )
            known_rows.append((row, value))

        best_by_origin = {}
        retry_groups = {}

        for row, value in known_rows:
            retry_group_id = row.get(
                "retry_group_id"
            )
            if retry_group_id is not None:
                retry_groups.setdefault(
                    retry_group_id,
                    [],
                ).append((row, value))
                continue

            origin_key = (
                row["origin_id"],
                row["polarity"],
            )
            current = best_by_origin.get(
                origin_key
            )
            if (
                current is None
                or value > current["value"]
                or (
                    value == current["value"]
                    and row["order"]
                    < current["row"]["order"]
                )
            ):
                best_by_origin[origin_key] = {
                    "row": row,
                    "value": value,
                }

        retry_quarantined_group_count = 0
        for retry_group_id, items in retry_groups.items():
            polarities = {
                row["polarity"]
                for row, _ in items
            }
            if len(polarities) > 1:
                retry_quarantined_group_count += 1
                continue

            row, value = max(
                items,
                key=lambda item: (
                    item[1],
                    -item[0]["order"],
                ),
            )
            best_by_origin[
                (
                    f"retry:{retry_group_id}",
                    row["polarity"],
                )
            ] = {
                "row": row,
                "value": value,
            }

        support = {}
        oppose = {}

        for (
            _origin,
            polarity,
        ), winner in best_by_origin.items():
            row = winner["row"]
            value = winner["value"]

            root_dist = (
                self.agent.evidence_aggregator
                .lineage.root_weights(
                    row["source"]
                )
            )
            target = (
                support
                if polarity == 1
                else oppose
            )

            for root, share in root_dist.items():
                effective = value * share
                target[root] = max(
                    target.get(root, 0.0),
                    effective,
                )

        s = (
            1.0
            - math.prod(
                1.0 - x
                for x in support.values()
            )
            if support
            else 0.0
        )
        o = (
            1.0
            - math.prod(
                1.0 - x
                for x in oppose.values()
            )
            if oppose
            else 0.0
        )

        aggregator = self.agent.evidence_aggregator
        below_min_count = 0
        if 0 < s < aggregator.min_evidence:
            below_min_count += 1
        if 0 < o < aggregator.min_evidence:
            below_min_count += 1

        if (
            s < aggregator.min_evidence
            and o < aggregator.min_evidence
        ):
            status = "unresolved"
        elif (
            s >= aggregator.conflict
            and o >= aggregator.conflict
        ):
            status = "conflicted"
        elif (
            s >= aggregator.min_evidence
            and o >= aggregator.conflict
            and o >= s * 0.75
        ):
            status = "conflicted"
        elif (
            s >= aggregator.accept
            and o < aggregator.conflict
        ):
            status = "accepted"
        elif (
            o >= aggregator.accept
            and s < aggregator.conflict
        ):
            status = "rejected"
        else:
            status = "unresolved"

        known_in_scope = (
            in_scope_records
            - unknown_source_count
        )
        origin_winner_count = len(
            best_by_origin
        )
        # Residual count intentionally includes retry-quarantined records;
        # dedicated group count disambiguates that new V2.24 cause.
        duplicate_drop_count = max(
            0,
            known_in_scope
            - origin_winner_count,
        )

        result = IndexedEvidenceAggregate(
            status=status,
            support_score=s,
            oppose_score=o,
            total_records=total_records,
            in_scope_records=in_scope_records,
            out_of_scope_count=(
                total_records
                - in_scope_records
            ),
            unknown_source_count=(
                unknown_source_count
            ),
            duplicate_drop_count=(
                duplicate_drop_count
            ),
            source_candidate_count=len(
                per_source
            ),
            origin_winner_count=(
                origin_winner_count
            ),
            cold_candidate_rows=len(
                cold["candidates"]
            ),
            hot_records_scanned=(
                hot_records_scanned
            ),
            aggregate_below_min_count=(
                below_min_count
            ),
            retry_quarantined_group_count=(
                retry_quarantined_group_count
            ),
        )

        self._cache_put(key, result)
        return result, False

    def state(self) -> Dict:
        return {
            "cache_entries": len(
                self._cache
            ),
            "cache_limit":
                self.cache_limit,
            "cache_hits":
                self.cache_hits,
            "cache_misses":
                self.cache_misses,
        }



_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"
_PICKLE_COMPAT_CLASSES = (
    SourceProfile,
    Evidence,
    BeliefContext,
    BeliefContextManager,
    BeliefShiftCandidate,
    BeliefShiftDecisionRecord,
    ContextualBeliefRevisionPolicy,
    SourceLineage,
    EvidenceAggregator,
    GroundedFact,
    GroundingStore,
    Rule,
    RuleVersionConflict,
    RuleValidator,
    Justification,
    RelevanceSlicer,
    ProvenanceGraph,
    TruthEvaluator,
    EpistemicVerdict,
    AuditReport,
    AdmissionStatus,
    Episode,
    KnowledgeAdmissionPolicy,
    DependencyGraph,
    TruthMaintenanceSystem,
    EvidenceAuditMode,
    IndexedEvidenceAggregate,
    ExactEvidenceQueryEngine,
)
for _cls in _PICKLE_COMPAT_CLASSES:
    _cls.__module__ = _CANONICAL_PICKLE_MODULE
del _cls

__all__ = ['SourceProfile', 'Evidence', 'BeliefContext', 'BeliefContextManager', 'BeliefShiftCandidate', 'BeliefShiftDecisionRecord', 'ContextualBeliefRevisionPolicy', 'SourceLineage', 'EvidenceAggregator', 'GroundedFact', 'GroundingStore', 'Rule', 'RuleVersionConflict', 'RuleValidator', 'Justification', 'RelevanceSlicer', 'ProvenanceGraph', 'TruthEvaluator', 'EpistemicVerdict', 'AuditReport', 'AdmissionStatus', 'Episode', 'KnowledgeAdmissionPolicy', 'DependencyGraph', 'TruthMaintenanceSystem', 'EvidenceAuditMode', 'IndexedEvidenceAggregate', 'ExactEvidenceQueryEngine', 'BeliefShiftDecisionMemory', 'EpisodeMemory']
