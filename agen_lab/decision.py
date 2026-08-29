"""Decision/Q/TD/risk subsystem — physical extraction M5B.

Owns decision records, contextual Q/TD learning, uncertainty-aware policies,
action safety/risk gates, meta-risk, and trajectory risk selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .world_model import OutcomePrediction

@dataclass
class DecisionRecord:
    decision_id: int
    context: str
    candidates: Tuple[str, ...]
    selected_action: str
    policy_scores: Dict[str, float]
    utility_estimates: Dict[str, float]
    epistemic_scores: Dict[str, float]
    exploration_scores: Dict[str, float]
    reward: Optional[float] = None

    # V2.12: epistemic regime/belief context used when deciding.
    belief_context_id: Optional[str] = None

    # V2.6: provenance pemilihan strategi.
    # policy_scores tetap menyimpan skor policy V2.4.
    selection_mode: str = "policy"
    strategy_scores: Optional[Dict[str, float]] = None
    counterfactual_rewards: Optional[Dict[str, float]] = None

    # V2.15 — uncertainty-aware decision provenance.
    risk_mode: Optional[str] = None
    uncertainty_scores: Optional[Dict[str, float]] = None
    blocked_actions: Tuple[str, ...] = ()
    prediction_ids: Optional[Dict[str, int]] = None

    # V2.16 — adaptive risk/meta-decision provenance.
    meta_decision_id: Optional[int] = None

    # V3.1/V3.2 integration — optional provenance; legacy construction stays valid.
    risk_adjusted_scores: Optional[Dict[str, float]] = None
    uncertainty_audit: Optional[Dict[str, Dict]] = None
    trajectory_decision_id: Optional[int] = None
    selected_trajectory_id: Optional[str] = None
    planned_actions: Tuple[str, ...] = ()
    trajectory_failure_bounds: Optional[Tuple[float, float]] = None

    # V2.22 — logical family vs immutable execution identity.
    candidate_action_instances: Optional[Dict[str, str]] = None
    selected_action_family: Optional[str] = None
    selected_action_instance_id: Optional[str] = None

    # V2.25 — raw display context vs canonical learning state identity.
    state_key: Optional[str] = None

    # V2.26 — structured actual outcome provenance.
    objective_outcome: Optional[
        Dict[str, Optional[float]]
    ] = None
    objective_aggregation: Optional[Dict] = None
    objective_profile_signature: Optional[str] = None

    # V2.27 — exact preference identity + scalar learning namespace.
    objective_profile_instance_id: Optional[str] = None
    scalar_state_key: Optional[str] = None


@dataclass
class TransitionRecord:
    transition_id: int
    decision_id: int
    context: str
    action: str
    reward: float
    next_context: Optional[str]
    next_actions: Tuple[str, ...]
    done: bool
    utility_before: float
    utility_after: float
    belief_context_id: Optional[str] = None
    next_belief_context_id: Optional[str] = None

    # V2.22
    action_family: Optional[str] = None
    action_instance_id: Optional[str] = None
    next_action_instance_ids: Tuple[str, ...] = ()

    # V2.25
    state_key: Optional[str] = None
    next_state_key: Optional[str] = None

    # V2.26
    objective_outcome: Optional[
        Dict[str, Optional[float]]
    ] = None
    objective_aggregation: Optional[Dict] = None
    objective_profile_signature: Optional[str] = None

    # V2.27
    objective_profile_instance_id: Optional[str] = None
    scalar_state_key: Optional[str] = None
    next_scalar_state_key: Optional[str] = None


@dataclass
class TrajectoryDecisionRecord:
    """Audit read-only untuk pemilihan rencana multi-langkah."""

    trajectory_decision_id: int
    context: str
    belief_context_id: Optional[str]
    candidate_trajectories: Dict[str, Tuple[str, ...]]
    selected_trajectory_id: str
    selected_actions: Tuple[str, ...]
    selected_first_action: str
    risk_mode: str
    base_scores: Dict[str, float]
    risk_adjusted_scores: Dict[str, float]
    trajectory_audit: Dict[str, Dict]

    # V2.22
    candidate_trajectory_action_instances: Optional[
        Dict[str, Tuple[str, ...]]
    ] = None
    selected_action_instance_ids: Tuple[str, ...] = ()
    selected_first_action_instance_id: Optional[str] = None
    state_key: Optional[str] = None

    # V2.27 — base/risk scores may depend on preference scalarization.
    objective_profile_instance_id: Optional[str] = None
    scalar_state_key: Optional[str] = None


class DecisionPolicy:
    """
    V2.12 — regime-scoped utility learner.

    Two independent stores are maintained:

    Legacy:
        Q(environment_context, action)

    Scoped:
        Q(belief_context_id, environment_context, action)

    Agent-level decisions default to the scoped path. Direct DecisionPolicy
    calls without belief_context_id preserve backward compatibility.

    A belief-context shift therefore does NOT erase old utility and does NOT
    silently reuse old Q-values in the new epistemic regime.
    """

    def __init__(
        self,
        learning_rate: float = 0.30,
        utility_weight: float = 0.65,
        epistemic_weight: float = 0.25,
        exploration_weight: float = 0.10,
        initial_utility: float = 0.50,
        discount_factor: float = 0.90,
    ):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate harus di (0,1]")
        if not 0.0 <= discount_factor < 1.0:
            raise ValueError("discount_factor harus di [0,1)")

        weights = (
            utility_weight
            + epistemic_weight
            + exploration_weight
        )
        if abs(weights - 1.0) > 1e-9:
            raise ValueError(
                "Bobot decision policy harus berjumlah 1.0"
            )

        self.learning_rate = learning_rate
        self.utility_weight = utility_weight
        self.epistemic_weight = epistemic_weight
        self.exploration_weight = exploration_weight
        self.initial_utility = initial_utility
        self.discount_factor = discount_factor

        # Backward-compatible unscoped store.
        self.q_values: Dict[Tuple, float] = {}
        self.counts: Dict[Tuple, int] = {}

        # V2.12 scoped store.
        self.scoped_q_values: Dict[
            Tuple[str, str, str],
            float,
        ] = {}
        self.scoped_counts: Dict[
            Tuple[str, str, str],
            int,
        ] = {}

    def utility(
        self,
        context: str,
        action: str,
        belief_context_id: Optional[str] = None,
    ) -> float:
        if belief_context_id is None:
            return self.q_values.get(
                (context, action),
                self.initial_utility,
            )

        key = (belief_context_id, context, action)
        return self.scoped_q_values.get(
            key,
            self.q_values.get(key, self.initial_utility),
        )

    def count(
        self,
        context: str,
        action: str,
        belief_context_id: Optional[str] = None,
    ) -> int:
        if belief_context_id is None:
            return self.counts.get(
                (context, action),
                0,
            )

        key = (belief_context_id, context, action)
        return self.scoped_counts.get(
            key,
            self.counts.get(key, 0),
        )

    def exploration_bonus(
        self,
        context: str,
        action: str,
        candidate_count: int,
        belief_context_id: Optional[str] = None,
    ) -> float:
        n = self.count(
            context,
            action,
            belief_context_id=belief_context_id,
        )
        return 1.0 / math.sqrt(n + 1.0)

    def _store_update(
        self,
        context: str,
        action: str,
        value: float,
        belief_context_id: Optional[str],
    ):
        if belief_context_id is None:
            key = (context, action)
            self.q_values[key] = value
            self.counts[key] = (
                self.counts.get(key, 0) + 1
            )
            return

        key = (
            belief_context_id,
            context,
            action,
        )
        self.scoped_q_values[key] = value
        self.scoped_counts[key] = (
            self.scoped_counts.get(key, 0) + 1
        )
        # V3.0 unified audit view. The canonical scoped stores remain the
        # authoritative V2.21 representation and are not removed.
        self.q_values[key] = value
        self.counts[key] = self.scoped_counts[key]

    def update_transition(
        self,
        context: str,
        action: str,
        reward: float,
        next_context: Optional[str],
        next_actions: Optional[List[str]],
        done: bool,
        belief_context_id: Optional[str] = None,
        next_belief_context_id: Optional[str] = None,
    ) -> float:
        """
        Bounded TD backup.

        If next_belief_context_id is omitted, the transition stays in the
        decision's belief scope. A caller may explicitly supply a different
        next scope for a genuine cross-regime transition.
        """
        if not 0.0 <= reward <= 1.0:
            raise ValueError(
                "reward harus di rentang 0..1"
            )

        old = self.utility(
            context,
            action,
            belief_context_id=belief_context_id,
        )

        if done:
            target = reward
        else:
            resolved_next_scope = (
                belief_context_id
                if next_belief_context_id is None
                else next_belief_context_id
            )

            if next_context is None or not next_actions:
                next_value = self.initial_utility
            else:
                next_value = max(
                    self.utility(
                        next_context,
                        a,
                        belief_context_id=resolved_next_scope,
                    )
                    for a in next_actions
                )

            gamma = self.discount_factor
            target = (
                (1.0 - gamma) * reward
                + gamma * next_value
            )

        new = (
            old
            + self.learning_rate
            * (target - old)
        )
        new = max(0.0, min(1.0, new))

        self._store_update(
            context,
            action,
            new,
            belief_context_id,
        )
        return new

    def score_actions(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[
            Dict[str, float]
        ] = None,
        belief_context_id: Optional[str] = None,
    ):
        epistemic_scores = epistemic_scores or {}

        policy_scores = {}
        utilities = {}
        epistemics = {}
        explorations = {}

        for action in candidates:
            utility = self.utility(
                context,
                action,
                belief_context_id=belief_context_id,
            )
            epistemic = max(
                0.0,
                min(
                    1.0,
                    epistemic_scores.get(
                        action,
                        0.0,
                    ),
                ),
            )
            exploration = self.exploration_bonus(
                context,
                action,
                len(candidates),
                belief_context_id=belief_context_id,
            )

            total = (
                self.utility_weight * utility
                + self.epistemic_weight * epistemic
                + self.exploration_weight
                * exploration
            )

            utilities[action] = utility
            epistemics[action] = epistemic
            explorations[action] = exploration
            policy_scores[action] = total

        return (
            policy_scores,
            utilities,
            epistemics,
            explorations,
        )

    def select(
        self,
        context: str,
        candidates: List[str],
        epistemic_scores: Optional[
            Dict[str, float]
        ] = None,
        belief_context_id: Optional[str] = None,
    ):
        if not candidates:
            raise ValueError(
                "Candidates tidak boleh kosong"
            )

        candidates = sorted(set(candidates))

        (
            scores,
            utilities,
            epistemics,
            explorations,
        ) = self.score_actions(
            context,
            candidates,
            epistemic_scores,
            belief_context_id=belief_context_id,
        )

        selected = max(
            candidates,
            key=lambda action: (
                scores[action],
                action,
            ),
        )

        return (
            selected,
            scores,
            utilities,
            epistemics,
            explorations,
        )

    def update(
        self,
        context: str,
        action: str,
        reward: float,
        belief_context_id: Optional[str] = None,
    ) -> float:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(
                "reward harus di rentang 0..1"
            )

        old = self.utility(
            context,
            action,
            belief_context_id=belief_context_id,
        )
        new = (
            old
            + self.learning_rate
            * (reward - old)
        )

        self._store_update(
            context,
            action,
            new,
            belief_context_id,
        )
        return new

    def scoped_state(
        self,
        belief_context_id: str,
        context: Optional[str] = None,
    ) -> Dict:
        q_values = {}
        counts = {}

        for (
            scope,
            env_context,
            action,
        ), value in self.scoped_q_values.items():
            if scope != belief_context_id:
                continue
            if (
                context is not None
                and env_context != context
            ):
                continue

            key = (
                (env_context, action)
                if context is None
                else action
            )
            q_values[key] = value

        for (
            scope,
            env_context,
            action,
        ), value in self.scoped_counts.items():
            if scope != belief_context_id:
                continue
            if (
                context is not None
                and env_context != context
            ):
                continue

            key = (
                (env_context, action)
                if context is None
                else action
            )
            counts[key] = value

        return {
            "belief_context_id": belief_context_id,
            "context": context,
            "q_values": q_values,
            "counts": counts,
        }


class UncertaintyDecisionMode(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class UncertaintyRiskProfile:
    """
    Decision-layer policy only.

    This profile does NOT modify truth/evidence and does NOT modify the
    statistical world model. It only decides how a forecast is consumed.

    known_bad gate:
        If enough evidence exists and even the UPPER confidence bound of
        success is below known_bad_success_upper, an action is excluded when
        another non-blocked candidate exists.

    This prevents uncertainty bonus from rewarding actions already strongly
    demonstrated to fail.
    """
    mode: UncertaintyDecisionMode = UncertaintyDecisionMode.BALANCED

    prediction_weight: float = 0.30
    epistemic_penalty: float = 0.20
    aleatoric_penalty: float = 0.10
    information_bonus: float = 0.12

    minimum_known_bad_samples: int = 10
    known_bad_success_upper: float = 0.20

    # Default semantic:
    # reward/utility = objective
    # success probability = constraint / known-bad signal
    #
    # Domains where "success" itself is the true objective may explicitly
    # choose a non-zero success_component_weight.
    reward_component_weight: float = 1.00
    success_component_weight: float = 0.00

    def __post_init__(self):
        bounded = {
            "prediction_weight": self.prediction_weight,
            "epistemic_penalty": self.epistemic_penalty,
            "aleatoric_penalty": self.aleatoric_penalty,
            "information_bonus": self.information_bonus,
            "known_bad_success_upper": self.known_bad_success_upper,
            "reward_component_weight": self.reward_component_weight,
            "success_component_weight": self.success_component_weight,
        }
        for name, value in bounded.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} harus di rentang 0..1"
                )

        if self.minimum_known_bad_samples < 1:
            raise ValueError(
                "minimum_known_bad_samples harus >= 1"
            )

        total = (
            self.reward_component_weight
            + self.success_component_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "reward_component_weight + success_component_weight "
                "harus 1.0"
            )


@dataclass
class UncertaintyDecisionResult:
    selected_action: str
    mode: UncertaintyDecisionMode
    belief_context_id: str
    base_scores: Dict[str, float]
    decision_scores: Dict[str, float]
    outcome_scores: Dict[str, float]
    uncertainty_scores: Dict[str, float]
    blocked_actions: Tuple[str, ...]
    predictions: Dict[str, OutcomePrediction]
    forced_fallback: bool = False


@dataclass(frozen=True)
class MetaRiskSignals:
    """
    Inputs visible to the meta-policy.

    None of these fields expose a hidden environment regime:
    - novelty comes from scoped sample support
    - uncertainty comes from prediction intervals
    - reliability comes from scoped calibration
    - failure consequence is declared by the domain adapter
    """
    belief_context_id: str
    context: str
    candidate_count: int
    relevant_candidate_count: int
    relevant_actions: Tuple[str, ...]

    novelty_fraction: float
    mean_uncertainty: float
    max_uncertainty: float
    min_model_reliability: float

    mean_failure_consequence: float
    max_failure_consequence: float


@dataclass
class MetaRiskDecision:
    meta_decision_id: int
    belief_context_id: str
    context: str
    selected_mode: UncertaintyDecisionMode
    signals: MetaRiskSignals
    reason: str
    candidate_actions: Tuple[str, ...]
    candidate_action_instances: Optional[Dict[str, str]] = None
    state_key: Optional[str] = None

    # V2.27
    objective_profile_instance_id: Optional[str] = None
    scalar_state_key: Optional[str] = None


class AdaptiveRiskModePolicy:
    """
    Deterministic, auditable meta-policy.

    Rules:
    1. High consequence + high uncertainty/poor calibration => CONSERVATIVE.
    2. High novelty/uncertainty + low consequence => EXPLORATORY.
    3. Otherwise => BALANCED.

    This policy does not learn a hidden mode label. It responds only to
    observable cognitive state and adapter-declared failure consequence.
    """

    def __init__(
        self,
        high_consequence_threshold: float = 0.70,
        low_consequence_threshold: float = 0.30,
        high_uncertainty_threshold: float = 0.55,
        conservative_uncertainty_threshold: float = 0.45,
        low_reliability_threshold: float = 0.35,
        high_novelty_threshold: float = 0.50,
        relevance_margin: float = 0.12,
    ):
        values = {
            "high_consequence_threshold":
                high_consequence_threshold,
            "low_consequence_threshold":
                low_consequence_threshold,
            "high_uncertainty_threshold":
                high_uncertainty_threshold,
            "conservative_uncertainty_threshold":
                conservative_uncertainty_threshold,
            "low_reliability_threshold":
                low_reliability_threshold,
            "high_novelty_threshold":
                high_novelty_threshold,
            "relevance_margin":
                relevance_margin,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} harus 0..1"
                )

        self.high_consequence_threshold = (
            high_consequence_threshold
        )
        self.low_consequence_threshold = (
            low_consequence_threshold
        )
        self.high_uncertainty_threshold = (
            high_uncertainty_threshold
        )
        self.conservative_uncertainty_threshold = (
            conservative_uncertainty_threshold
        )
        self.low_reliability_threshold = (
            low_reliability_threshold
        )
        self.high_novelty_threshold = (
            high_novelty_threshold
        )
        self.relevance_margin = relevance_margin

    def _prediction_uncertainty(
        self,
        prediction: OutcomePrediction,
    ) -> float:
        u = prediction.uncertainty
        if u is None:
            return 1.0

        epistemic = (
            0.5 * u.reward_interval_width
            + 0.5 * u.success_interval_width
        )
        aleatoric = (
            u.reward_aleatoric_std
            if u.reward_aleatoric_std is not None
            else 0.0
        )

        return max(
            0.0,
            min(
                1.0,
                0.85 * epistemic
                + 0.15 * aleatoric,
            ),
        )

    def signals(
        self,
        context: str,
        predictions: Dict[str, OutcomePrediction],
        failure_consequences: Dict[str, float],
        belief_context_id: str,
        base_scores: Optional[Dict[str, float]] = None,
    ) -> MetaRiskSignals:
        if not predictions:
            raise ValueError(
                "predictions tidak boleh kosong"
            )
        if set(predictions) != set(failure_consequences):
            raise ValueError(
                "failure_consequences harus mencakup semua kandidat"
            )

        if base_scores is None:
            relevant_actions = sorted(
                predictions
            )
        else:
            if set(base_scores) != set(predictions):
                raise ValueError(
                    "base_scores harus mencakup semua predictions"
                )

            best_base = max(
                base_scores.values()
            )
            relevant_actions = sorted(
                action
                for action, score
                in base_scores.items()
                if score
                >= best_base - self.relevance_margin
            )

        if not relevant_actions:
            # Defensive only; max candidate should always survive.
            relevant_actions = [
                max(
                    predictions,
                    key=lambda action: (
                        base_scores[action]
                        if base_scores is not None
                        else 0.0,
                        action,
                    ),
                )
            ]

        uncertainty_values = [
            self._prediction_uncertainty(
                predictions[action]
            )
            for action in relevant_actions
        ]

        novelty_flags = [
            1.0
            if (
                predictions[action].uncertainty is None
                or predictions[
                    action
                ].uncertainty.insufficient_data
            )
            else 0.0
            for action in relevant_actions
        ]

        reliability_values = [
            max(
                0.0,
                min(
                    1.0,
                    predictions[
                        action
                    ].model_reliability,
                ),
            )
            for action in relevant_actions
        ]

        consequences = [
            max(
                0.0,
                min(
                    1.0,
                    failure_consequences[action],
                ),
            )
            for action in relevant_actions
        ]

        n = len(predictions)
        relevant_n = len(
            relevant_actions
        )

        return MetaRiskSignals(
            belief_context_id=belief_context_id,
            context=context,
            candidate_count=n,
            relevant_candidate_count=relevant_n,
            relevant_actions=tuple(
                relevant_actions
            ),
            novelty_fraction=(
                sum(novelty_flags)
                / relevant_n
            ),
            mean_uncertainty=(
                sum(uncertainty_values)
                / relevant_n
            ),
            max_uncertainty=max(
                uncertainty_values
            ),
            min_model_reliability=min(
                reliability_values
            ),
            mean_failure_consequence=(
                sum(consequences)
                / relevant_n
            ),
            max_failure_consequence=max(
                consequences
            ),
        )

    def select_mode(
        self,
        signals: MetaRiskSignals,
    ) -> Tuple[UncertaintyDecisionMode, str]:
        high_consequence = (
            signals.max_failure_consequence
            >= self.high_consequence_threshold
        )

        if (
            high_consequence
            and (
                signals.mean_uncertainty
                >= self.conservative_uncertainty_threshold
                or signals.min_model_reliability
                <= self.low_reliability_threshold
            )
        ):
            return (
                UncertaintyDecisionMode.CONSERVATIVE,
                "high consequence with unresolved prediction risk",
            )

        low_consequence = (
            signals.max_failure_consequence
            <= self.low_consequence_threshold
        )

        if (
            low_consequence
            and (
                signals.novelty_fraction
                >= self.high_novelty_threshold
                or signals.mean_uncertainty
                >= self.high_uncertainty_threshold
            )
        ):
            return (
                UncertaintyDecisionMode.EXPLORATORY,
                "low consequence with high novelty/uncertainty",
            )

        return (
            UncertaintyDecisionMode.BALANCED,
            "default balance: risk is neither high-uncertain nor low-cost novel",
        )

    def profile_for_mode(
        self,
        mode: UncertaintyDecisionMode,
    ) -> UncertaintyRiskProfile:
        if mode == UncertaintyDecisionMode.CONSERVATIVE:
            return UncertaintyRiskProfile(
                mode=mode,
                prediction_weight=0.35,
                epistemic_penalty=0.30,
                aleatoric_penalty=0.15,
                information_bonus=0.0,
            )

        if mode == UncertaintyDecisionMode.EXPLORATORY:
            return UncertaintyRiskProfile(
                mode=mode,
                prediction_weight=0.30,
                epistemic_penalty=0.05,
                aleatoric_penalty=0.05,
                information_bonus=0.18,
            )

        return UncertaintyRiskProfile(
            mode=UncertaintyDecisionMode.BALANCED,
        )


class UncertaintyAwareDecisionPolicy:
    """
    Consumes OutcomePrediction without retraining it.

    CONSERVATIVE:
        uses lower confidence bounds of the configured objective.

    BALANCED:
        uses point estimate minus epistemic/aleatoric penalties.

    By default the configured objective is reward only. Success probability
    is used as a failure constraint, not positive utility. This prevents a
    valid no-op (for example WAIT) from looking valuable merely because it
    executes successfully.

    EXPLORATORY:
        may add a bounded information bonus for poorly-known actions,
        but the known-bad gate is applied first.

    No mode treats uncertainty itself as evidence of high utility.
    """
    def _uncertainty_score(
        self,
        prediction: OutcomePrediction,
    ) -> float:
        u = prediction.uncertainty
        if u is None:
            return 1.0

        epistemic = (
            0.5 * u.reward_interval_width
            + 0.5 * u.success_interval_width
        )
        aleatoric = (
            u.reward_aleatoric_std
            if u.reward_aleatoric_std is not None
            else 0.0
        )
        return max(
            0.0,
            min(
                1.0,
                0.80 * epistemic
                + 0.20 * aleatoric,
            ),
        )

    def _known_bad(
        self,
        prediction: OutcomePrediction,
        profile: UncertaintyRiskProfile,
    ) -> bool:
        u = prediction.uncertainty
        if u is None:
            return False

        return (
            (
                prediction.success_sample_count
                if hasattr(
                    prediction,
                    "success_sample_count",
                )
                else prediction.sample_count
            )
            >= profile.minimum_known_bad_samples
            and u.success_upper
            < profile.known_bad_success_upper
        )

    def _outcome_score(
        self,
        prediction: OutcomePrediction,
        profile: UncertaintyRiskProfile,
    ) -> float:
        u = prediction.uncertainty

        if u is None:
            reward_lower = prediction.predicted_reward
            success_lower = (
                prediction.predicted_success_probability
            )
            epistemic = 1.0
            aleatoric = 0.0
        else:
            reward_lower = u.reward_lower
            success_lower = u.success_lower
            epistemic = (
                0.5 * u.reward_interval_width
                + 0.5 * u.success_interval_width
            )
            aleatoric = (
                u.reward_aleatoric_std
                if u.reward_aleatoric_std is not None
                else 0.0
            )

        mean_outcome = (
            profile.reward_component_weight
            * prediction.predicted_reward
            + profile.success_component_weight
            * prediction.predicted_success_probability
        )

        lower_bound_outcome = (
            profile.reward_component_weight
            * reward_lower
            + profile.success_component_weight
            * success_lower
        )

        if profile.mode == UncertaintyDecisionMode.CONSERVATIVE:
            raw = (
                lower_bound_outcome
                - profile.aleatoric_penalty
                * aleatoric
            )

        elif profile.mode == UncertaintyDecisionMode.BALANCED:
            raw = (
                mean_outcome
                - profile.epistemic_penalty
                * epistemic
                - profile.aleatoric_penalty
                * aleatoric
            )

        else:
            # Exploration bonus is allowed only after the known-bad gate.
            raw = (
                mean_outcome
                - 0.5 * profile.aleatoric_penalty
                * aleatoric
                + profile.information_bonus
                * epistemic
            )

        return max(0.0, min(1.0, raw))

    def select(
        self,
        base_scores: Dict[str, float],
        predictions: Dict[str, OutcomePrediction],
        profile: UncertaintyRiskProfile,
    ) -> UncertaintyDecisionResult:
        if not base_scores:
            raise ValueError(
                "base_scores tidak boleh kosong"
            )

        if set(base_scores) != set(predictions):
            raise ValueError(
                "base_scores dan predictions harus memiliki kandidat sama"
            )

        candidates = sorted(base_scores)

        blocked = {
            action
            for action in candidates
            if self._known_bad(
                predictions[action],
                profile,
            )
        }

        allowed = [
            action
            for action in candidates
            if action not in blocked
        ]

        forced_fallback = False
        if not allowed:
            # Never produce an impossible decision set. If every action is
            # known-bad, choose the least-bad option and expose the fallback.
            allowed = list(candidates)
            forced_fallback = True

        outcome_scores = {
            action: self._outcome_score(
                predictions[action],
                profile,
            )
            for action in candidates
        }

        uncertainty_scores = {
            action: self._uncertainty_score(
                predictions[action]
            )
            for action in candidates
        }

        decision_scores = {}
        for action in candidates:
            reliability = max(
                0.0,
                min(
                    1.0,
                    predictions[action].model_reliability,
                ),
            )
            effective_prediction_weight = (
                profile.prediction_weight
                * reliability
            )

            score = (
                (1.0 - effective_prediction_weight)
                * base_scores[action]
                + effective_prediction_weight
                * outcome_scores[action]
            )

            if action in blocked and not forced_fallback:
                score = -1.0

            decision_scores[action] = score

        selected = max(
            allowed,
            key=lambda action: (
                decision_scores[action],
                action,
            ),
        )

        scope = predictions[selected].belief_context_id
        if scope is None:
            scope = "legacy"

        return UncertaintyDecisionResult(
            selected_action=selected,
            mode=profile.mode,
            belief_context_id=scope,
            base_scores=dict(base_scores),
            decision_scores=decision_scores,
            outcome_scores=outcome_scores,
            uncertainty_scores=uncertainty_scores,
            blocked_actions=tuple(sorted(blocked)),
            predictions=predictions,
            forced_fallback=forced_fallback,
        )


LegacyUncertaintyAwareDecisionPolicy = UncertaintyAwareDecisionPolicy


class RiskMode(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    EXPLORATORY = "EXPLORATORY"


@dataclass(frozen=True)
class ActionRiskEstimate:
    action: str
    predicted_reward: float
    success_probability: float
    success_lower: float
    success_upper: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    state_uncertainty: float
    sample_count: int
    failure_consequence: float = 1.0
    information_gain: float = 0.0
    action_instance_id: Optional[str] = None
    state_key: Optional[str] = None

    # V2.27 — predicted_reward is preference-profile dependent.
    objective_profile_instance_id: Optional[str] = None

    # V2.29 — explicit epistemic-scope provenance for preference-aware risk.
    # Technical success statistics remain profile-independent, but a bound
    # estimate must still name the belief context/state/action instance it was
    # computed from so policy-layer mixing fails safe.
    belief_context_id: Optional[str] = None

    def __post_init__(self):
        bounded = (
            self.predicted_reward,
            self.success_probability,
            self.success_lower,
            self.success_upper,
            self.epistemic_uncertainty,
            self.aleatoric_uncertainty,
            self.state_uncertainty,
            self.failure_consequence,
            self.information_gain,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("ActionRiskEstimate values harus di [0,1]")
        if self.success_lower > self.success_upper:
            raise ValueError("success_lower tidak boleh > success_upper")
        if self.sample_count < 0:
            raise ValueError("sample_count tidak boleh negatif")


@dataclass(frozen=True)
class SafetyGateAssessment:
    action: str
    allowed: bool
    certified_safe: bool
    known_bad: bool
    uncertain: bool
    lower_failure_probability: float
    upper_failure_probability: float
    reason: str


class ChanceConstrainedSafetyGate:
    """Conformal chance constraint with an explicit known-bad gate."""

    def __init__(
        self,
        max_failure_probability: float = 0.25,
        min_samples: int = 3,
    ):
        if not 0.0 < max_failure_probability < 1.0:
            raise ValueError("max_failure_probability harus di (0,1)")
        if min_samples < 1:
            raise ValueError("min_samples minimal 1")
        self.max_failure_probability = max_failure_probability
        self.min_samples = min_samples

    def assess(
        self,
        estimate: ActionRiskEstimate,
        mode: RiskMode,
    ) -> SafetyGateAssessment:
        lower_failure = 1.0 - estimate.success_upper
        upper_failure = 1.0 - estimate.success_lower
        enough = estimate.sample_count >= self.min_samples

        certified_safe = (
            enough
            and upper_failure <= self.max_failure_probability
        )
        known_bad = (
            enough
            and lower_failure > self.max_failure_probability
        )
        uncertain = not certified_safe and not known_bad

        if known_bad:
            allowed = False
            reason = "KNOWN_BAD"
        elif mode == RiskMode.CONSERVATIVE:
            allowed = certified_safe
            reason = (
                "CERTIFIED_SAFE"
                if certified_safe
                else "NOT_CERTIFIED"
            )
        else:
            allowed = True
            reason = (
                "CERTIFIED_SAFE"
                if certified_safe
                else "UNCERTAIN_ALLOWED"
            )

        return SafetyGateAssessment(
            action=estimate.action,
            allowed=allowed,
            certified_safe=certified_safe,
            known_bad=known_bad,
            uncertain=uncertain,
            lower_failure_probability=lower_failure,
            upper_failure_probability=upper_failure,
            reason=reason,
        )


class MetaRiskPolicy:
    """Selects risk mode from decision-relevant uncertainty only."""

    def __init__(
        self,
        competitive_margin: float = 0.03,
    ):
        if not 0.0 <= competitive_margin <= 1.0:
            raise ValueError("competitive_margin harus di [0,1]")
        self.competitive_margin = competitive_margin

    def choose(
        self,
        base_scores: Dict[str, float],
        estimates: Dict[str, ActionRiskEstimate],
        model_reliability: float,
        requested_mode: Optional[RiskMode] = None,
    ) -> Dict:
        if not base_scores:
            raise ValueError("base_scores tidak boleh kosong")
        best = max(base_scores.values())
        relevant_actions = sorted(
            action
            for action, score in base_scores.items()
            if score >= best - self.competitive_margin
        )
        relevant = [estimates[action] for action in relevant_actions]

        novelty = max(
            (item.epistemic_uncertainty for item in relevant),
            default=0.0,
        )
        uncertainty = max(
            (
                0.45 * item.epistemic_uncertainty
                + 0.25 * item.aleatoric_uncertainty
                + 0.30 * item.state_uncertainty
                for item in relevant
            ),
            default=0.0,
        )
        consequence = max(
            (item.failure_consequence for item in relevant),
            default=0.0,
        )
        reliability = max(0.0, min(1.0, model_reliability))

        if requested_mode is not None:
            mode = requested_mode
            reason = "REQUESTED"
        elif (
            consequence >= 0.75
            and (uncertainty >= 0.45 or reliability < 0.75)
        ):
            mode = RiskMode.CONSERVATIVE
            reason = "HIGH_CONSEQUENCE_UNCERTAINTY"
        elif (
            novelty >= 0.25
            and consequence < 0.50
            and reliability >= 0.50
        ):
            mode = RiskMode.EXPLORATORY
            reason = "SAFE_INFORMATION_GAIN"
        else:
            mode = RiskMode.BALANCED
            reason = "BALANCED_DEFAULT"

        return {
            "mode": mode,
            "reason": reason,
            "relevant_actions": relevant_actions,
            "decision_relevant_novelty": novelty,
            "decision_relevant_uncertainty": uncertainty,
            "max_failure_consequence": consequence,
            "model_reliability": reliability,
        }


class UncertaintyAwareDecisionPolicy(LegacyUncertaintyAwareDecisionPolicy):
    """
    Risk-adjusted ranking plus conformal chance gate.

    The policy is read-only with respect to Q, ensemble, calibrator, evidence,
    and source reliability. It ranks estimates; actual outcome learning stays
    in the caller's explicit update path.
    """

    def __init__(
        self,
        safety_gate: Optional[ChanceConstrainedSafetyGate] = None,
        meta_policy: Optional[MetaRiskPolicy] = None,
    ):
        self.safety_gate = (
            safety_gate
            if safety_gate is not None
            else ChanceConstrainedSafetyGate()
        )
        self.meta_policy = (
            meta_policy
            if meta_policy is not None
            else MetaRiskPolicy()
        )

    def rank(
        self,
        base_scores: Dict[str, float],
        estimates: Dict[str, ActionRiskEstimate],
        model_reliability: float = 1.0,
        requested_mode: Optional[RiskMode] = None,
    ) -> Dict:
        if set(base_scores) != set(estimates):
            raise ValueError(
                "base_scores dan estimates harus memiliki action yang sama"
            )

        meta = self.meta_policy.choose(
            base_scores,
            estimates,
            model_reliability,
            requested_mode=requested_mode,
        )
        mode = meta["mode"]
        assessments = {
            action: self.safety_gate.assess(estimate, mode)
            for action, estimate in estimates.items()
        }

        mode_weights = {
            RiskMode.CONSERVATIVE: (0.75, 0.30, 0.00),
            RiskMode.BALANCED: (0.50, 0.18, 0.05),
            RiskMode.EXPLORATORY: (0.25, 0.08, 0.20),
        }
        risk_weight, uncertainty_weight, information_weight = (
            mode_weights[mode]
        )

        adjusted_scores = {}
        audit = {}
        for action, estimate in estimates.items():
            gate = assessments[action]
            combined_uncertainty = (
                0.45 * estimate.epistemic_uncertainty
                + 0.25 * estimate.aleatoric_uncertainty
                + 0.30 * estimate.state_uncertainty
            )
            risk_exposure = (
                gate.upper_failure_probability
                * estimate.failure_consequence
            )
            information_bonus = (
                estimate.information_gain
                * (1.0 - estimate.failure_consequence)
            )
            evidence_factor = min(
                1.0,
                estimate.sample_count
                / self.safety_gate.min_samples,
            )
            prediction_contribution = (
                0.20
                * max(0.0, min(1.0, model_reliability))
                * evidence_factor
                * estimate.predicted_reward
            )
            score = (
                base_scores[action]
                + prediction_contribution
                - risk_weight * risk_exposure
                - uncertainty_weight * combined_uncertainty
                + information_weight * information_bonus
            )
            adjusted_scores[action] = score
            audit[action] = {
                "allowed": gate.allowed,
                "gate_reason": gate.reason,
                "certified_safe": gate.certified_safe,
                "known_bad": gate.known_bad,
                "uncertain": gate.uncertain,
                "success_probability": estimate.success_probability,
                "success_interval": (
                    estimate.success_lower,
                    estimate.success_upper,
                ),
                "lower_failure_probability": (
                    gate.lower_failure_probability
                ),
                "upper_failure_probability": (
                    gate.upper_failure_probability
                ),
                "epistemic_uncertainty": (
                    estimate.epistemic_uncertainty
                ),
                "aleatoric_uncertainty": (
                    estimate.aleatoric_uncertainty
                ),
                "state_uncertainty": estimate.state_uncertainty,
                "combined_uncertainty": combined_uncertainty,
                "failure_consequence": estimate.failure_consequence,
                "risk_exposure": risk_exposure,
                "information_gain": estimate.information_gain,
                "sample_count": estimate.sample_count,
                "prediction_evidence_factor": evidence_factor,
                "prediction_contribution": prediction_contribution,
                "base_score": base_scores[action],
                "risk_adjusted_score": score,
            }

        eligible = sorted(
            action
            for action, gate in assessments.items()
            if gate.allowed
        )
        selected = (
            max(
                eligible,
                key=lambda action: (
                    adjusted_scores[action],
                    action,
                ),
            )
            if eligible
            else None
        )

        return {
            "selected_action": selected,
            "abstained": selected is None,
            "risk_mode": mode,
            "risk_mode_reason": meta["reason"],
            "relevant_actions": meta["relevant_actions"],
            "decision_relevant_novelty": (
                meta["decision_relevant_novelty"]
            ),
            "decision_relevant_uncertainty": (
                meta["decision_relevant_uncertainty"]
            ),
            "eligible_actions": eligible,
            "blocked_actions": sorted(
                action
                for action, gate in assessments.items()
                if not gate.allowed
            ),
            "risk_adjusted_scores": adjusted_scores,
            "action_audit": audit,
        }


@dataclass(frozen=True)
class TrajectoryRiskEstimate:
    """
    Calibrated step estimates for one candidate trajectory.

    The trajectory is conditional on executing the steps in order. The core
    never invents future states; a domain adapter must construct each step.
    """

    trajectory_id: str
    steps: Tuple[ActionRiskEstimate, ...]

    def __post_init__(self):
        if not self.trajectory_id:
            raise ValueError("trajectory_id tidak boleh kosong")
        if not self.steps:
            raise ValueError("Trajectory harus memiliki minimal satu step")

    @property
    def actions(self) -> Tuple[str, ...]:
        return tuple(step.action for step in self.steps)

    @property
    def horizon(self) -> int:
        return len(self.steps)

    @property
    def product_success_estimate(self) -> float:
        """Diagnostic product estimate; never used as a safety certificate."""
        value = 1.0
        for step in self.steps:
            value *= step.success_probability
        return max(0.0, min(1.0, value))

    @property
    def mean_combined_uncertainty(self) -> float:
        values = [
            0.45 * step.epistemic_uncertainty
            + 0.25 * step.aleatoric_uncertainty
            + 0.30 * step.state_uncertainty
            for step in self.steps
        ]
        return sum(values) / len(values)

    @property
    def max_failure_consequence(self) -> float:
        return max(step.failure_consequence for step in self.steps)

    @property
    def mean_information_gain(self) -> float:
        return sum(
            step.information_gain for step in self.steps
        ) / len(self.steps)

    @property
    def min_sample_count(self) -> int:
        return min(step.sample_count for step in self.steps)


@dataclass(frozen=True)
class TrajectorySafetyAssessment:
    trajectory_id: str
    allowed: bool
    certified_safe: bool
    known_bad: bool
    uncertain: bool
    lower_failure_probability: float
    upper_failure_probability: float
    product_success_estimate: float
    risk_budget: float
    risk_budget_remaining: float
    risk_budget_excess: float
    bottleneck_step_index: int
    bottleneck_action: str
    step_assessments: Tuple[SafetyGateAssessment, ...]
    reason: str


class TrajectoryChanceConstrainedSafetyGate:
    """
    Multi-step chance constraint without an independence assumption.

    The upper probability of any failure uses Boole's inequality:

        P(any failure) <= min(1, sum_i P(failure_i))

    The lower bound uses max_i P(failure_i). The product of point-success
    predictions is retained only as a diagnostic/ranking feature and is never
    accepted as a safety certificate.
    """

    def __init__(
        self,
        max_trajectory_failure_probability: float = 0.25,
        action_gate: Optional[ChanceConstrainedSafetyGate] = None,
        max_horizon: int = 32,
    ):
        if not 0.0 < max_trajectory_failure_probability < 1.0:
            raise ValueError(
                "max_trajectory_failure_probability harus di (0,1)"
            )
        if max_horizon < 1:
            raise ValueError("max_horizon minimal 1")
        self.max_trajectory_failure_probability = (
            max_trajectory_failure_probability
        )
        self.action_gate = (
            action_gate
            if action_gate is not None
            else ChanceConstrainedSafetyGate()
        )
        self.max_horizon = max_horizon

    def assess(
        self,
        estimate: TrajectoryRiskEstimate,
        mode: RiskMode,
    ) -> TrajectorySafetyAssessment:
        if estimate.horizon > self.max_horizon:
            raise ValueError(
                f"Trajectory horizon {estimate.horizon} melebihi "
                f"batas {self.max_horizon}"
            )

        step_assessments = tuple(
            self.action_gate.assess(step, mode)
            for step in estimate.steps
        )
        lower_failures = [
            step.lower_failure_probability
            for step in step_assessments
        ]
        upper_failures = [
            step.upper_failure_probability
            for step in step_assessments
        ]

        lower_failure = max(lower_failures)
        upper_failure = min(1.0, sum(upper_failures))
        bottleneck_index = max(
            range(estimate.horizon),
            key=lambda index: (
                upper_failures[index],
                estimate.steps[index].action,
            ),
        )
        enough = all(
            step.sample_count >= self.action_gate.min_samples
            for step in estimate.steps
        )
        known_bad_step = any(
            assessment.known_bad
            for assessment in step_assessments
        )
        budget = self.max_trajectory_failure_probability
        known_bad = (
            known_bad_step
            or (enough and lower_failure > budget)
        )
        certified_safe = (
            enough
            and not known_bad
            and upper_failure <= budget
        )
        uncertain = not certified_safe and not known_bad

        if known_bad_step:
            allowed = False
            reason = "STEP_KNOWN_BAD"
        elif known_bad:
            allowed = False
            reason = "TRAJECTORY_KNOWN_BAD"
        elif mode == RiskMode.CONSERVATIVE:
            allowed = certified_safe
            reason = (
                "TRAJECTORY_CERTIFIED_SAFE"
                if certified_safe
                else "TRAJECTORY_NOT_CERTIFIED"
            )
        else:
            allowed = True
            reason = (
                "TRAJECTORY_CERTIFIED_SAFE"
                if certified_safe
                else "TRAJECTORY_UNCERTAIN_ALLOWED"
            )

        return TrajectorySafetyAssessment(
            trajectory_id=estimate.trajectory_id,
            allowed=allowed,
            certified_safe=certified_safe,
            known_bad=known_bad,
            uncertain=uncertain,
            lower_failure_probability=lower_failure,
            upper_failure_probability=upper_failure,
            product_success_estimate=(
                estimate.product_success_estimate
            ),
            risk_budget=budget,
            risk_budget_remaining=max(0.0, budget - upper_failure),
            risk_budget_excess=max(0.0, upper_failure - budget),
            bottleneck_step_index=bottleneck_index,
            bottleneck_action=estimate.steps[bottleneck_index].action,
            step_assessments=step_assessments,
            reason=reason,
        )


class TrajectoryRiskPolicy:
    """Risk-aware ranking for receding-horizon candidate trajectories."""

    def __init__(
        self,
        safety_gate: Optional[
            TrajectoryChanceConstrainedSafetyGate
        ] = None,
        competitive_margin: float = 0.03,
        horizon_penalty: float = 0.01,
    ):
        if not 0.0 <= competitive_margin <= 1.0:
            raise ValueError("competitive_margin harus di [0,1]")
        if not 0.0 <= horizon_penalty <= 1.0:
            raise ValueError("horizon_penalty harus di [0,1]")
        self.safety_gate = (
            safety_gate
            if safety_gate is not None
            else TrajectoryChanceConstrainedSafetyGate()
        )
        self.competitive_margin = competitive_margin
        self.horizon_penalty = horizon_penalty

    def _choose_mode(
        self,
        base_scores: Dict[str, float],
        estimates: Dict[str, TrajectoryRiskEstimate],
        model_reliability: float,
        requested_mode: Optional[RiskMode],
    ) -> Dict:
        best = max(base_scores.values())
        relevant_ids = sorted(
            trajectory_id
            for trajectory_id, score in base_scores.items()
            if score >= best - self.competitive_margin
        )
        relevant = [estimates[item] for item in relevant_ids]
        novelty = max(
            (
                max(
                    step.epistemic_uncertainty
                    for step in estimate.steps
                )
                for estimate in relevant
            ),
            default=0.0,
        )
        uncertainty = max(
            (
                estimate.mean_combined_uncertainty
                for estimate in relevant
            ),
            default=0.0,
        )
        consequence = max(
            (
                estimate.max_failure_consequence
                for estimate in relevant
            ),
            default=0.0,
        )
        reliability = max(0.0, min(1.0, model_reliability))

        if requested_mode is not None:
            mode = requested_mode
            reason = "REQUESTED"
        elif (
            consequence >= 0.75
            and (uncertainty >= 0.45 or reliability < 0.75)
        ):
            mode = RiskMode.CONSERVATIVE
            reason = "HIGH_CONSEQUENCE_TRAJECTORY_UNCERTAINTY"
        elif (
            novelty >= 0.25
            and consequence < 0.50
            and reliability >= 0.50
        ):
            mode = RiskMode.EXPLORATORY
            reason = "SAFE_TRAJECTORY_INFORMATION_GAIN"
        else:
            mode = RiskMode.BALANCED
            reason = "BALANCED_TRAJECTORY_DEFAULT"

        return {
            "mode": mode,
            "reason": reason,
            "relevant_trajectories": relevant_ids,
            "decision_relevant_novelty": novelty,
            "decision_relevant_uncertainty": uncertainty,
            "max_failure_consequence": consequence,
            "model_reliability": reliability,
        }

    def rank(
        self,
        base_scores: Dict[str, float],
        estimates: Dict[str, TrajectoryRiskEstimate],
        model_reliability: float = 1.0,
        requested_mode: Optional[RiskMode] = None,
    ) -> Dict:
        if not base_scores:
            raise ValueError("base_scores trajectory tidak boleh kosong")
        if set(base_scores) != set(estimates):
            raise ValueError(
                "base_scores dan trajectory estimates harus memiliki id sama"
            )

        meta = self._choose_mode(
            base_scores,
            estimates,
            model_reliability,
            requested_mode,
        )
        mode = meta["mode"]
        assessments = {
            trajectory_id: self.safety_gate.assess(estimate, mode)
            for trajectory_id, estimate in estimates.items()
        }
        mode_weights = {
            RiskMode.CONSERVATIVE: (0.75, 0.30, 0.00),
            RiskMode.BALANCED: (0.50, 0.18, 0.05),
            RiskMode.EXPLORATORY: (0.25, 0.08, 0.20),
        }
        risk_weight, uncertainty_weight, information_weight = (
            mode_weights[mode]
        )

        adjusted_scores: Dict[str, float] = {}
        audit: Dict[str, Dict] = {}
        reliability = max(0.0, min(1.0, model_reliability))

        for trajectory_id, estimate in estimates.items():
            gate = assessments[trajectory_id]
            evidence_factor = min(
                1.0,
                estimate.min_sample_count
                / self.safety_gate.action_gate.min_samples,
            )
            prediction_contribution = (
                0.20
                * reliability
                * evidence_factor
                * estimate.product_success_estimate
            )
            # Chance probability remains influential even when sandbox
            # consequences are intentionally low.
            consequence_scale = (
                0.50 + 0.50 * estimate.max_failure_consequence
            )
            risk_exposure = (
                gate.upper_failure_probability
                * consequence_scale
            )
            information_bonus = (
                estimate.mean_information_gain
                * (1.0 - estimate.max_failure_consequence)
            )
            length_penalty = (
                self.horizon_penalty
                * max(0, estimate.horizon - 1)
            )
            score = (
                base_scores[trajectory_id]
                + prediction_contribution
                - risk_weight * risk_exposure
                - uncertainty_weight
                * estimate.mean_combined_uncertainty
                + information_weight * information_bonus
                - length_penalty
            )
            adjusted_scores[trajectory_id] = score
            audit[trajectory_id] = {
                "allowed": gate.allowed,
                "gate_reason": gate.reason,
                "certified_safe": gate.certified_safe,
                "known_bad": gate.known_bad,
                "uncertain": gate.uncertain,
                "actions": estimate.actions,
                "horizon": estimate.horizon,
                "lower_failure_probability": (
                    gate.lower_failure_probability
                ),
                "upper_failure_probability": (
                    gate.upper_failure_probability
                ),
                "product_success_estimate": (
                    gate.product_success_estimate
                ),
                "risk_budget": gate.risk_budget,
                "risk_budget_remaining": gate.risk_budget_remaining,
                "risk_budget_excess": gate.risk_budget_excess,
                "bottleneck_step_index": gate.bottleneck_step_index,
                "bottleneck_action": gate.bottleneck_action,
                "mean_combined_uncertainty": (
                    estimate.mean_combined_uncertainty
                ),
                "max_failure_consequence": (
                    estimate.max_failure_consequence
                ),
                "mean_information_gain": (
                    estimate.mean_information_gain
                ),
                "minimum_sample_count": estimate.min_sample_count,
                "prediction_evidence_factor": evidence_factor,
                "prediction_contribution": prediction_contribution,
                "risk_exposure": risk_exposure,
                "length_penalty": length_penalty,
                "base_score": base_scores[trajectory_id],
                "risk_adjusted_score": score,
                "step_audit": tuple(
                    {
                        "action": step.action,
                        "certified_safe": step.certified_safe,
                        "known_bad": step.known_bad,
                        "uncertain": step.uncertain,
                        "lower_failure_probability": (
                            step.lower_failure_probability
                        ),
                        "upper_failure_probability": (
                            step.upper_failure_probability
                        ),
                    }
                    for step in gate.step_assessments
                ),
            }

        eligible = sorted(
            trajectory_id
            for trajectory_id, assessment in assessments.items()
            if assessment.allowed
        )
        selected = (
            max(
                eligible,
                key=lambda trajectory_id: (
                    adjusted_scores[trajectory_id],
                    trajectory_id,
                ),
            )
            if eligible
            else None
        )

        return {
            "selected_trajectory_id": selected,
            "abstained": selected is None,
            "risk_mode": mode,
            "risk_mode_reason": meta["reason"],
            "relevant_trajectories": (
                meta["relevant_trajectories"]
            ),
            "decision_relevant_novelty": (
                meta["decision_relevant_novelty"]
            ),
            "decision_relevant_uncertainty": (
                meta["decision_relevant_uncertainty"]
            ),
            "eligible_trajectories": eligible,
            "blocked_trajectories": sorted(
                trajectory_id
                for trajectory_id, assessment in assessments.items()
                if not assessment.allowed
            ),
            "risk_adjusted_scores": adjusted_scores,
            "trajectory_audit": audit,
        }


# =========================================================================
# V2.29 — PREFERENCE-AWARE RISK / TRAJECTORY INTEGRATION
# =========================================================================

@dataclass(frozen=True)
class PreferenceAwareUtilityEstimate:
    """Read-only utility estimate under one exact objective-profile version.

    ``source`` is intentionally explicit:
    - ``profile_q``: actual scalar experience learned under this exact profile;
    - ``reweighted_actual_vector_history``: profile-independent actual objective
      vectors reinterpreted read-only under this profile;
    - ``neutral_prior``: no utility experience is available.

    Reweighted history is not current-profile scalar calibration and never
    creates Q/world-model samples by itself.
    """

    belief_context_id: Optional[str]
    context: str
    state_key: str
    action_reference: str
    action_instance_id: str
    objective_profile_instance_id: str
    objective_profile_signature: str
    source: str
    mean: float
    variance: Optional[float]
    aleatoric_std: Optional[float]
    epistemic_lower: float
    epistemic_upper: float
    epistemic_radius: float
    support: int
    total_count: int
    unscorable_count: int
    coverage: float
    mask_count: int
    q_sample_count: int = 0
    learning_mutation: bool = False
    reweighted_history_is_scalar_calibration: bool = False

    def __post_init__(self):
        if not self.context:
            raise ValueError("context tidak boleh kosong")
        if not self.state_key:
            raise ValueError("state_key tidak boleh kosong")
        if not self.action_reference:
            raise ValueError("action_reference tidak boleh kosong")
        if not self.action_instance_id:
            raise ValueError("action_instance_id tidak boleh kosong")
        if not self.objective_profile_instance_id:
            raise ValueError("objective_profile_instance_id tidak boleh kosong")
        if self.source not in {
            "profile_q",
            "reweighted_actual_vector_history",
            "neutral_prior",
        }:
            raise ValueError(f"source preference utility tidak dikenal: {self.source}")
        if not 0.0 <= self.mean <= 1.0:
            raise ValueError("mean preference utility harus di [0,1]")
        if self.variance is not None and self.variance < 0.0:
            raise ValueError("variance preference utility tidak boleh negatif")
        if self.aleatoric_std is not None and self.aleatoric_std < 0.0:
            raise ValueError("aleatoric_std tidak boleh negatif")
        if not 0.0 <= self.epistemic_lower <= self.epistemic_upper <= 1.0:
            raise ValueError("epistemic interval harus di [0,1]")
        if not 0.0 <= self.epistemic_radius <= 1.0:
            raise ValueError("epistemic_radius harus di [0,1]")
        if self.support < 0 or self.total_count < 0 or self.unscorable_count < 0:
            raise ValueError("sample count preference utility tidak boleh negatif")
        if self.q_sample_count < 0:
            raise ValueError("q_sample_count tidak boleh negatif")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage harus di [0,1]")
        if self.learning_mutation:
            raise ValueError(
                "PreferenceAwareUtilityEstimate adalah read-only estimate; "
                "learning_mutation harus False"
            )
        if self.reweighted_history_is_scalar_calibration:
            raise ValueError(
                "Reweighted objective history tidak boleh dinyatakan sebagai "
                "current-profile scalar calibration"
            )

    @property
    def lower_confidence_bound(self) -> float:
        return self.epistemic_lower

    @property
    def upper_confidence_bound(self) -> float:
        return self.epistemic_upper


class PreferenceAwareRiskPolicy:
    """Decision policy over profile-aware utility + independent technical risk.

    This policy is deliberately read-only. It ranks already-computed estimates
    and does not own Q/world-model/calibration updates.
    """

    def __init__(
        self,
        safety_gate: Optional[ChanceConstrainedSafetyGate] = None,
    ):
        self.safety_gate = (
            safety_gate
            if safety_gate is not None
            else ChanceConstrainedSafetyGate()
        )

    @staticmethod
    def _validate_provenance(
        utility_estimates: Dict[str, PreferenceAwareUtilityEstimate],
        technical_estimates: Dict[str, ActionRiskEstimate],
    ) -> None:
        if not utility_estimates:
            raise ValueError("utility_estimates tidak boleh kosong")
        if set(utility_estimates) != set(technical_estimates):
            raise ValueError(
                "utility_estimates dan technical_estimates harus memiliki action yang sama"
            )

        utilities = list(utility_estimates.values())
        profile_ids = {item.objective_profile_instance_id for item in utilities}
        profile_signatures = {item.objective_profile_signature for item in utilities}
        belief_contexts = {item.belief_context_id for item in utilities}
        state_keys = {item.state_key for item in utilities}

        if len(profile_ids) != 1:
            raise ValueError("Mixed objective-profile provenance pada preference risk")
        if len(profile_signatures) != 1:
            raise ValueError("Mixed objective-profile signature pada preference risk")
        if len(belief_contexts) != 1:
            raise ValueError("Mixed BeliefContext provenance pada preference risk")
        if len(state_keys) != 1:
            raise ValueError("Mixed canonical-state provenance pada preference risk")

        for action, utility in utility_estimates.items():
            technical = technical_estimates[action]
            if action != utility.action_reference or action != technical.action:
                raise ValueError("Action reference tidak konsisten pada preference risk")
            if technical.state_key != utility.state_key:
                raise ValueError("Mixed state provenance antara utility dan technical estimate")
            if technical.action_instance_id != utility.action_instance_id:
                raise ValueError("Mixed action-version provenance pada preference risk")
            tech_profile = technical.objective_profile_instance_id
            if tech_profile is not None and tech_profile != utility.objective_profile_instance_id:
                raise ValueError("Mixed objective-profile provenance antara utility dan technical estimate")
            tech_belief = getattr(technical, "belief_context_id", None)
            if tech_belief is not None and tech_belief != utility.belief_context_id:
                raise ValueError("Mixed BeliefContext provenance antara utility dan technical estimate")
            if utility.learning_mutation:
                raise ValueError("Preference utility estimate tidak boleh membawa learning mutation")

    def rank(
        self,
        nonutility_scores: Dict[str, float],
        utility_estimates: Dict[str, PreferenceAwareUtilityEstimate],
        technical_estimates: Dict[str, ActionRiskEstimate],
        requested_mode: Optional[RiskMode] = None,
    ) -> Dict:
        if set(nonutility_scores) != set(utility_estimates):
            raise ValueError(
                "nonutility_scores dan utility_estimates harus memiliki action yang sama"
            )
        self._validate_provenance(utility_estimates, technical_estimates)

        mode = requested_mode if requested_mode is not None else RiskMode.BALANCED
        if not isinstance(mode, RiskMode):
            mode = RiskMode(mode)

        # Policy coefficients are decision semantics only; they do not alter
        # objective truth/history. Coverage is never folded into utility mean.
        weights = {
            RiskMode.CONSERVATIVE: {
                "volatility": 0.35,
                "epistemic": 0.30,
                "coverage": 0.20,
                "risk": 0.30,
                "information": 0.00,
            },
            RiskMode.BALANCED: {
                "volatility": 0.18,
                "epistemic": 0.12,
                "coverage": 0.08,
                "risk": 0.20,
                "information": 0.04,
            },
            RiskMode.EXPLORATORY: {
                "volatility": 0.05,
                "epistemic": 0.03,
                "coverage": 0.00,
                "risk": 0.10,
                "information": 0.32,
            },
        }[mode]

        assessments = {
            action: self.safety_gate.assess(technical, mode)
            for action, technical in technical_estimates.items()
        }

        scores: Dict[str, float] = {}
        audit: Dict[str, Dict] = {}
        for action, utility in utility_estimates.items():
            technical = technical_estimates[action]
            gate = assessments[action]
            std = utility.aleatoric_std or 0.0
            coverage_gap = 1.0 - utility.coverage
            volatility_penalty = weights["volatility"] * min(1.0, std)
            epistemic_penalty = weights["epistemic"] * utility.epistemic_radius
            coverage_policy_term = weights["coverage"] * coverage_gap
            upper_failure = gate.upper_failure_probability
            risk_exposure = upper_failure * technical.failure_consequence
            technical_risk_penalty = weights["risk"] * risk_exposure

            # Information bonus is bounded and driven by missing/scorable
            # coverage uncertainty. It is zero for fully covered history and
            # cannot make KNOWN_BAD actions eligible because the gate is applied
            # independently before selection.
            information_bonus = (
                weights["information"]
                * coverage_gap
                * (1.0 - technical.failure_consequence)
            )

            score = (
                utility.mean
                + nonutility_scores[action]
                - volatility_penalty
                - epistemic_penalty
                - coverage_policy_term
                - technical_risk_penalty
                + information_bonus
            )
            scores[action] = score
            audit[action] = {
                "utility_source": utility.source,
                "utility_mean": utility.mean,
                "utility_variance": utility.variance,
                "utility_std": utility.aleatoric_std,
                "utility_support": utility.support,
                "q_sample_count": utility.q_sample_count,
                "epistemic_interval": (
                    utility.epistemic_lower,
                    utility.epistemic_upper,
                ),
                "epistemic_radius": utility.epistemic_radius,
                "coverage": utility.coverage,
                "unscorable_count": utility.unscorable_count,
                "mask_count": utility.mask_count,
                "volatility_penalty": volatility_penalty,
                "epistemic_policy_term": epistemic_penalty,
                "coverage_policy_term": coverage_policy_term,
                "information_bonus": information_bonus,
                "technical_success_probability": technical.success_probability,
                "technical_success_interval": (
                    technical.success_lower,
                    technical.success_upper,
                ),
                "technical_sample_count": technical.sample_count,
                "failure_consequence": technical.failure_consequence,
                "technical_risk_penalty": technical_risk_penalty,
                "known_bad": gate.known_bad,
                "certified_safe": gate.certified_safe,
                "uncertain_technical": gate.uncertain,
                "allowed": gate.allowed,
                "gate_reason": gate.reason,
                "risk_adjusted_score": score,
                "objective_profile_instance_id": utility.objective_profile_instance_id,
                "objective_profile_signature": utility.objective_profile_signature,
                "belief_context_id": utility.belief_context_id,
                "state_key": utility.state_key,
                "action_instance_id": utility.action_instance_id,
                "reweighted_history_is_scalar_calibration": False,
                "learning_mutation": False,
            }

        eligible = sorted(
            action for action, assessment in assessments.items()
            if assessment.allowed
        )
        selected = (
            max(eligible, key=lambda action: (scores[action], action))
            if eligible
            else None
        )

        return {
            "selected_action": selected,
            "abstained": selected is None,
            "risk_mode": mode,
            "risk_mode_reason": (
                "REQUESTED" if requested_mode is not None
                else "PREFERENCE_BALANCED_DEFAULT"
            ),
            "eligible_actions": eligible,
            "blocked_actions": sorted(
                action for action, assessment in assessments.items()
                if not assessment.allowed
            ),
            "risk_adjusted_scores": scores,
            "action_audit": audit,
            "learning_mutation": False,
        }


@dataclass(frozen=True)
class PreferenceAwareTrajectoryEstimate:
    """Read-only multi-state trajectory estimate from preference-aware steps.

    Cross-step covariance is *not* claimed. The trajectory volatility value is
    a conservative aggregation proxy over exact per-step utility statistics.
    """

    trajectory_id: str
    utility_steps: Tuple[PreferenceAwareUtilityEstimate, ...]
    technical_steps: Tuple[ActionRiskEstimate, ...]

    def __post_init__(self):
        if not self.trajectory_id:
            raise ValueError("trajectory_id tidak boleh kosong")
        if not self.utility_steps:
            raise ValueError("Preference trajectory harus memiliki minimal satu step")
        if len(self.utility_steps) != len(self.technical_steps):
            raise ValueError("utility_steps dan technical_steps harus sama panjang")

        profile_ids = {step.objective_profile_instance_id for step in self.utility_steps}
        signatures = {step.objective_profile_signature for step in self.utility_steps}
        belief_contexts = {step.belief_context_id for step in self.utility_steps}
        if len(profile_ids) != 1:
            raise ValueError("Mixed objective-profile provenance dalam trajectory")
        if len(signatures) != 1:
            raise ValueError("Mixed objective-profile signature dalam trajectory")
        if len(belief_contexts) != 1:
            raise ValueError("Mixed BeliefContext provenance dalam trajectory")

        for utility, technical in zip(self.utility_steps, self.technical_steps):
            if utility.action_reference != technical.action:
                raise ValueError("Trajectory action reference provenance tidak cocok")
            if utility.state_key != technical.state_key:
                raise ValueError("Trajectory state provenance tidak cocok")
            if utility.action_instance_id != technical.action_instance_id:
                raise ValueError("Trajectory action-version provenance tidak cocok")
            tech_profile = technical.objective_profile_instance_id
            if tech_profile is not None and tech_profile != utility.objective_profile_instance_id:
                raise ValueError("Trajectory profile provenance tidak cocok")
            tech_belief = getattr(technical, "belief_context_id", None)
            if tech_belief is not None and tech_belief != utility.belief_context_id:
                raise ValueError("Trajectory BeliefContext provenance tidak cocok")

    @property
    def objective_profile_instance_id(self) -> str:
        return self.utility_steps[0].objective_profile_instance_id

    @property
    def objective_profile_signature(self) -> str:
        return self.utility_steps[0].objective_profile_signature

    @property
    def belief_context_id(self) -> Optional[str]:
        return self.utility_steps[0].belief_context_id

    @property
    def horizon(self) -> int:
        return len(self.utility_steps)

    @property
    def mean(self) -> float:
        return sum(step.mean for step in self.utility_steps) / self.horizon

    @property
    def aleatoric_std(self) -> float:
        # Root-sum-square scaled to the mean utility horizon. This is a policy
        # risk proxy only because cross-step covariance is unknown.
        return math.sqrt(sum((step.aleatoric_std or 0.0) ** 2 for step in self.utility_steps)) / self.horizon

    @property
    def epistemic_radius(self) -> float:
        return max(step.epistemic_radius for step in self.utility_steps)

    @property
    def coverage(self) -> float:
        return min(step.coverage for step in self.utility_steps)

    @property
    def support(self) -> int:
        return min(step.support for step in self.utility_steps)


class PreferenceAwareTrajectoryRiskPolicy:
    """Read-only trajectory ranking over V2.29 per-step estimates."""

    def __init__(
        self,
        safety_gate: Optional[ChanceConstrainedSafetyGate] = None,
    ):
        self.safety_gate = (
            safety_gate
            if safety_gate is not None
            else ChanceConstrainedSafetyGate()
        )

    def rank(
        self,
        trajectory_estimates: Dict[str, PreferenceAwareTrajectoryEstimate],
        requested_mode: Optional[RiskMode] = None,
    ) -> Dict:
        if not trajectory_estimates:
            raise ValueError("trajectory_estimates tidak boleh kosong")
        for key, estimate in trajectory_estimates.items():
            if key != estimate.trajectory_id:
                raise ValueError("trajectory_id key tidak konsisten")

        profile_ids = {item.objective_profile_instance_id for item in trajectory_estimates.values()}
        signatures = {item.objective_profile_signature for item in trajectory_estimates.values()}
        belief_contexts = {item.belief_context_id for item in trajectory_estimates.values()}
        if len(profile_ids) != 1 or len(signatures) != 1 or len(belief_contexts) != 1:
            raise ValueError("Mixed trajectory preference provenance")

        mode = requested_mode if requested_mode is not None else RiskMode.BALANCED
        if not isinstance(mode, RiskMode):
            mode = RiskMode(mode)

        weights = {
            RiskMode.CONSERVATIVE: (0.40, 0.30, 0.20, 0.30, 0.00),
            RiskMode.BALANCED: (0.22, 0.12, 0.08, 0.20, 0.04),
            RiskMode.EXPLORATORY: (0.06, 0.03, 0.00, 0.10, 0.32),
        }[mode]
        vol_w, epi_w, cov_w, risk_w, info_w = weights

        scores = {}
        audit = {}
        blocked = []
        eligible = []

        for trajectory_id, estimate in trajectory_estimates.items():
            step_gates = tuple(
                self.safety_gate.assess(technical, mode)
                for technical in estimate.technical_steps
            )
            known_bad = any(gate.known_bad for gate in step_gates)
            allowed = all(gate.allowed for gate in step_gates)
            if allowed:
                eligible.append(trajectory_id)
            else:
                blocked.append(trajectory_id)

            upper_any_failure = min(
                1.0,
                sum(gate.upper_failure_probability for gate in step_gates),
            )
            max_consequence = max(step.failure_consequence for step in estimate.technical_steps)
            coverage_gap = 1.0 - estimate.coverage
            information_bonus = info_w * coverage_gap * (1.0 - max_consequence)
            volatility_penalty = vol_w * min(1.0, estimate.aleatoric_std)
            epistemic_penalty = epi_w * estimate.epistemic_radius
            coverage_policy_term = cov_w * coverage_gap
            risk_penalty = risk_w * upper_any_failure * max_consequence
            score = (
                estimate.mean
                - volatility_penalty
                - epistemic_penalty
                - coverage_policy_term
                - risk_penalty
                + information_bonus
            )
            scores[trajectory_id] = score
            audit[trajectory_id] = {
                "utility_mean": estimate.mean,
                "utility_std_proxy": estimate.aleatoric_std,
                "epistemic_radius": estimate.epistemic_radius,
                "coverage": estimate.coverage,
                "support": estimate.support,
                "volatility_penalty": volatility_penalty,
                "epistemic_policy_term": epistemic_penalty,
                "coverage_policy_term": coverage_policy_term,
                "information_bonus": information_bonus,
                "upper_any_failure_bound": upper_any_failure,
                "risk_penalty": risk_penalty,
                "known_bad": known_bad,
                "allowed": allowed,
                "risk_adjusted_score": score,
                "horizon": estimate.horizon,
                "objective_profile_instance_id": estimate.objective_profile_instance_id,
                "belief_context_id": estimate.belief_context_id,
                "counterfactual_is_experience": False,
                "learning_mutation": False,
            }

        eligible = sorted(eligible)
        selected = (
            max(eligible, key=lambda trajectory_id: (scores[trajectory_id], trajectory_id))
            if eligible
            else None
        )
        return {
            "selected_trajectory_id": selected,
            "abstained": selected is None,
            "risk_mode": mode,
            "risk_mode_reason": (
                "REQUESTED" if requested_mode is not None
                else "PREFERENCE_BALANCED_DEFAULT"
            ),
            "eligible_trajectories": eligible,
            "blocked_trajectories": sorted(blocked),
            "risk_adjusted_scores": scores,
            "trajectory_audit": audit,
            "counterfactual_is_experience": False,
            "learning_mutation": False,
        }


_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"
_PICKLE_COMPAT_CLASSES = (
    DecisionRecord,
    TransitionRecord,
    TrajectoryDecisionRecord,
    DecisionPolicy,
    UncertaintyDecisionMode,
    UncertaintyRiskProfile,
    UncertaintyDecisionResult,
    MetaRiskSignals,
    MetaRiskDecision,
    AdaptiveRiskModePolicy,
    UncertaintyAwareDecisionPolicy,
    RiskMode,
    ActionRiskEstimate,
    SafetyGateAssessment,
    ChanceConstrainedSafetyGate,
    MetaRiskPolicy,
    TrajectoryRiskEstimate,
    TrajectorySafetyAssessment,
    TrajectoryChanceConstrainedSafetyGate,
    TrajectoryRiskPolicy,
    PreferenceAwareUtilityEstimate,
    PreferenceAwareRiskPolicy,
    PreferenceAwareTrajectoryEstimate,
    PreferenceAwareTrajectoryRiskPolicy,
)
for _cls in _PICKLE_COMPAT_CLASSES:
    _cls.__module__ = _CANONICAL_PICKLE_MODULE
del _cls

# Memory containers and RouteAction are owned by physical modules.
from .memory import (
    TransitionMemory, DecisionMemory, TrajectoryDecisionMemory,
    MetaRiskDecisionMemory,
)
from .planning import RouteAction

__all__ = [
    "DecisionRecord",
    "TransitionRecord",
    "TrajectoryDecisionRecord",
    "DecisionPolicy",
    "UncertaintyDecisionMode",
    "UncertaintyRiskProfile",
    "UncertaintyDecisionResult",
    "MetaRiskSignals",
    "MetaRiskDecision",
    "AdaptiveRiskModePolicy",
    "UncertaintyAwareDecisionPolicy",
    "RiskMode",
    "ActionRiskEstimate",
    "SafetyGateAssessment",
    "ChanceConstrainedSafetyGate",
    "MetaRiskPolicy",
    "TrajectoryRiskEstimate",
    "TrajectorySafetyAssessment",
    "TrajectoryChanceConstrainedSafetyGate",
    "TrajectoryRiskPolicy",
    "PreferenceAwareUtilityEstimate",
    "PreferenceAwareRiskPolicy",
    "PreferenceAwareTrajectoryEstimate",
    "PreferenceAwareTrajectoryRiskPolicy",
    "TransitionMemory", "DecisionMemory", "TrajectoryDecisionMemory",
    "MetaRiskDecisionMemory", "RouteAction",
]
