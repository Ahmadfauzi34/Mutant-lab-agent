"""World-model/prediction subsystem — physical extraction M5A.

Owns low-coupling empirical prediction primitives. Planner-bound world episode
classes remain temporarily in the compatibility kernel.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .memory import CounterfactualMemory, PredictionMemory, PredictionErrorMemory
from .planning import Point, SpaceTimeNode
from .epistemic import AuditReport

if TYPE_CHECKING:
    from .decision import DecisionRecord

@dataclass
class WorldOutcome:
    success: bool
    reached_goal: bool
    safety_score: float
    efficiency_score: float
    reward: float
    reasons: Tuple[str, ...]


@dataclass
class PredictionErrorRecord:
    prediction_error_id: int
    context: str
    action_name: str
    predicted_reward: float
    actual_reward: float
    reward_error: float
    safety_error: float
    efficiency_error: float
    success_error: float
    aggregate_error: float
    model_accuracy: float
    prediction_world_signature: Tuple
    actual_world_signature: Optional[Tuple]
    state_drift: bool
    reliability_before: float
    reliability_after: float
    calibrated: bool
    belief_context_id: Optional[str] = None
    action_instance_id: Optional[str] = None
    state_key: Optional[str] = None

    # V2.26
    objective_errors: Optional[Dict[str, float]] = None
    objective_profile_signature: Optional[str] = None

    # V2.27
    objective_profile_instance_id: Optional[str] = None
    scalar_state_key: Optional[str] = None


@dataclass(frozen=True)
class PredictionUncertainty:
    """
    Uncertainty is decomposed, not collapsed into a single magic score.

    reward interval:
        Hoeffding-style concentration interval for bounded reward [0,1].

    reward_aleatoric_std:
        observed sample standard deviation; None when n < 2.

    success interval:
        Wilson interval for Bernoulli success probability.

    These intervals assume observations inside the scoped model are
    sufficiently comparable. They are not truth probabilities and do not
    guarantee coverage under regime drift.
    """
    confidence_level: float
    sample_count: int

    reward_lower: float
    reward_upper: float
    reward_epistemic_radius: float
    reward_aleatoric_std: Optional[float]

    success_lower: float
    success_upper: float
    success_epistemic_radius: float

    insufficient_data: bool

    # V2.27 — success knowledge is profile-independent.
    success_sample_count: int = 0

    @property
    def reward_interval_width(self) -> float:
        return self.reward_upper - self.reward_lower

    @property
    def success_interval_width(self) -> float:
        return self.success_upper - self.success_lower


class PredictionUncertaintyEstimator:
    """
    CPU-only uncertainty estimator for bounded outcomes.

    Deliberately separates:
    - epistemic/sample uncertainty
    - observed aleatoric reward variability
    - model calibration reliability (stored elsewhere)
    """
    def __init__(
        self,
        confidence_level: float = 0.95,
        minimum_sufficient_samples: int = 10,
    ):
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                "confidence_level harus di (0,1)"
            )
        if minimum_sufficient_samples < 1:
            raise ValueError(
                "minimum_sufficient_samples harus >= 1"
            )

        self.confidence_level = confidence_level
        self.minimum_sufficient_samples = (
            minimum_sufficient_samples
        )

        # Fixed normal approximation for 95% Wilson by default.
        # Other common levels are supported without scipy.
        lookup = {
            0.80: 1.2815515655446004,
            0.90: 1.6448536269514722,
            0.95: 1.959963984540054,
            0.99: 2.5758293035489004,
        }
        rounded = round(confidence_level, 2)
        if rounded not in lookup:
            raise ValueError(
                "confidence_level supported: 0.80, 0.90, 0.95, 0.99"
            )
        self.z = lookup[rounded]

    def _hoeffding_reward_interval(
        self,
        mean: float,
        n: int,
    ) -> Tuple[float, float, float]:
        if n <= 0:
            return 0.0, 1.0, 0.5

        alpha = 1.0 - self.confidence_level
        radius = math.sqrt(
            math.log(2.0 / alpha)
            / (2.0 * n)
        )

        lower = max(0.0, mean - radius)
        upper = min(1.0, mean + radius)

        return (
            lower,
            upper,
            max(
                mean - lower,
                upper - mean,
            ),
        )

    def _wilson_success_interval(
        self,
        probability: float,
        n: int,
    ) -> Tuple[float, float, float]:
        if n <= 0:
            return 0.0, 1.0, 0.5

        z2 = self.z * self.z
        denominator = 1.0 + z2 / n

        center = (
            probability
            + z2 / (2.0 * n)
        ) / denominator

        half = (
            self.z
            / denominator
            * math.sqrt(
                (
                    probability
                    * (1.0 - probability)
                    / n
                )
                + (
                    z2
                    / (4.0 * n * n)
                )
            )
        )

        lower = max(0.0, center - half)
        upper = min(1.0, center + half)

        return (
            lower,
            upper,
            max(
                probability - lower,
                upper - probability,
            ),
        )

    def estimate(
        self,
        reward_mean: float,
        reward_std: Optional[float],
        success_probability: float,
        sample_count: int,
        success_sample_count: Optional[
            int
        ] = None,
    ) -> PredictionUncertainty:
        (
            reward_lower,
            reward_upper,
            reward_radius,
        ) = self._hoeffding_reward_interval(
            reward_mean,
            sample_count,
        )

        resolved_success_count = (
            sample_count
            if success_sample_count
                is None
            else int(
                success_sample_count
            )
        )

        (
            success_lower,
            success_upper,
            success_radius,
        ) = self._wilson_success_interval(
            success_probability,
            resolved_success_count,
        )

        return PredictionUncertainty(
            confidence_level=self.confidence_level,
            sample_count=sample_count,
            reward_lower=reward_lower,
            reward_upper=reward_upper,
            reward_epistemic_radius=reward_radius,
            reward_aleatoric_std=reward_std,
            success_lower=success_lower,
            success_upper=success_upper,
            success_epistemic_radius=success_radius,
            insufficient_data=(
                sample_count
                    < self.minimum_sufficient_samples
                or resolved_success_count
                    < self.minimum_sufficient_samples
            ),
            success_sample_count=(
                resolved_success_count
            ),
        )


@dataclass
class OutcomePrediction:
    prediction_id: int
    context: str
    action_name: str
    belief_context_id: Optional[str]
    predicted_reward: float
    predicted_success_probability: float
    sample_count: int
    uncertainty: Optional[PredictionUncertainty] = None

    # V2.27 — independent support for technical success constraint.
    success_sample_count: int = 0
    model_reliability: float = 0.5
    model_source: str = "context_scoped_empirical"
    action_instance_id: Optional[str] = None
    state_key: Optional[str] = None

    # V2.26
    predicted_objectives: Optional[Dict[str, float]] = None
    objective_sample_counts: Optional[Dict[str, int]] = None
    objective_profile_signature: Optional[str] = None

    # V2.27
    objective_profile_instance_id: Optional[str] = None
    scalar_state_key: Optional[str] = None
    reweighted_objective_utility: Optional[float] = None
    reweighted_objective_support: int = 0

    # V2.28 — exact joint/missingness-aware historical utility distribution.
    reweighted_objective_variance: Optional[float] = None
    reweighted_objective_std: Optional[float] = None
    reweighted_objective_coverage: float = 0.0
    reweighted_objective_unscorable_count: int = 0
    reweighted_objective_mask_count: int = 0
    reweighted_objective_uncertainty: Optional[
        PredictionUncertainty
    ] = None

    @property
    def predicted_success(self) -> bool:
        return self.predicted_success_probability >= 0.5


class ContextScopedWorldModel:
    """
    CPU-first empirical outcome model.

    Learns only from ACTUAL outcomes:
        (belief_context, environment_context, action)
            -> reward mean
            -> reward variance / standard deviation
            -> success probability
            -> sample count

    New belief contexts start from neutral priors and do not inherit outcome
    statistics from previous regimes. Historical models remain queryable.
    """
    def __init__(
        self,
        prior_reward: float = 0.5,
        prior_success: float = 0.5,
    ):
        if not 0.0 <= prior_reward <= 1.0:
            raise ValueError(
                "prior_reward harus 0..1"
            )
        if not 0.0 <= prior_success <= 1.0:
            raise ValueError(
                "prior_success harus 0..1"
            )

        self.prior_reward = prior_reward
        self.prior_success = prior_success

        # key -> [reward_sum, reward_sq_sum, success_sum, count]
        self._stats: Dict[
            Tuple[Optional[str], str, str],
            List[float],
        ] = {}

    def _key(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str],
    ) -> Tuple[Optional[str], str, str]:
        return (
            belief_context_id,
            context,
            action_name,
        )

    def count(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
    ) -> int:
        values = self._stats.get(
            self._key(
                context,
                action_name,
                belief_context_id,
            )
        )
        return (
            int(values[3])
            if values is not None
            else 0
        )

    def statistics(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        values = self._stats.get(
            self._key(
                context,
                action_name,
                belief_context_id,
            )
        )

        if values is None or values[3] <= 0:
            return {
                "reward_mean": self.prior_reward,
                "reward_variance": None,
                "reward_std": None,
                "success_probability": self.prior_success,
                "count": 0,
            }

        (
            reward_sum,
            reward_sq_sum,
            success_sum,
            count_raw,
        ) = values

        count = int(count_raw)
        reward_mean = reward_sum / count
        success_probability = (
            success_sum / count
        )

        reward_variance: Optional[float]
        reward_std: Optional[float]

        if count < 2:
            reward_variance = None
            reward_std = None
        else:
            # Unbiased sample variance, bounded against floating error.
            numerator = (
                reward_sq_sum
                - (reward_sum * reward_sum) / count
            )
            reward_variance = max(
                0.0,
                numerator / (count - 1),
            )
            reward_std = math.sqrt(
                reward_variance
            )

        return {
            "reward_mean": reward_mean,
            "reward_variance": reward_variance,
            "reward_std": reward_std,
            "success_probability": success_probability,
            "count": count,
        }

    def predict_values(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
    ) -> Tuple[float, float, int]:
        stats = self.statistics(
            context,
            action_name,
            belief_context_id,
        )
        return (
            stats["reward_mean"],
            stats["success_probability"],
            stats["count"],
        )

    def update(
        self,
        context: str,
        action_name: str,
        reward: float,
        success: bool,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(
                "reward harus 0..1"
            )

        key = self._key(
            context,
            action_name,
            belief_context_id,
        )
        values = self._stats.setdefault(
            key,
            [0.0, 0.0, 0.0, 0.0],
        )

        values[0] += reward
        values[1] += reward * reward
        values[2] += float(success)
        values[3] += 1.0

        stats = self.statistics(
            context,
            action_name,
            belief_context_id,
        )

        return {
            "belief_context_id": belief_context_id,
            "context": context,
            "action_name": action_name,
            **stats,
        }

    def state(
        self,
        belief_context_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict:
        output = {}

        for (
            scope,
            env_context,
            action_name,
        ) in self._stats:
            if (
                belief_context_id is not None
                and scope != belief_context_id
            ):
                continue

            if (
                context is not None
                and env_context != context
            ):
                continue

            stats = self.statistics(
                env_context,
                action_name,
                scope,
            )

            key = (
                (env_context, action_name)
                if context is None
                else action_name
            )

            output[key] = {
                "belief_context_id": scope,
                **stats,
            }

        return output


class WorldModelReliability:
    """
    Fractional Beta-style calibration.

    Legacy profile:
        (belief_context_id, environment_context)

    V2.22 version-isolated profile:
        (belief_context_id, environment_context, action_instance_id)

    Calls without action_instance_id preserve the exact V2.21 behavior.
    Registered action versions use the isolated profile so a calibrated
    RUN_TEST@v1 cannot silently confer trust to RUN_TEST@v2.
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError(
                "Model reliability prior harus positif"
            )
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Exact V2.21 store.
        self._profiles: Dict[
            Tuple[Optional[str], str],
            List[float],
        ] = {}

        # V2.22 action-instance calibration.
        self._action_profiles: Dict[
            Tuple[Optional[str], str, str],
            List[float],
        ] = {}

    def _key(
        self,
        context: str,
        belief_context_id: Optional[str],
    ) -> Tuple[Optional[str], str]:
        return (
            belief_context_id,
            context,
        )

    def _action_key(
        self,
        context: str,
        belief_context_id: Optional[str],
        action_instance_id: str,
    ) -> Tuple[Optional[str], str, str]:
        return (
            belief_context_id,
            context,
            action_instance_id,
        )

    def _profile(
        self,
        context: str,
        belief_context_id: Optional[str] = None,
        action_instance_id: Optional[str] = None,
    ) -> List[float]:
        if action_instance_id is None:
            return self._profiles.setdefault(
                self._key(
                    context,
                    belief_context_id,
                ),
                [
                    self.prior_alpha,
                    self.prior_beta,
                ],
            )

        return self._action_profiles.setdefault(
            self._action_key(
                context,
                belief_context_id,
                action_instance_id,
            ),
            [
                self.prior_alpha,
                self.prior_beta,
            ],
        )

    def reliability(
        self,
        context: str,
        belief_context_id: Optional[str] = None,
        action_instance_id: Optional[str] = None,
    ) -> float:
        alpha, beta = self._profile(
            context,
            belief_context_id,
            action_instance_id,
        )
        return alpha / (alpha + beta)

    def update(
        self,
        context: str,
        accuracy: float,
        belief_context_id: Optional[str] = None,
        action_instance_id: Optional[str] = None,
    ) -> float:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                "accuracy harus 0..1"
            )

        profile = self._profile(
            context,
            belief_context_id,
            action_instance_id,
        )
        profile[0] += accuracy
        profile[1] += 1.0 - accuracy

        return self.reliability(
            context,
            belief_context_id,
            action_instance_id,
        )

    def state(
        self,
        context: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        action_instance_id: Optional[str] = None,
    ) -> Dict:
        if action_instance_id is not None:
            if context is None:
                raise ValueError(
                    "context wajib untuk action-instance calibration state"
                )
            alpha, beta = self._profile(
                context,
                belief_context_id,
                action_instance_id,
            )
            return {
                "belief_context_id":
                    belief_context_id,
                "context": context,
                "action_instance_id":
                    action_instance_id,
                "alpha": alpha,
                "beta": beta,
                "reliability":
                    alpha / (alpha + beta),
            }

        if context is not None:
            alpha, beta = self._profile(
                context,
                belief_context_id,
            )
            return {
                "belief_context_id":
                    belief_context_id,
                "context": context,
                "alpha": alpha,
                "beta": beta,
                "reliability":
                    alpha / (alpha + beta),
            }

        output = {}
        for (
            scope,
            env_context,
        ), values in self._profiles.items():
            if (
                belief_context_id is not None
                and scope != belief_context_id
            ):
                continue
            output[(scope, env_context)] = {
                "alpha": values[0],
                "beta": values[1],
                "reliability": (
                    values[0]
                    / (values[0] + values[1])
                ),
            }

        action_output = {}
        for (
            scope,
            env_context,
            instance_id,
        ), values in self._action_profiles.items():
            if (
                belief_context_id is not None
                and scope != belief_context_id
            ):
                continue
            action_output[
                (
                    scope,
                    env_context,
                    instance_id,
                )
            ] = {
                "alpha": values[0],
                "beta": values[1],
                "reliability": (
                    values[0]
                    / (values[0] + values[1])
                ),
            }

        if action_output:
            return {
                "legacy": output,
                "action_instances":
                    action_output,
            }

        return output


class PredictionErrorEvaluator:
    """
    Error decomposition between predicted and actual world outcomes.
    """
    def evaluate(
        self,
        predicted: WorldOutcome,
        actual: WorldOutcome,
    ) -> Dict[str, float]:
        reward_error = abs(
            predicted.reward - actual.reward
        )
        safety_error = abs(
            predicted.safety_score - actual.safety_score
        )
        efficiency_error = abs(
            predicted.efficiency_score - actual.efficiency_score
        )
        success_error = (
            0.0
            if predicted.success == actual.success
            else 1.0
        )

        aggregate_error = (
            0.40 * reward_error
            + 0.25 * safety_error
            + 0.20 * efficiency_error
            + 0.15 * success_error
        )
        aggregate_error = max(
            0.0,
            min(1.0, aggregate_error),
        )
        accuracy = 1.0 - aggregate_error

        return {
            "reward_error": reward_error,
            "safety_error": safety_error,
            "efficiency_error": efficiency_error,
            "success_error": success_error,
            "aggregate_error": aggregate_error,
            "model_accuracy": accuracy,
        }


class CounterfactualStrategyPolicy:
    """
    V2.7:
    influence of simulated outcomes is calibrated by learned
    world-model reliability.

    effective_weight =
        counterfactual_weight * model_reliability

    Prediction still influences selection only, never direct Q update.
    """
    def __init__(self, counterfactual_weight: float = 0.40):
        if not 0.0 <= counterfactual_weight <= 1.0:
            raise ValueError("counterfactual_weight harus 0..1")
        self.counterfactual_weight = counterfactual_weight

    def effective_weight(
        self,
        model_reliability: float,
    ) -> float:
        reliability = max(
            0.0,
            min(1.0, model_reliability),
        )
        return self.counterfactual_weight * reliability

    def combine(
        self,
        base_scores: Dict[str, float],
        predicted_rewards: Dict[str, float],
        model_reliability: float = 1.0,
    ) -> Dict[str, float]:
        w = self.effective_weight(
            model_reliability
        )
        return {
            action: (
                (1.0 - w) * base_scores[action]
                + w * predicted_rewards.get(action, 0.0)
            )
            for action in base_scores
        }


@dataclass
class EnsembleMemberState:
    """Posterior and reward statistics for one online bootstrap member."""

    success_alpha: float = 1.0
    success_beta: float = 1.0
    reward_weight: float = 0.0
    reward_sum: float = 0.0
    reward_sum_sq: float = 0.0

    @property
    def success_probability(self) -> float:
        return self.success_alpha / (
            self.success_alpha + self.success_beta
        )

    @property
    def success_posterior_variance(self) -> float:
        alpha = self.success_alpha
        beta = self.success_beta
        total = alpha + beta
        return (alpha * beta) / (
            total * total * (total + 1.0)
        )

    @property
    def reward_mean(self) -> float:
        if self.reward_weight <= 0.0:
            return 0.50
        return self.reward_sum / self.reward_weight


@dataclass(frozen=True)
class EnsemblePrediction:
    belief_context_id: Optional[str]
    environment_state: str
    action: str
    predicted_reward: float
    success_probability: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    information_gain: float
    sample_count: int
    member_success_probabilities: Tuple[float, ...]
    member_reward_means: Tuple[float, ...]


class OnlineBootstrapEnsemble:
    """
    Small deterministic online ensemble for CPU-only uncertainty estimation.

    - Bootstrap multiplicities approximate Poisson(1) online bagging.
    - Beta posterior variance + member disagreement form epistemic uncertainty.
    - Observed outcome variance forms aleatoric uncertainty.
    - Only update() mutates learning state; predict() is read-only.

    Uncertainty outputs are normalized to [0, 1].
    """

    def __init__(
        self,
        members: int = 7,
        seed: int = 1729,
    ):
        if members < 3:
            raise ValueError("Ensemble membutuhkan minimal 3 member")
        self.members = members
        self.seed = seed
        self._member_state: List[Dict[Tuple, EnsembleMemberState]] = [
            {} for _ in range(members)
        ]
        self._actual_counts: Dict[Tuple, int] = {}
        self._actual_success_sum: Dict[Tuple, float] = {}
        self._actual_reward_sum: Dict[Tuple, float] = {}
        self._actual_reward_sum_sq: Dict[Tuple, float] = {}
        self._observation_counter = 0

    @staticmethod
    def _key(
        belief_context_id: Optional[str],
        environment_state: str,
        action: str,
    ) -> Tuple:
        return belief_context_id, environment_state, action

    def count(
        self,
        belief_context_id: Optional[str],
        environment_state: str,
        action: str,
    ) -> int:
        return self._actual_counts.get(
            self._key(
                belief_context_id,
                environment_state,
                action,
            ),
            0,
        )

    def _bootstrap_weight(
        self,
        key: Tuple,
        observation_id: str,
        member_index: int,
    ) -> int:
        payload = (
            f"{self.seed}|{observation_id}|{member_index}|{key!r}"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        uniform = int.from_bytes(digest[:8], "big") / float(2**64)

        # CDF thresholds for a Poisson(1)-like bounded bootstrap weight.
        if uniform < 0.367879:
            return 0
        if uniform < 0.735759:
            return 1
        if uniform < 0.919699:
            return 2
        if uniform < 0.981012:
            return 3
        return 4

    def update(
        self,
        belief_context_id: Optional[str],
        environment_state: str,
        action: str,
        reward: float,
        success: bool,
        observation_id: Optional[str] = None,
    ):
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reward harus di rentang 0..1")

        key = self._key(
            belief_context_id,
            environment_state,
            action,
        )
        self._observation_counter += 1
        resolved_id = (
            observation_id
            if observation_id is not None
            else str(self._observation_counter)
        )

        weights = [
            self._bootstrap_weight(key, resolved_id, index)
            for index in range(self.members)
        ]
        if not any(weights):
            digest = hashlib.sha256(
                f"fallback|{self.seed}|{resolved_id}|{key!r}".encode(
                    "utf-8"
                )
            ).digest()
            weights[digest[0] % self.members] = 1

        for index, weight in enumerate(weights):
            if weight <= 0:
                continue
            state = self._member_state[index].setdefault(
                key,
                EnsembleMemberState(),
            )
            if success:
                state.success_alpha += weight
            else:
                state.success_beta += weight
            state.reward_weight += weight
            state.reward_sum += weight * reward
            state.reward_sum_sq += weight * reward * reward

        self._actual_counts[key] = self._actual_counts.get(key, 0) + 1
        self._actual_success_sum[key] = (
            self._actual_success_sum.get(key, 0.0)
            + float(success)
        )
        self._actual_reward_sum[key] = (
            self._actual_reward_sum.get(key, 0.0)
            + reward
        )
        self._actual_reward_sum_sq[key] = (
            self._actual_reward_sum_sq.get(key, 0.0)
            + reward * reward
        )

    def predict(
        self,
        belief_context_id: Optional[str],
        environment_state: str,
        action: str,
    ) -> EnsemblePrediction:
        key = self._key(
            belief_context_id,
            environment_state,
            action,
        )

        states = [
            member.get(key, EnsembleMemberState())
            for member in self._member_state
        ]
        probabilities = tuple(
            state.success_probability
            for state in states
        )
        rewards = tuple(
            state.reward_mean
            for state in states
        )

        mean_probability = sum(probabilities) / self.members
        mean_reward = sum(rewards) / self.members
        disagreement_variance = sum(
            (value - mean_probability) ** 2
            for value in probabilities
        ) / self.members
        posterior_variance = sum(
            state.success_posterior_variance
            for state in states
        ) / self.members

        epistemic = min(
            1.0,
            2.0 * math.sqrt(
                max(0.0, disagreement_variance + posterior_variance)
            ),
        )

        count = self._actual_counts.get(key, 0)
        if count < 2:
            aleatoric = 0.0
        else:
            empirical_probability = (
                self._actual_success_sum[key] / count
            )
            aleatoric = min(
                1.0,
                2.0 * math.sqrt(
                    max(
                        0.0,
                        empirical_probability
                        * (1.0 - empirical_probability),
                    )
                ),
            )

        return EnsemblePrediction(
            belief_context_id=belief_context_id,
            environment_state=environment_state,
            action=action,
            predicted_reward=max(0.0, min(1.0, mean_reward)),
            success_probability=max(
                0.0,
                min(1.0, mean_probability),
            ),
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            information_gain=epistemic,
            sample_count=count,
            member_success_probabilities=probabilities,
            member_reward_means=rewards,
        )

    def state(self) -> Dict:
        return {
            "members": self.members,
            "actual_observations": sum(self._actual_counts.values()),
            "keys": len(self._actual_counts),
        }


@dataclass(frozen=True)
class ConformalInterval:
    point_prediction: float
    lower: float
    upper: float
    quantile: float
    alpha: float
    calibration_size: int
    source: str

    @property
    def width(self) -> float:
        return self.upper - self.lower


class OnlineConformalCalibrator:
    """
    Leakage-safe online split-conformal residual calibrator.

    interval() reads only past residuals. observe() evaluates the already-built
    interval first, then appends the new nonconformity score.
    """

    GLOBAL_SCOPE = ("__global__",)

    def __init__(
        self,
        alpha: float = 0.10,
        min_calibration: int = 20,
        max_scores: int = 1000,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha harus di (0,1)")
        if min_calibration < 1:
            raise ValueError("min_calibration minimal 1")
        if max_scores < min_calibration:
            raise ValueError("max_scores harus >= min_calibration")
        self.alpha = alpha
        self.min_calibration = min_calibration
        self.max_scores = max_scores
        self._scores: Dict[Hashable, List[float]] = {
            self.GLOBAL_SCOPE: []
        }
        self._metrics: Dict[Hashable, List[float]] = {
            self.GLOBAL_SCOPE: [0.0, 0.0, 0.0]
        }

    def _append_score(self, scope: Hashable, score: float):
        items = self._scores.setdefault(scope, [])
        items.append(score)
        if len(items) > self.max_scores:
            del items[:len(items) - self.max_scores]

    def finite_sample_quantile(self, scores: List[float]) -> float:
        if not scores:
            return 1.0
        ordered = sorted(scores)
        rank = math.ceil(
            (len(ordered) + 1) * (1.0 - self.alpha)
        )
        index = min(len(ordered), max(1, rank)) - 1
        return ordered[index]

    def interval(
        self,
        point_prediction: float,
        scope: Hashable,
    ) -> ConformalInterval:
        point = max(0.0, min(1.0, point_prediction))
        local = self._scores.get(scope, [])
        global_scores = self._scores[self.GLOBAL_SCOPE]

        if len(local) >= self.min_calibration:
            scores = local
            source = "local"
        elif len(global_scores) >= self.min_calibration:
            scores = global_scores
            source = "global_fallback"
        else:
            scores = []
            source = "warmup"

        quantile = self.finite_sample_quantile(scores)
        return ConformalInterval(
            point_prediction=point,
            lower=max(0.0, point - quantile),
            upper=min(1.0, point + quantile),
            quantile=quantile,
            alpha=self.alpha,
            calibration_size=len(scores),
            source=source,
        )

    def observe(
        self,
        point_prediction: float,
        actual: float,
        scope: Hashable,
        interval: Optional[ConformalInterval] = None,
    ) -> Dict:
        if not 0.0 <= actual <= 1.0:
            raise ValueError("actual harus di rentang 0..1")
        prediction_interval = (
            interval
            if interval is not None
            else self.interval(point_prediction, scope)
        )
        covered = (
            prediction_interval.lower
            <= actual
            <= prediction_interval.upper
        )

        metric_scopes = (
            (scope,)
            if scope == self.GLOBAL_SCOPE
            else (scope, self.GLOBAL_SCOPE)
        )
        for metric_scope in metric_scopes:
            metrics = self._metrics.setdefault(
                metric_scope,
                [0.0, 0.0, 0.0],
            )
            metrics[0] += 1.0
            metrics[1] += float(covered)
            metrics[2] += prediction_interval.width

        score = abs(
            max(0.0, min(1.0, point_prediction)) - actual
        )
        self._append_score(scope, score)
        if scope != self.GLOBAL_SCOPE:
            self._append_score(self.GLOBAL_SCOPE, score)

        return {
            "covered": covered,
            "score": score,
            "interval": prediction_interval,
        }

    def state(self, scope: Optional[Hashable] = None) -> Dict:
        resolved_scope = (
            self.GLOBAL_SCOPE
            if scope is None
            else scope
        )
        evaluated, covered, width_sum = self._metrics.get(
            resolved_scope,
            [0.0, 0.0, 0.0],
        )
        return {
            "scope": resolved_scope,
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "calibration_scores": len(
                self._scores.get(resolved_scope, [])
            ),
            "evaluated_intervals": int(evaluated),
            "covered_intervals": int(covered),
            "empirical_coverage": (
                covered / evaluated
                if evaluated else None
            ),
            "average_width": (
                width_sum / evaluated
                if evaluated else None
            ),
        }




# M6C — planning/epistemic world bridge
@dataclass
class WorldDecisionEpisode:
    decision_id: int
    context: str
    selected_action: str
    trajectory: Optional[List[SpaceTimeNode]]
    audits: List[AuditReport]
    outcome: WorldOutcome
    world_signature: Optional[Tuple] = None
    action_instance_id: Optional[str] = None
    state_key: Optional[str] = None

    # V2.27
    objective_profile_instance_id: Optional[str] = None
    scalar_state_key: Optional[str] = None


class WorldOutcomeEvaluator:
    """
    Mengubah konsekuensi planner menjadi reward bounded [0,1].

    Penting:
    - reward adalah utility/action outcome, bukan truth score
    - critical failure => reward 0
    - safety dan efficiency tetap dilaporkan terpisah
    """

    def evaluate_route(
        self,
        start: Point,
        goal: Point,
        trajectory: Optional[List[SpaceTimeNode]],
        audits: List[AuditReport],
    ) -> WorldOutcome:
        reasons: List[str] = []

        reached_goal = bool(
            trajectory
            and trajectory[-1].p == goal
        )

        critical_failure = any(
            (not audit.passed)
            and audit.risk_level == "CRITICAL"
            for audit in audits
        )

        success = reached_goal and not critical_failure

        # Safety bersifat konservatif: ambil tingkat terburuk.
        safety_score = 1.0
        for audit in audits:
            if audit.passed:
                continue

            reasons.append(audit.challenge_name)

            if audit.risk_level == "CRITICAL":
                safety_score = 0.0
            elif audit.risk_level == "HIGH":
                safety_score = min(safety_score, 0.20)
            elif audit.risk_level == "MEDIUM":
                safety_score = min(safety_score, 0.60)
            else:
                safety_score = min(safety_score, 0.80)

        # Efficiency relatif terhadap lintasan Manhattan minimum.
        if not trajectory:
            efficiency_score = 0.0
        else:
            actual_transitions = max(0, len(trajectory) - 1)
            optimal_transitions = start.manhattan(goal)

            if optimal_transitions == 0:
                efficiency_score = 1.0
            elif actual_transitions <= 0:
                efficiency_score = 0.0
            else:
                efficiency_score = min(
                    1.0,
                    optimal_transitions / actual_transitions,
                )

        # Critical / gagal mencapai goal selalu 0.
        if not success:
            reward = 0.0
        else:
            reward = (
                0.65 * safety_score
                + 0.35 * efficiency_score
            )

        reward = max(0.0, min(1.0, reward))

        if not reasons:
            reasons.append(
                "Goal_Reached"
                if success
                else "Goal_Not_Reached"
            )

        return WorldOutcome(
            success=success,
            reached_goal=reached_goal,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            reward=reward,
            reasons=tuple(reasons),
        )


@dataclass
class CounterfactualEstimate:
    counterfactual_id: int
    context: str
    action_name: str
    world_signature: Tuple
    trajectory: Optional[List[SpaceTimeNode]]
    audits: List[AuditReport]
    predicted_outcome: WorldOutcome
    belief_context_id: Optional[str] = None
    action_instance_id: Optional[str] = None
    state_key: Optional[str] = None


@dataclass
class StrategyExecution:
    decision: DecisionRecord
    counterfactuals: Dict[str, CounterfactualEstimate]
    actual_episode: WorldDecisionEpisode
    prediction_error: Optional[PredictionErrorRecord] = None



_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"
_PICKLE_COMPAT_CLASSES = (
    WorldOutcome,
    PredictionErrorRecord,
    PredictionUncertainty,
    PredictionUncertaintyEstimator,
    OutcomePrediction,
    ContextScopedWorldModel,
    WorldModelReliability,
    PredictionErrorEvaluator,
    CounterfactualStrategyPolicy,
    EnsembleMemberState,
    EnsemblePrediction,
    OnlineBootstrapEnsemble,
    ConformalInterval,
    OnlineConformalCalibrator,
    WorldDecisionEpisode,
    WorldOutcomeEvaluator,
    CounterfactualEstimate,
    StrategyExecution,
)
for _cls in _PICKLE_COMPAT_CLASSES:
    _cls.__module__ = _CANONICAL_PICKLE_MODULE
del _cls

__all__ = [
    "WorldOutcome",
    "PredictionErrorRecord",
    "PredictionUncertainty",
    "PredictionUncertaintyEstimator",
    "OutcomePrediction",
    "ContextScopedWorldModel",
    "WorldModelReliability",
    "PredictionErrorEvaluator",
    "CounterfactualStrategyPolicy",
    "EnsembleMemberState",
    "EnsemblePrediction",
    "OnlineBootstrapEnsemble",
    "ConformalInterval",
    "OnlineConformalCalibrator",
    "WorldDecisionEpisode", "WorldOutcomeEvaluator",
    "CounterfactualEstimate", "StrategyExecution",
    "CounterfactualMemory", "PredictionMemory", "PredictionErrorMemory",
]
