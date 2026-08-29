"""Objective subsystem — physically extracted in modularization M2.

This module contains the real V2.28 implementation for:
- structured objective outcomes;
- immutable/versioned objective profiles;
- scalar objective aggregation;
- marginal objective statistics;
- joint moments/covariance with exact missingness semantics;
- profile-independent technical success statistics.

Compatibility note
------------------
The implementation lives here physically, but class ``__module__`` values are
kept as ``agen_kognitif_v2_28`` so trusted-local V2.28 pickle/checkpoint
identity remains compatible with the pre-modular monolith.  The compatibility
kernel imports and re-exports these exact class objects.
"""
from __future__ import annotations

import hashlib
import json
import math

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

OBJECTIVE_COMPONENTS = (
    "task_progress",
    "correctness",
    "execution_cost",
    "reversibility",
    "user_acceptance",
)

OBJECTIVE_BENEFIT_COMPONENTS = frozenset({
    "task_progress",
    "correctness",
    "reversibility",
    "user_acceptance",
})


@dataclass(frozen=True)
class ObjectiveOutcome:
    """Structured ACTUAL task outcome. Missing component means unknown."""
    task_progress: Optional[float] = None
    correctness: Optional[float] = None
    execution_cost: Optional[float] = None
    reversibility: Optional[float] = None
    user_acceptance: Optional[float] = None

    def __post_init__(self):
        for name in OBJECTIVE_COMPONENTS:
            value = getattr(self, name)
            if value is None:
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"objective component {name} harus 0..1 atau None"
                )
        if not self.observed_components():
            raise ValueError(
                "ObjectiveOutcome harus memiliki minimal satu component"
            )

    def observed_components(self) -> Dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in OBJECTIVE_COMPONENTS
            if getattr(self, name) is not None
        }

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            name: (
                None
                if getattr(self, name) is None
                else float(getattr(self, name))
            )
            for name in OBJECTIVE_COMPONENTS
        }

    @classmethod
    def coerce(cls, value) -> "ObjectiveOutcome":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            unknown = set(value) - set(OBJECTIVE_COMPONENTS)
            if unknown:
                raise ValueError(
                    f"objective component tidak dikenal: {sorted(unknown)}"
                )
            return cls(**value)
        raise TypeError(
            "objective_outcome harus ObjectiveOutcome atau dict"
        )


@dataclass(frozen=True)
class ObjectiveUtilityProfile:
    """
    Immutable scalarization profile with explicit lifecycle identity.

    profile_id:
        stable logical preference family.

    profile_version:
        immutable version inside that family.

    valid_from / valid_until:
        audit lifecycle only. Preference versioning is independent from
        BeliefContext and does NOT create an epistemic regime shift.

    `signature` intentionally describes semantic weights, not lifecycle
    identity. `instance_id` identifies the exact version.
    """
    profile_id: str = "default_balanced_objective"
    task_progress_weight: float = 0.30
    correctness_weight: float = 0.30
    execution_cost_weight: float = 0.15
    reversibility_weight: float = 0.10
    user_acceptance_weight: float = 0.15

    # V2.27 lifecycle identity.
    profile_version: Optional[int] = None
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    note: str = ""

    def __post_init__(self):
        if not self.profile_id:
            raise ValueError(
                "profile_id tidak boleh kosong"
            )

        weights = self.weights()
        for name, value in weights.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"weight {name} harus 0..1"
                )

        if abs(
            sum(weights.values()) - 1.0
        ) > 1e-9:
            raise ValueError(
                "objective weights harus berjumlah 1.0"
            )

        if (
            self.profile_version is not None
            and self.profile_version < 1
        ):
            raise ValueError(
                "profile_version harus >= 1"
            )

        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError(
                "valid_until profile harus > valid_from"
            )

    def weights(self) -> Dict[str, float]:
        return {
            "task_progress":
                self.task_progress_weight,
            "correctness":
                self.correctness_weight,
            "execution_cost":
                self.execution_cost_weight,
            "reversibility":
                self.reversibility_weight,
            "user_acceptance":
                self.user_acceptance_weight,
        }

    @property
    def signature(self) -> str:
        payload = json.dumps(
            {
                "profile_id":
                    self.profile_id,
                "weights":
                    self.weights(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return (
            f"{self.profile_id}:"
            f"{hashlib.sha256(payload).hexdigest()[:16]}"
        )

    @property
    def instance_id(self) -> str:
        version = (
            self.profile_version
            if self.profile_version is not None
            else "unregistered"
        )
        return (
            f"{self.profile_id}@v{version}"
        )

    def applies_to(
        self,
        as_of: Optional[int] = None,
    ) -> bool:
        if as_of is None:
            return (
                self.valid_until is None
            )

        if (
            self.valid_from is not None
            and as_of < self.valid_from
        ):
            return False
        if (
            self.valid_until is not None
            and as_of >= self.valid_until
        ):
            return False
        return True


class ObjectiveProfileVersionConflict(ValueError):
    pass


class ObjectiveProfileRegistry:
    """
    Versioned preference registry.

    Actual objective vectors are not stored here and never change when a
    preference profile changes. This registry only versions the scalarization
    policy used to interpret those vectors.
    """

    def __init__(self):
        self.versions: Dict[
            str,
            List[ObjectiveUtilityProfile],
        ] = {}
        self.active_instance_by_family: Dict[
            str,
            str,
        ] = {}

    @staticmethod
    def _same_semantics(
        left: ObjectiveUtilityProfile,
        right: ObjectiveUtilityProfile,
    ) -> bool:
        return (
            left.profile_id
                == right.profile_id
            and left.weights()
                == right.weights()
        )

    def all_versions(
        self,
        profile_id: str,
    ) -> List[ObjectiveUtilityProfile]:
        return list(
            self.versions.get(
                profile_id,
                [],
            )
        )

    def get_version(
        self,
        profile_id: str,
        profile_version: int,
    ) -> Optional[ObjectiveUtilityProfile]:
        for profile in self.versions.get(
            profile_id,
            [],
        ):
            if (
                profile.profile_version
                == profile_version
            ):
                return profile
        return None

    def get_instance(
        self,
        instance_id: str,
    ) -> Optional[ObjectiveUtilityProfile]:
        if "@v" not in instance_id:
            return None
        family, raw_version = (
            instance_id.rsplit(
                "@v",
                1,
            )
        )
        try:
            version = int(
                raw_version
            )
        except ValueError:
            return None
        return self.get_version(
            family,
            version,
        )

    def register(
        self,
        profile: ObjectiveUtilityProfile,
        activated_at: Optional[int] = None,
    ) -> ObjectiveUtilityProfile:
        family = profile.profile_id
        existing = self.versions.setdefault(
            family,
            [],
        )

        # Exact semantic/lifecycle registration is idempotent.
        for old in existing:
            if (
                self._same_semantics(
                    old,
                    profile,
                )
                and old.valid_from
                    == (
                        profile.valid_from
                        if profile.valid_from is not None
                        else activated_at
                    )
                and old.valid_until
                    == profile.valid_until
                and (
                    profile.profile_version is None
                    or old.profile_version
                        == profile.profile_version
                )
            ):
                return old

        version = (
            profile.profile_version
            if profile.profile_version is not None
            else (
                max(
                    (
                        p.profile_version or 0
                        for p in existing
                    ),
                    default=0,
                )
                + 1
            )
        )

        if any(
            p.profile_version == version
            for p in existing
        ):
            raise ObjectiveProfileVersionConflict(
                "Objective profile version collision: "
                f"{family}@v{version}"
            )

        valid_from = (
            profile.valid_from
            if profile.valid_from is not None
            else activated_at
        )

        candidate = replace(
            profile,
            profile_version=version,
            valid_from=valid_from,
        )

        # A family has one active preference version at a time.
        active = [
            p
            for p in existing
            if p.valid_until is None
        ]
        if active:
            raise ObjectiveProfileVersionConflict(
                "Objective profile family sudah memiliki versi aktif: "
                f"{active[-1].instance_id}. "
                "Gunakan supersede()."
            )

        existing.append(
            candidate
        )
        existing.sort(
            key=lambda p: (
                p.profile_version or 0
            )
        )
        self.active_instance_by_family[
            family
        ] = candidate.instance_id
        return candidate

    def active(
        self,
        profile_id: str,
    ) -> ObjectiveUtilityProfile:
        instance = (
            self.active_instance_by_family.get(
                profile_id
            )
        )
        if instance is None:
            raise KeyError(
                f"Objective profile family '{profile_id}' tidak memiliki versi aktif"
            )
        profile = self.get_instance(
            instance
        )
        if profile is None:
            raise KeyError(
                f"Objective profile instance '{instance}' tidak ditemukan"
            )
        return profile

    def resolve(
        self,
        reference: str,
    ) -> ObjectiveUtilityProfile:
        exact = self.get_instance(
            reference
        )
        if exact is not None:
            return exact

        if reference in self.versions:
            return self.active(
                reference
            )

        raise KeyError(
            f"Objective profile '{reference}' tidak ditemukan"
        )

    def supersede(
        self,
        current_reference: str,
        successor: ObjectiveUtilityProfile,
        observed_at: int,
    ) -> Tuple[
        ObjectiveUtilityProfile,
        ObjectiveUtilityProfile,
    ]:
        current = self.resolve(
            current_reference
        )

        if (
            current.profile_id
            != successor.profile_id
        ):
            raise ObjectiveProfileVersionConflict(
                "Successor harus berada pada profile_id family yang sama"
            )

        if (
            current.valid_from is not None
            and observed_at <= current.valid_from
        ):
            raise ValueError(
                "observed_at supersession harus setelah valid_from profile"
            )

        family = current.profile_id
        versions = self.versions[
            family
        ]

        closed = replace(
            current,
            valid_until=observed_at,
        )
        for index, profile in enumerate(
            versions
        ):
            if (
                profile.profile_version
                == current.profile_version
            ):
                versions[index] = closed
                break

        self.active_instance_by_family.pop(
            family,
            None,
        )

        successor = replace(
            successor,
            profile_version=None,
            valid_from=observed_at,
            valid_until=None,
        )
        registered = self.register(
            successor,
            activated_at=observed_at,
        )
        return closed, registered

    def state(self) -> Dict:
        families = {}
        for family in sorted(
            self.versions
        ):
            families[family] = [
                {
                    "profile_id":
                        p.profile_id,
                    "profile_version":
                        p.profile_version,
                    "instance_id":
                        p.instance_id,
                    "signature":
                        p.signature,
                    "weights":
                        p.weights(),
                    "valid_from":
                        p.valid_from,
                    "valid_until":
                        p.valid_until,
                    "note":
                        p.note,
                    "active": (
                        self.active_instance_by_family.get(
                            family
                        )
                        == p.instance_id
                    ),
                }
                for p in self.versions[
                    family
                ]
            ]

        return {
            "families": families,
            "active_instance_by_family":
                dict(
                    self.active_instance_by_family
                ),
            "version_count": sum(
                len(items)
                for items
                in families.values()
            ),
        }


@dataclass(frozen=True)
class ObjectiveAggregation:
    profile_id: str
    profile_signature: str
    scalar_utility: float
    observed_components: Tuple[str, ...]
    effective_weights: Dict[str, float]
    utility_components: Dict[str, float]
    raw_components: Dict[str, float]

    def as_dict(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "profile_signature": self.profile_signature,
            "scalar_utility": self.scalar_utility,
            "observed_components": tuple(self.observed_components),
            "effective_weights": dict(self.effective_weights),
            "utility_components": dict(self.utility_components),
            "raw_components": dict(self.raw_components),
        }


class ObjectiveUtilityAggregator:
    def __init__(self, profile=None):
        self.profile = (
            profile
            if profile is not None
            else ObjectiveUtilityProfile()
        )

    def aggregate(self, outcome) -> ObjectiveAggregation:
        outcome = ObjectiveOutcome.coerce(outcome)
        raw = outcome.observed_components()
        configured = self.profile.weights()

        active_weight = sum(configured[name] for name in raw)
        if active_weight <= 0:
            raise ValueError(
                "observed objective components memiliki total weight 0"
            )

        effective_weights = {
            name: configured[name] / active_weight
            for name in raw
        }
        utility_components = {
            name: (
                value
                if name in OBJECTIVE_BENEFIT_COMPONENTS
                else 1.0 - value
            )
            for name, value in raw.items()
        }
        scalar = sum(
            effective_weights[name] * utility_components[name]
            for name in raw
        )
        scalar = max(0.0, min(1.0, scalar))

        return ObjectiveAggregation(
            profile_id=self.profile.profile_id,
            profile_signature=self.profile.signature,
            scalar_utility=scalar,
            observed_components=tuple(
                name for name in OBJECTIVE_COMPONENTS if name in raw
            ),
            effective_weights=effective_weights,
            utility_components=utility_components,
            raw_components=raw,
        )


class ContextScopedObjectiveModel:
    """Actual per-component statistics, scoped by context/state/action."""

    def __init__(self):
        self._stats: Dict[
            Tuple[Optional[str], str, str, str],
            List[float],
        ] = {}

    @staticmethod
    def _key(context, action_name, component, belief_context_id):
        return (
            belief_context_id,
            context,
            action_name,
            component,
        )

    def update(
        self,
        context: str,
        action_name: str,
        outcome,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        outcome = ObjectiveOutcome.coerce(outcome)
        updated = {}
        for component, value in outcome.observed_components().items():
            key = self._key(
                context,
                action_name,
                component,
                belief_context_id,
            )
            values = self._stats.setdefault(
                key,
                [0.0, 0.0, 0.0],
            )
            values[0] += value
            values[1] += value * value
            values[2] += 1.0
            updated[component] = self.component_statistics(
                context,
                action_name,
                component,
                belief_context_id,
            )
        return updated

    def component_statistics(
        self,
        context: str,
        action_name: str,
        component: str,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        if component not in OBJECTIVE_COMPONENTS:
            raise ValueError(
                f"objective component tidak dikenal: {component}"
            )
        values = self._stats.get(
            self._key(
                context,
                action_name,
                component,
                belief_context_id,
            )
        )
        if values is None:
            return {
                "mean": None,
                "variance": None,
                "std": None,
                "count": 0,
            }

        total, squared, raw_count = values
        count = int(raw_count)
        mean = total / count
        if count < 2:
            variance = None
            std = None
        else:
            variance = max(
                0.0,
                (
                    squared
                    - (total * total) / count
                ) / (count - 1),
            )
            std = math.sqrt(variance)

        return {
            "mean": mean,
            "variance": variance,
            "std": std,
            "count": count,
        }

    def statistics(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
    ) -> Dict[str, Dict]:
        return {
            component: self.component_statistics(
                context,
                action_name,
                component,
                belief_context_id,
            )
            for component in OBJECTIVE_COMPONENTS
        }

    def predicted_means(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
    ) -> Dict[str, float]:
        return {
            component: values["mean"]
            for component, values in self.statistics(
                context,
                action_name,
                belief_context_id,
            ).items()
            if values["count"] > 0
        }

    def state(
        self,
        belief_context_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict:
        output = {}
        for (
            scope,
            state_key,
            action,
            component,
        ), values in self._stats.items():
            if (
                belief_context_id is not None
                and scope != belief_context_id
            ):
                continue
            if (
                context is not None
                and state_key != context
            ):
                continue
            output[
                (scope, state_key, action, component)
            ] = list(values)
        return output


class ContextScopedJointObjectiveModel:
    """
    V2.28 exact joint objective sufficient statistics.

    The V2.26 component model stores independent per-component moments.
    That is insufficient for exact preference reweighting when:
    - objective components covary;
    - records have different missing-component masks;
    - profile weights change after experience was collected.

    This model therefore keeps joint moments per observed-component mask:

        belief_context × canonical_state × action_instance × component_mask

    With five objective components there are at most 2^5 - 1 = 31 non-empty
    masks per state/action. Storage therefore grows with distinct masks, not
    linearly with repeated experiences.

    For each mask we keep:
    - count
    - sum(x_i)
    - sum(x_i * x_j), including diagonal squares

    These sufficient statistics are enough to reconstruct the exact sample
    mean/variance of any later linear scalarization whose normalization is
    determined by that mask, without storing every raw outcome.
    """

    MAX_MASKS_PER_STATE_ACTION = (
        (1 << len(OBJECTIVE_COMPONENTS))
        - 1
    )

    def __init__(self):
        self._groups: Dict[
            Tuple[
                Optional[str],
                str,
                str,
                Tuple[str, ...],
            ],
            Dict,
        ] = {}

    @staticmethod
    def _mask_from_outcome(
        outcome: ObjectiveOutcome,
    ) -> Tuple[str, ...]:
        observed = outcome.observed_components()
        return tuple(
            name
            for name in OBJECTIVE_COMPONENTS
            if name in observed
        )

    @staticmethod
    def _key(
        context: str,
        action_name: str,
        mask: Tuple[str, ...],
        belief_context_id: Optional[str],
    ):
        return (
            belief_context_id,
            context,
            action_name,
            tuple(mask),
        )

    @staticmethod
    def _pair_key(
        left: str,
        right: str,
    ) -> Tuple[str, str]:
        order = {
            name: index
            for index, name
            in enumerate(
                OBJECTIVE_COMPONENTS
            )
        }
        return (
            (left, right)
            if order[left] <= order[right]
            else (right, left)
        )

    def update(
        self,
        context: str,
        action_name: str,
        outcome,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        outcome = ObjectiveOutcome.coerce(
            outcome
        )
        observed = (
            outcome.observed_components()
        )
        mask = self._mask_from_outcome(
            outcome
        )
        key = self._key(
            context,
            action_name,
            mask,
            belief_context_id,
        )

        group = self._groups.get(
            key
        )
        if group is None:
            group = {
                "count": 0,
                "sums": {
                    component: 0.0
                    for component in mask
                },
                "cross_sums": {
                    (left, right): 0.0
                    for i, left
                    in enumerate(mask)
                    for right
                    in mask[i:]
                },
            }
            self._groups[key] = group

        group["count"] += 1

        for component in mask:
            group["sums"][
                component
            ] += observed[
                component
            ]

        for i, left in enumerate(
            mask
        ):
            for right in mask[i:]:
                group["cross_sums"][
                    (left, right)
                ] += (
                    observed[left]
                    * observed[right]
                )

        return self.group_statistics(
            context,
            action_name,
            mask,
            belief_context_id,
        )

    def group_statistics(
        self,
        context: str,
        action_name: str,
        mask: Tuple[str, ...],
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        mask = tuple(mask)
        group = self._groups.get(
            self._key(
                context,
                action_name,
                mask,
                belief_context_id,
            )
        )

        if group is None:
            return {
                "mask": mask,
                "count": 0,
                "means": {},
                "variances": {},
                "covariances": {},
            }

        count = int(
            group["count"]
        )
        means = {
            component: (
                group["sums"][
                    component
                ] / count
            )
            for component in mask
        }

        variances = {}
        covariances = {}

        for i, left in enumerate(
            mask
        ):
            for right in mask[i:]:
                cross = (
                    group[
                        "cross_sums"
                    ][
                        (left, right)
                    ]
                )

                if count < 2:
                    covariance = None
                else:
                    numerator = (
                        cross
                        - (
                            group["sums"][
                                left
                            ]
                            * group["sums"][
                                right
                            ]
                        ) / count
                    )
                    covariance = (
                        numerator
                        / (count - 1)
                    )
                    if (
                        left == right
                        and covariance < 0
                        and abs(
                            covariance
                        ) < 1e-15
                    ):
                        covariance = 0.0

                pair = (
                    left,
                    right,
                )
                covariances[
                    pair
                ] = covariance

                if left == right:
                    variances[
                        left
                    ] = (
                        None
                        if covariance is None
                        else max(
                            0.0,
                            covariance,
                        )
                    )

        return {
            "mask": mask,
            "count": count,
            "means": means,
            "variances":
                variances,
            "covariances":
                covariances,
        }

    @staticmethod
    def _profile_linear_form(
        mask: Tuple[str, ...],
        profile: ObjectiveUtilityProfile,
    ) -> Optional[
        Tuple[
            float,
            Dict[str, float],
            Dict[str, float],
        ]
    ]:
        """
        Return scalarization as:

            utility = intercept + sum(coeff_i * x_i)

        Cost components contribute:
            w * (1 - x)
        so execution_cost has a negative coefficient plus positive intercept.

        Missing-component renormalization is exact because the mask is fixed
        for every record inside one group.
        """
        configured = (
            profile.weights()
        )
        active_weight = sum(
            configured[
                component
            ]
            for component in mask
        )

        if active_weight <= 0:
            return None

        effective = {
            component: (
                configured[
                    component
                ]
                / active_weight
            )
            for component in mask
        }

        intercept = 0.0
        coefficients = {}

        for component in mask:
            weight = effective[
                component
            ]
            if (
                component
                in OBJECTIVE_BENEFIT_COMPONENTS
            ):
                coefficients[
                    component
                ] = weight
            else:
                # execution_cost
                intercept += weight
                coefficients[
                    component
                ] = -weight

        return (
            intercept,
            coefficients,
            effective,
        )

    @staticmethod
    def _group_scalar_raw_moments(
        group: Dict,
        mask: Tuple[str, ...],
        profile: ObjectiveUtilityProfile,
    ) -> Optional[Dict]:
        form = (
            ContextScopedJointObjectiveModel
            ._profile_linear_form(
                mask,
                profile,
            )
        )
        if form is None:
            return None

        (
            intercept,
            coefficients,
            effective_weights,
        ) = form

        count = int(
            group["count"]
        )

        linear_sum = sum(
            coefficients[
                component
            ]
            * group["sums"][
                component
            ]
            for component in mask
        )

        scalar_sum = (
            count
            * intercept
            + linear_sum
        )

        scalar_square_sum = (
            count
            * intercept
            * intercept
            + 2.0
            * intercept
            * linear_sum
        )

        # Diagonal and off-diagonal quadratic terms.
        for i, left in enumerate(
            mask
        ):
            left_coeff = (
                coefficients[
                    left
                ]
            )
            for right in mask[i:]:
                right_coeff = (
                    coefficients[
                        right
                    ]
                )
                cross_sum = (
                    group[
                        "cross_sums"
                    ][
                        (left, right)
                    ]
                )

                factor = (
                    1.0
                    if left == right
                    else 2.0
                )
                scalar_square_sum += (
                    factor
                    * left_coeff
                    * right_coeff
                    * cross_sum
                )

        mean = (
            scalar_sum / count
            if count > 0
            else None
        )

        if count < 2:
            variance = None
            std = None
        else:
            variance = max(
                0.0,
                (
                    scalar_square_sum
                    - (
                        scalar_sum
                        * scalar_sum
                    ) / count
                )
                / (count - 1),
            )
            if variance < 1e-15:
                variance = 0.0
            std = math.sqrt(
                variance
            )

        return {
            "count": count,
            "sum": scalar_sum,
            "square_sum":
                scalar_square_sum,
            "mean": mean,
            "variance": variance,
            "std": std,
            "intercept": intercept,
            "coefficients":
                coefficients,
            "effective_weights":
                effective_weights,
        }

    def reweighted_distribution(
        self,
        context: str,
        action_name: str,
        profile: ObjectiveUtilityProfile,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        """
        Exact historical scalar-utility distribution under `profile`.

        Exactness includes the original per-record missingness semantics:
        each component mask is renormalized independently exactly as
        ObjectiveUtilityAggregator would have done on the raw record.

        If a mask contains only components whose new profile weights are zero,
        that group is unscorable under the new preference and is reported
        explicitly rather than silently interpreted as utility 0.
        """
        matching = []

        for (
            scope,
            state_key,
            action,
            mask,
        ), group in self._groups.items():
            if (
                scope
                != belief_context_id
                or state_key != context
                or action
                    != action_name
            ):
                continue
            matching.append(
                (
                    mask,
                    group,
                )
            )

        total_count = sum(
            int(
                group["count"]
            )
            for _, group
            in matching
        )

        scorable_count = 0
        unscorable_count = 0
        scalar_sum = 0.0
        scalar_square_sum = 0.0
        mask_breakdown = []

        for (
            mask,
            group,
        ) in sorted(
            matching,
            key=lambda item:
                item[0],
        ):
            moments = (
                self._group_scalar_raw_moments(
                    group,
                    mask,
                    profile,
                )
            )

            if moments is None:
                count = int(
                    group["count"]
                )
                unscorable_count += (
                    count
                )
                mask_breakdown.append({
                    "mask":
                        tuple(mask),
                    "count":
                        count,
                    "scorable":
                        False,
                    "reason":
                        "active_profile_weight_zero_for_mask",
                })
                continue

            scorable_count += (
                moments[
                    "count"
                ]
            )
            scalar_sum += (
                moments[
                    "sum"
                ]
            )
            scalar_square_sum += (
                moments[
                    "square_sum"
                ]
            )

            mask_breakdown.append({
                "mask":
                    tuple(mask),
                "count":
                    moments["count"],
                "scorable":
                    True,
                "mean":
                    moments["mean"],
                "variance":
                    moments[
                        "variance"
                    ],
                "std":
                    moments["std"],
                "effective_weights":
                    dict(
                        moments[
                            "effective_weights"
                        ]
                    ),
            })

        if scorable_count <= 0:
            mean = None
            variance = None
            std = None
        else:
            mean = (
                scalar_sum
                / scorable_count
            )

            if scorable_count < 2:
                variance = None
                std = None
            else:
                variance = max(
                    0.0,
                    (
                        scalar_square_sum
                        - (
                            scalar_sum
                            * scalar_sum
                        )
                        / scorable_count
                    )
                    / (
                        scorable_count
                        - 1
                    ),
                )
                if variance < 1e-15:
                    variance = 0.0
                std = math.sqrt(
                    variance
                )

        return {
            "profile_id":
                profile.profile_id,
            "profile_version":
                profile.profile_version,
            "profile_instance_id":
                profile.instance_id,
            "profile_signature":
                profile.signature,
            "total_count":
                total_count,
            "scorable_count":
                scorable_count,
            "unscorable_count":
                unscorable_count,
            "coverage": (
                (
                    scorable_count
                    / total_count
                )
                if total_count > 0
                else 0.0
            ),
            "mask_count":
                len(matching),
            "scorable_mask_count":
                sum(
                    1
                    for item
                    in mask_breakdown
                    if item[
                        "scorable"
                    ]
                ),
            "mean":
                mean,
            "variance":
                variance,
            "std":
                std,
            "sum":
                scalar_sum,
            "square_sum":
                scalar_square_sum,
            "mask_breakdown":
                mask_breakdown,
            "exact_missingness":
                True,
            "learning_mutation":
                False,
        }

    def mask_statistics(
        self,
        context: str,
        action_name: str,
        belief_context_id: Optional[str] = None,
    ) -> List[Dict]:
        output = []
        for (
            scope,
            state_key,
            action,
            mask,
        ), _ in sorted(
            self._groups.items(),
            key=lambda item:
                (
                    item[0][0] or "",
                    item[0][1],
                    item[0][2],
                    item[0][3],
                ),
        ):
            if (
                scope
                != belief_context_id
                or state_key != context
                or action
                    != action_name
            ):
                continue

            output.append(
                self.group_statistics(
                    context,
                    action_name,
                    mask,
                    belief_context_id,
                )
            )
        return output

    def group_count(
        self,
        context: Optional[str] = None,
        action_name: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> int:
        count = 0
        for (
            scope,
            state_key,
            action,
            _,
        ) in self._groups:
            if (
                belief_context_id
                is not None
                and scope
                    != belief_context_id
            ):
                continue
            if (
                context is not None
                and state_key
                    != context
            ):
                continue
            if (
                action_name is not None
                and action
                    != action_name
            ):
                continue
            count += 1
        return count

    def state(
        self,
        belief_context_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict:
        output = {}

        for (
            scope,
            state_key,
            action,
            mask,
        ), group in self._groups.items():
            if (
                belief_context_id
                is not None
                and scope
                    != belief_context_id
            ):
                continue
            if (
                context is not None
                and state_key
                    != context
            ):
                continue

            output[
                (
                    scope,
                    state_key,
                    action,
                    mask,
                )
            ] = {
                "count":
                    int(
                        group[
                            "count"
                        ]
                    ),
                "sums":
                    dict(
                        group[
                            "sums"
                        ]
                    ),
                "cross_sums":
                    dict(
                        group[
                            "cross_sums"
                        ]
                    ),
            }

        return output


class ContextScopedSuccessConstraintModel:
    """
    Profile-independent technical success statistics.

    Preference changes may reinterpret utility, cost, correctness tradeoffs,
    etc. They must NOT erase whether an intervention historically executed /
    succeeded. This model therefore scopes only by:

        belief_context × canonical_state × action_instance
    """

    def __init__(self):
        self._stats: Dict[
            Tuple[
                Optional[str],
                str,
                str,
            ],
            List[float],
        ] = {}

    @staticmethod
    def _key(
        context: str,
        action_name: str,
        belief_context_id: Optional[str],
    ):
        return (
            belief_context_id,
            context,
            action_name,
        )

    def update(
        self,
        context: str,
        action_name: str,
        success: bool,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        key = self._key(
            context,
            action_name,
            belief_context_id,
        )
        values = self._stats.setdefault(
            key,
            [0.0, 0.0],
        )
        values[0] += float(
            bool(success)
        )
        values[1] += 1.0

        return self.statistics(
            context,
            action_name,
            belief_context_id,
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
        if values is None:
            return {
                "success_probability": 0.5,
                "count": 0,
            }

        success_sum, raw_count = (
            values
        )
        count = int(
            raw_count
        )
        return {
            "success_probability": (
                success_sum / count
                if count > 0
                else 0.5
            ),
            "count": count,
        }

    def state(
        self,
        belief_context_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Dict:
        output = {}

        for (
            scope,
            state_key,
            action_name,
        ), _ in self._stats.items():
            if (
                belief_context_id
                is not None
                and scope
                    != belief_context_id
            ):
                continue
            if (
                context is not None
                and state_key != context
            ):
                continue

            key = (
                (state_key, action_name)
                if context is None
                else action_name
            )
            output[key] = {
                "belief_context_id":
                    scope,
                **self.statistics(
                    state_key,
                    action_name,
                    scope,
                ),
            }

        return output


# -------------------------------------------------------------------------
# Trusted-local checkpoint compatibility.
# -------------------------------------------------------------------------
# Existing V2.28 checkpoints encode these classes under the canonical module
# ``agen_kognitif_v2_28``.  Physical source location may change, but serialized
# identity must not change during refactor-only modularization.
_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"

_PICKLE_COMPAT_CLASSES = (
    ObjectiveOutcome,
    ObjectiveUtilityProfile,
    ObjectiveProfileVersionConflict,
    ObjectiveProfileRegistry,
    ObjectiveAggregation,
    ObjectiveUtilityAggregator,
    ContextScopedObjectiveModel,
    ContextScopedJointObjectiveModel,
    ContextScopedSuccessConstraintModel,
)

for _cls in _PICKLE_COMPAT_CLASSES:
    _cls.__module__ = _CANONICAL_PICKLE_MODULE

del _cls

__all__ = [
    "OBJECTIVE_COMPONENTS",
    "OBJECTIVE_BENEFIT_COMPONENTS",
    "ObjectiveOutcome",
    "ObjectiveUtilityProfile",
    "ObjectiveProfileVersionConflict",
    "ObjectiveProfileRegistry",
    "ObjectiveAggregation",
    "ObjectiveUtilityAggregator",
    "ContextScopedObjectiveModel",
    "ContextScopedJointObjectiveModel",
    "ContextScopedSuccessConstraintModel",
]
