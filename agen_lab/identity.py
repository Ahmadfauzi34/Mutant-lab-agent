"""Identity subsystem — physically extracted in modularization M3.

This module owns the real V2.28 implementation for:
- explicit canonical state identity / aliases;
- immutable action implementation identity / version registry.

Objective-profile identity remains physically owned by ``agen_lab.objectives``.
It is re-exported here only as a convenience because the public identity seam
historically grouped all identity concepts together.

Dependency rule
---------------
This module is standalone with respect to the compatibility kernel. It may use
stdlib and ``agen_lab.objectives`` only for preference-profile re-exports.

Trusted-local checkpoint compatibility
--------------------------------------
Extracted State/Action classes retain serialized identity under
``agen_kognitif_v2_28``. The compatibility kernel imports and re-exports these
exact class objects.
"""
from __future__ import annotations

import json

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class StateCanonicalDefinition:
    """
    Explicit immutable equivalence class for environment-state strings.

    canonical_id is the learning identity. equivalence_fingerprint documents
    WHY aliases are considered semantically equivalent. The core never guesses
    equivalence from formatting heuristics.
    """
    canonical_id: str
    equivalence_fingerprint: str
    note: str = ""

    def __post_init__(self):
        if not self.canonical_id:
            raise ValueError(
                "canonical_id state tidak boleh kosong"
            )
        if not self.equivalence_fingerprint:
            raise ValueError(
                "equivalence_fingerprint state tidak boleh kosong"
            )


@dataclass(frozen=True)
class ResolvedStateIdentity:
    reference: str
    canonical_id: str
    registered: bool
    equivalence_fingerprint: Optional[str] = None


class StateIdentityConflict(ValueError):
    pass


class StateIdentityRegistry:
    """
    Adapter-controlled exact alias registry.

    Safety properties:
    - no heuristic global merging;
    - aliases are immutable once registered;
    - one canonical_id has one immutable equivalence fingerprint;
    - changing equivalence semantics requires a NEW canonical_id.

    Unregistered states remain legacy-compatible: raw string == learning key.
    """

    def __init__(self):
        self.definitions: Dict[
            str,
            StateCanonicalDefinition,
        ] = {}
        self.alias_to_canonical: Dict[str, str] = {}

    def register(
        self,
        canonical_id: str,
        equivalence_fingerprint: str,
        aliases: Tuple[str, ...] = (),
        note: str = "",
    ) -> StateCanonicalDefinition:
        candidate = StateCanonicalDefinition(
            canonical_id=canonical_id,
            equivalence_fingerprint=equivalence_fingerprint,
            note=note,
        )

        old = self.definitions.get(canonical_id)
        if old is not None:
            if old != candidate:
                raise StateIdentityConflict(
                    "Canonical state identity sudah ada dengan semantics "
                    f"berbeda: {canonical_id}"
                )
            definition = old
        else:
            definition = candidate
            self.definitions[canonical_id] = definition

        references = (canonical_id,) + tuple(aliases)
        if len(set(references)) != len(references):
            raise StateIdentityConflict(
                "Alias state duplikat dalam satu registration"
            )

        for reference in references:
            current = self.alias_to_canonical.get(reference)
            if current is not None and current != canonical_id:
                raise StateIdentityConflict(
                    f"State alias '{reference}' sudah terikat ke "
                    f"'{current}', tidak boleh dipindah ke '{canonical_id}'"
                )

        for reference in references:
            self.alias_to_canonical[reference] = canonical_id

        return definition

    def add_alias(
        self,
        canonical_id: str,
        alias: str,
    ) -> StateCanonicalDefinition:
        definition = self.definitions.get(canonical_id)
        if definition is None:
            raise KeyError(
                f"Canonical state '{canonical_id}' belum diregistrasi"
            )
        current = self.alias_to_canonical.get(alias)
        if current is not None and current != canonical_id:
            raise StateIdentityConflict(
                f"State alias '{alias}' sudah terikat ke '{current}'"
            )
        self.alias_to_canonical[alias] = canonical_id
        return definition

    def resolve(
        self,
        state_reference: str,
    ) -> ResolvedStateIdentity:
        canonical_id = self.alias_to_canonical.get(
            state_reference
        )
        if canonical_id is None:
            return ResolvedStateIdentity(
                reference=state_reference,
                canonical_id=state_reference,
                registered=False,
            )
        definition = self.definitions[canonical_id]
        return ResolvedStateIdentity(
            reference=state_reference,
            canonical_id=canonical_id,
            registered=True,
            equivalence_fingerprint=(
                definition.equivalence_fingerprint
            ),
        )

    def aliases_for(
        self,
        canonical_id: str,
    ) -> Tuple[str, ...]:
        if canonical_id not in self.definitions:
            return ()
        return tuple(sorted(
            reference
            for reference, resolved
            in self.alias_to_canonical.items()
            if resolved == canonical_id
            and reference != canonical_id
        ))

    def state(
        self,
        canonical_id: Optional[str] = None,
    ) -> Dict:
        ids = (
            [canonical_id]
            if canonical_id is not None
            else sorted(self.definitions)
        )
        definitions = {}
        for cid in ids:
            definition = self.definitions.get(cid)
            if definition is None:
                continue
            definitions[cid] = {
                "canonical_id": definition.canonical_id,
                "equivalence_fingerprint": (
                    definition.equivalence_fingerprint
                ),
                "note": definition.note,
                "aliases": self.aliases_for(cid),
            }
        return {
            "canonical_states": definitions,
            "canonical_count": len(definitions),
            "alias_count": sum(
                len(item["aliases"])
                for item in definitions.values()
            ),
        }

    @staticmethod
    def canonical_mapping(
        mapping: Dict,
        include_fields: Optional[Tuple[str, ...]] = None,
        exclude_fields: Tuple[str, ...] = (),
    ) -> str:
        """
        Deterministic structured-state helper.

        The adapter explicitly chooses included/excluded fields. No field is
        automatically ignored by the core.
        """
        if not isinstance(mapping, dict):
            raise TypeError("mapping state harus dict")

        if include_fields is None:
            selected = {
                key: value
                for key, value in mapping.items()
                if key not in set(exclude_fields)
            }
        else:
            missing = set(include_fields) - set(mapping)
            if missing:
                raise KeyError(
                    "Structured state kehilangan field: "
                    f"{sorted(missing)}"
                )
            overlap = set(include_fields) & set(exclude_fields)
            if overlap:
                raise ValueError(
                    "Field tidak boleh sekaligus include dan exclude: "
                    f"{sorted(overlap)}"
                )
            selected = {
                key: mapping[key]
                for key in include_fields
            }

        def normalize(value):
            if isinstance(value, dict):
                return {
                    str(key): normalize(inner)
                    for key, inner
                    in sorted(
                        value.items(),
                        key=lambda item: str(item[0]),
                    )
                }
            if isinstance(value, (set, frozenset)):
                normalized = [
                    normalize(inner)
                    for inner in value
                ]
                return sorted(
                    normalized,
                    key=lambda item: json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            if isinstance(value, tuple):
                return [normalize(inner) for inner in value]
            if isinstance(value, list):
                return [normalize(inner) for inner in value]
            if value is None or isinstance(
                value,
                (str, int, float, bool),
            ):
                return value
            raise TypeError(
                "Structured state hanya mendukung JSON-like values; "
                f"ditemukan {type(value).__name__}"
            )

        normalized = normalize(selected)
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class ActionDefinition:
    """Immutable implementation identity for one logical action family."""
    action_id: str
    implementation_fingerprint: str
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    action_version: Optional[int] = None
    note: str = ""

    def __post_init__(self):
        if not self.action_id:
            raise ValueError("action_id tidak boleh kosong")
        if not self.implementation_fingerprint:
            raise ValueError(
                "implementation_fingerprint tidak boleh kosong"
            )
        if (
            self.action_version is not None
            and self.action_version < 1
        ):
            raise ValueError("action_version harus >= 1")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError(
                "valid_until action harus lebih besar dari valid_from"
            )

    @property
    def instance_id(self) -> str:
        version = (
            self.action_version
            if self.action_version is not None
            else "unregistered"
        )
        return f"{self.action_id}@v{version}"

    def applies_at(
        self,
        as_of: Optional[int],
    ) -> Tuple[bool, str]:
        if as_of is None:
            return True, "unscoped_query"
        if (
            self.valid_from is not None
            and as_of < self.valid_from
        ):
            return False, (
                f"query t={as_of} sebelum action berlaku "
                f"t={self.valid_from}"
            )
        if (
            self.valid_until is not None
            and as_of >= self.valid_until
        ):
            return False, (
                f"query t={as_of} setelah action berakhir "
                f"t={self.valid_until}"
            )
        return True, "in_scope"


class ActionVersionConflict(ValueError):
    pass


class ActionRegistry:
    """
    Version-aware action/tool registry.

    Unregistered strings remain legacy-compatible.
    Registered families resolve to immutable instance IDs.
    """

    def __init__(self):
        self.latest: Dict[str, ActionDefinition] = {}
        self.versions: Dict[
            str,
            List[ActionDefinition],
        ] = {}

    @staticmethod
    def split_reference(
        action_reference: str,
    ) -> Tuple[str, Optional[int]]:
        if "@v" not in action_reference:
            return action_reference, None
        family, marker, raw_version = (
            action_reference.rpartition("@v")
        )
        if (
            not marker
            or not family
            or not raw_version.isdigit()
        ):
            return action_reference, None
        return family, int(raw_version)

    @staticmethod
    def _intervals_overlap(
        left: ActionDefinition,
        right: ActionDefinition,
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
        return max(left_start, right_start) < min(
            left_end,
            right_end,
        )

    @staticmethod
    def _same_registration(
        left: ActionDefinition,
        right: ActionDefinition,
    ) -> bool:
        return (
            left.action_id == right.action_id
            and left.implementation_fingerprint
                == right.implementation_fingerprint
            and left.valid_from == right.valid_from
            and left.valid_until == right.valid_until
            and left.note == right.note
        )

    def is_registered(
        self,
        action_reference: str,
    ) -> bool:
        family, version = self.split_reference(
            action_reference
        )
        if family not in self.versions:
            return False
        if version is None:
            return True
        return (
            self.get_version(
                family,
                version,
            )
            is not None
        )

    def all_versions(
        self,
        action_reference: str,
    ) -> List[ActionDefinition]:
        family, _ = self.split_reference(
            action_reference
        )
        return list(
            self.versions.get(
                family,
                [],
            )
        )

    def get_version(
        self,
        action_id: str,
        action_version: int,
    ) -> Optional[ActionDefinition]:
        for item in self.versions.get(
            action_id,
            [],
        ):
            if (
                item.action_version
                == action_version
            ):
                return item
        return None

    def register(
        self,
        definition: ActionDefinition,
    ) -> ActionDefinition:
        existing = self.versions.setdefault(
            definition.action_id,
            [],
        )

        for old in existing:
            if (
                self._same_registration(
                    old,
                    definition,
                )
                and (
                    definition.action_version is None
                    or old.action_version
                        == definition.action_version
                )
            ):
                return old

        version = (
            definition.action_version
            if definition.action_version
            is not None
            else (
                max(
                    (
                        item.action_version or 0
                        for item in existing
                    ),
                    default=0,
                )
                + 1
            )
        )
        candidate = replace(
            definition,
            action_version=version,
        )

        for old in existing:
            if (
                old.action_version
                == candidate.action_version
            ):
                if old == candidate:
                    return old
                raise ActionVersionConflict(
                    "Action version collision: "
                    f"{candidate.instance_id}"
                )

            if self._intervals_overlap(
                old,
                candidate,
            ):
                raise ActionVersionConflict(
                    "Action versions overlap: "
                    f"{old.instance_id} vs "
                    f"{candidate.instance_id}. "
                    "Gunakan supersede_action() atau valid_until."
                )

        existing.append(candidate)
        existing.sort(
            key=lambda item: (
                item.action_version or 0
            )
        )
        self.latest[
            candidate.action_id
        ] = existing[-1]
        return candidate

    def close_version(
        self,
        action_id: str,
        action_version: int,
        valid_until: int,
    ) -> ActionDefinition:
        items = self.versions.get(
            action_id,
            [],
        )
        for index, old in enumerate(items):
            if old.action_version != action_version:
                continue
            if (
                old.valid_from is not None
                and valid_until <= old.valid_from
            ):
                raise ValueError(
                    "valid_until harus setelah valid_from action"
                )
            if (
                old.valid_until is not None
                and valid_until > old.valid_until
            ):
                raise ValueError(
                    "close_version tidak boleh memperpanjang action lama"
                )

            closed = replace(
                old,
                valid_until=valid_until,
            )
            items[index] = closed
            if (
                self.latest.get(action_id) is old
                or (
                    self.latest.get(action_id)
                    is not None
                    and self.latest[
                        action_id
                    ].action_version
                        == action_version
                )
            ):
                self.latest[action_id] = max(
                    items,
                    key=lambda item: (
                        item.action_version or 0
                    ),
                )
            return closed

        raise KeyError(
            f"Action {action_id}@v{action_version} tidak ditemukan"
        )

    def resolve(
        self,
        action_reference: str,
        as_of: Optional[int] = None,
        require_active: bool = False,
    ) -> Optional[ActionDefinition]:
        family, explicit = self.split_reference(
            action_reference
        )
        if family not in self.versions:
            return None

        if explicit is not None:
            definition = self.get_version(
                family,
                explicit,
            )
            if definition is None:
                raise KeyError(
                    f"Action {action_reference} tidak ditemukan"
                )
            if require_active:
                active, reason = definition.applies_at(
                    as_of
                )
                if not active:
                    raise ActionVersionConflict(
                        f"Action {definition.instance_id} "
                        f"tidak aktif: {reason}"
                    )
            return definition

        active = [
            definition
            for definition
            in self.versions[family]
            if definition.applies_at(
                as_of
            )[0]
        ]
        if len(active) == 1:
            return active[0]
        if len(active) > 1:
            raise ActionVersionConflict(
                f"Ambiguous active action family '{family}'"
            )
        if require_active:
            raise ActionVersionConflict(
                f"Tidak ada active action version untuk '{family}'"
            )
        return self.latest.get(family)

    def supersede(
        self,
        action_id: str,
        implementation_fingerprint: str,
        observed_at: int,
        note: str = "",
    ) -> Dict:
        active = [
            definition
            for definition
            in self.versions.get(
                action_id,
                [],
            )
            if definition.applies_at(
                observed_at
            )[0]
        ]
        if len(active) != 1:
            raise ActionVersionConflict(
                "supersede_action memerlukan tepat satu "
                f"active version; ditemukan {len(active)} "
                f"untuk {action_id}"
            )

        previous = active[0]
        closed = self.close_version(
            action_id,
            previous.action_version,
            observed_at,
        )
        successor = self.register(
            ActionDefinition(
                action_id=action_id,
                implementation_fingerprint=(
                    implementation_fingerprint
                ),
                valid_from=observed_at,
                note=note,
            )
        )
        return {
            "operation": "supersede_action",
            "action_id": action_id,
            "observed_at": observed_at,
            "previous": closed,
            "successor": successor,
            "previous_instance_id":
                closed.instance_id,
            "successor_instance_id":
                successor.instance_id,
        }

    def state(
        self,
        action_id: Optional[str] = None,
    ) -> Dict:
        refs = (
            [action_id]
            if action_id is not None
            else sorted(self.versions)
        )
        families = {}
        for ref in refs:
            family, _ = self.split_reference(
                ref
            )
            items = self.versions.get(
                family,
                [],
            )
            if not items:
                continue
            families[family] = [
                {
                    "action_id":
                        item.action_id,
                    "action_version":
                        item.action_version,
                    "instance_id":
                        item.instance_id,
                    "implementation_fingerprint":
                        item.implementation_fingerprint,
                    "valid_from":
                        item.valid_from,
                    "valid_until":
                        item.valid_until,
                    "note": item.note,
                }
                for item in items
            ]

        return {
            "families": families,
            "logical_action_count":
                len(families),
            "version_count": sum(
                len(items)
                for items in families.values()
            ),
        }


@dataclass(frozen=True)
class ResolvedActionIdentity:
    reference: str
    family: str
    instance_id: str
    registered: bool
    action_version: Optional[int] = None
    implementation_fingerprint: Optional[str] = None


# Preference-profile identity is physically implemented in objectives.py.
from .objectives import (
    ObjectiveUtilityProfile,
    ObjectiveProfileVersionConflict,
    ObjectiveProfileRegistry,
)

# -------------------------------------------------------------------------
# Trusted-local checkpoint compatibility.
# -------------------------------------------------------------------------
_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"

_PICKLE_COMPAT_CLASSES = (
    StateCanonicalDefinition,
    ResolvedStateIdentity,
    StateIdentityConflict,
    StateIdentityRegistry,
    ActionDefinition,
    ActionVersionConflict,
    ActionRegistry,
    ResolvedActionIdentity,
)

for _cls in _PICKLE_COMPAT_CLASSES:
    _cls.__module__ = _CANONICAL_PICKLE_MODULE

del _cls

__all__ = [
    "StateCanonicalDefinition",
    "ResolvedStateIdentity",
    "StateIdentityConflict",
    "StateIdentityRegistry",
    "ActionDefinition",
    "ActionVersionConflict",
    "ActionRegistry",
    "ResolvedActionIdentity",
    "ObjectiveUtilityProfile",
    "ObjectiveProfileVersionConflict",
    "ObjectiveProfileRegistry",
]
