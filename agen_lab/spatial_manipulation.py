"""Counterfactual spatial manipulation — V2.35.

Standard-library-only, CPU-first model operators over the V2.33/V2.34
object-centric spatial representation.

Supported operators:
- MOVE: translate one object in its current scene frame;
- ROTATE: rotate one object's axis-aligned extent by quarter turns;
- PLACE_INSIDE: place one object inside a target container;
- STACK_ABOVE: place one object centered above a support, touching or with gap.

Every result is counterfactual/model state. No physical execution, Q/world
learning, Evidence update, or manipulation planning is performed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from .spatial import (
    SpatialError,
    SpatialGeometry2D,
    SpatialObject2D,
    SpatialPose2D,
    SpatialRelationType,
    SpatialScene2D,
)
from .spatial_transform import (
    SpatialLinearTransformKind,
    SpatialTransform2D,
)


MAX_MANIPULATION_SCENE_OBJECTS = 256


class SpatialManipulationError(SpatialError):
    pass


class SpatialManipulationKind(Enum):
    MOVE = "move"
    ROTATE = "rotate"
    PLACE_INSIDE = "place_inside"
    STACK_ABOVE = "stack_above"


class SpatialManipulationCheckKind(Enum):
    SUBJECT_EXISTS = "subject_exists"
    TARGET_EXISTS = "target_exists"
    DISTINCT_SUBJECT_TARGET = "distinct_subject_target"
    NONZERO_MOVE = "nonzero_move"
    NONZERO_ROTATION = "nonzero_rotation"
    REPRESENTABLE_ROTATION_EFFECT = "representable_rotation_effect"
    FITS_INSIDE_TARGET = "fits_inside_target"
    INSIDE_RELATION_ACHIEVED = "inside_relation_achieved"
    STACK_RELATION_ACHIEVED = "stack_relation_achieved"
    NO_FORBIDDEN_COLLISION = "no_forbidden_collision"


_FORBIDDEN_OCCUPANCY_RELATIONS = frozenset({
    SpatialRelationType.OVERLAPS,
    SpatialRelationType.COINCIDENT,
})


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SpatialManipulationError(
            f"{name} harus finite"
        )
    if result == 0.0:
        return 0.0
    return result


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _signature(prefix: str, payload) -> str:
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return f"{prefix}:sha256:{digest}"


@dataclass(frozen=True)
class SpatialManipulationOperator:
    kind: SpatialManipulationKind
    subject_id: str
    target_id: Optional[str] = None
    dx: float = 0.0
    dy: float = 0.0
    quarter_turns: int = 0
    offset_x: float = 0.0
    offset_y: float = 0.0
    gap: float = 0.0

    def __post_init__(self):
        if not isinstance(
            self.kind,
            SpatialManipulationKind,
        ):
            raise SpatialManipulationError(
                "kind harus SpatialManipulationKind"
            )
        if not isinstance(self.subject_id, str) or not self.subject_id:
            raise SpatialManipulationError(
                "subject_id tidak boleh kosong"
            )
        if self.target_id is not None and (
            not isinstance(self.target_id, str)
            or not self.target_id
        ):
            raise SpatialManipulationError(
                "target_id harus string non-empty atau None"
            )

        object.__setattr__(
            self,
            "dx",
            _finite(self.dx, "dx"),
        )
        object.__setattr__(
            self,
            "dy",
            _finite(self.dy, "dy"),
        )
        object.__setattr__(
            self,
            "offset_x",
            _finite(self.offset_x, "offset_x"),
        )
        object.__setattr__(
            self,
            "offset_y",
            _finite(self.offset_y, "offset_y"),
        )
        gap = _finite(self.gap, "gap")
        if gap < 0.0:
            raise SpatialManipulationError(
                "gap tidak boleh negatif"
            )
        object.__setattr__(self, "gap", gap)

        turns = int(self.quarter_turns)
        if turns != self.quarter_turns:
            raise SpatialManipulationError(
                "quarter_turns harus integer"
            )
        turns %= 4
        object.__setattr__(
            self,
            "quarter_turns",
            turns,
        )

        if self.kind in (
            SpatialManipulationKind.PLACE_INSIDE,
            SpatialManipulationKind.STACK_ABOVE,
        ):
            if self.target_id is None:
                raise SpatialManipulationError(
                    f"{self.kind.value} membutuhkan target_id"
                )
        elif self.target_id is not None:
            raise SpatialManipulationError(
                f"{self.kind.value} tidak menerima target_id"
            )

        if self.kind != SpatialManipulationKind.MOVE and (
            self.dx != 0.0 or self.dy != 0.0
        ):
            raise SpatialManipulationError(
                f"{self.kind.value} tidak menerima dx/dy"
            )

        if self.kind != SpatialManipulationKind.ROTATE and (
            self.quarter_turns != 0
        ):
            raise SpatialManipulationError(
                f"{self.kind.value} tidak menerima quarter_turns"
            )

        if self.kind != SpatialManipulationKind.PLACE_INSIDE and (
            self.offset_x != 0.0 or self.offset_y != 0.0
        ):
            raise SpatialManipulationError(
                f"{self.kind.value} tidak menerima offset"
            )

        if self.kind != SpatialManipulationKind.STACK_ABOVE and (
            self.gap != 0.0
        ):
            raise SpatialManipulationError(
                f"{self.kind.value} tidak menerima gap"
            )

    @classmethod
    def move(
        cls,
        subject_id: str,
        dx: float,
        dy: float,
    ) -> "SpatialManipulationOperator":
        return cls(
            kind=SpatialManipulationKind.MOVE,
            subject_id=subject_id,
            dx=dx,
            dy=dy,
        )

    @classmethod
    def rotate(
        cls,
        subject_id: str,
        quarter_turns: int,
    ) -> "SpatialManipulationOperator":
        return cls(
            kind=SpatialManipulationKind.ROTATE,
            subject_id=subject_id,
            quarter_turns=quarter_turns,
        )

    @classmethod
    def place_inside(
        cls,
        subject_id: str,
        target_id: str,
        *,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> "SpatialManipulationOperator":
        return cls(
            kind=SpatialManipulationKind.PLACE_INSIDE,
            subject_id=subject_id,
            target_id=target_id,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    @classmethod
    def stack_above(
        cls,
        subject_id: str,
        target_id: str,
        *,
        gap: float = 0.0,
    ) -> "SpatialManipulationOperator":
        return cls(
            kind=SpatialManipulationKind.STACK_ABOVE,
            subject_id=subject_id,
            target_id=target_id,
            gap=gap,
        )

    @property
    def semantic_signature(self) -> str:
        return _signature(
            "spatial_manipulation",
            self.to_descriptor(),
        )

    @property
    def is_action_experience(self) -> bool:
        return False

    def to_descriptor(self) -> Dict:
        return {
            "schema": "agen-spatial-manipulation-v1",
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "target_id": self.target_id,
            "dx": self.dx,
            "dy": self.dy,
            "quarter_turns": self.quarter_turns,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "gap": self.gap,
        }

    @classmethod
    def from_descriptor(
        cls,
        descriptor: Dict,
    ) -> "SpatialManipulationOperator":
        if not isinstance(descriptor, dict):
            raise SpatialManipulationError(
                "manipulation descriptor harus dict"
            )
        if descriptor.get("schema") != "agen-spatial-manipulation-v1":
            raise SpatialManipulationError(
                "manipulation descriptor schema tidak didukung"
            )
        try:
            kind = SpatialManipulationKind(
                descriptor.get("kind")
            )
        except Exception as exc:
            raise SpatialManipulationError(
                "manipulation kind tidak valid"
            ) from exc

        return cls(
            kind=kind,
            subject_id=descriptor.get("subject_id"),
            target_id=descriptor.get("target_id"),
            dx=descriptor.get("dx", 0.0),
            dy=descriptor.get("dy", 0.0),
            quarter_turns=descriptor.get(
                "quarter_turns",
                0,
            ),
            offset_x=descriptor.get("offset_x", 0.0),
            offset_y=descriptor.get("offset_y", 0.0),
            gap=descriptor.get("gap", 0.0),
        )


@dataclass(frozen=True)
class SpatialManipulationCheck:
    kind: SpatialManipulationCheckKind
    passed: bool
    detail: str

    def __post_init__(self):
        if not isinstance(
            self.kind,
            SpatialManipulationCheckKind,
        ):
            raise SpatialManipulationError(
                "check kind tidak valid"
            )
        if not isinstance(self.passed, bool):
            raise SpatialManipulationError(
                "check passed harus bool"
            )
        if not isinstance(self.detail, str):
            raise SpatialManipulationError(
                "check detail harus string"
            )


@dataclass(frozen=True)
class SpatialManipulationCollision:
    other_object_id: str
    relation_types: Tuple[SpatialRelationType, ...]

    def __post_init__(self):
        if (
            not isinstance(self.other_object_id, str)
            or not self.other_object_id
        ):
            raise SpatialManipulationError(
                "other_object_id tidak boleh kosong"
            )
        relation_types = tuple(self.relation_types)
        for item in relation_types:
            if not isinstance(item, SpatialRelationType):
                raise SpatialManipulationError(
                    "collision relation_types tidak valid"
                )
        object.__setattr__(
            self,
            "relation_types",
            tuple(
                sorted(
                    relation_types,
                    key=lambda item: item.value,
                )
            ),
        )


@dataclass(frozen=True)
class CounterfactualSpatialManipulation:
    simulation_id: str
    source_scene_id: str
    operator: SpatialManipulationOperator
    preconditions: Tuple[SpatialManipulationCheck, ...]
    feasible: bool
    predicted_scene: Optional[SpatialScene2D]
    subject_before: Optional[SpatialObject2D]
    subject_after: Optional[SpatialObject2D]
    predicted_subject_target_relations: Tuple[
        SpatialRelationType,
        ...
    ]
    collisions: Tuple[SpatialManipulationCollision, ...]

    def __post_init__(self):
        if not isinstance(self.simulation_id, str) or not self.simulation_id:
            raise SpatialManipulationError(
                "simulation_id tidak boleh kosong"
            )
        if not isinstance(self.source_scene_id, str) or not self.source_scene_id:
            raise SpatialManipulationError(
                "source_scene_id tidak boleh kosong"
            )
        if not isinstance(
            self.operator,
            SpatialManipulationOperator,
        ):
            raise SpatialManipulationError(
                "operator tidak valid"
            )
        if not isinstance(self.feasible, bool):
            raise SpatialManipulationError(
                "feasible harus bool"
            )

        if self.feasible:
            if self.predicted_scene is None or self.subject_after is None:
                raise SpatialManipulationError(
                    "feasible simulation membutuhkan predicted scene/effect"
                )
            if not all(
                check.passed
                for check in self.preconditions
            ):
                raise SpatialManipulationError(
                    "feasible simulation memiliki failed precondition"
                )
        else:
            if self.predicted_scene is not None:
                raise SpatialManipulationError(
                    "failed simulation tidak boleh punya predicted_scene"
                )

        object.__setattr__(
            self,
            "preconditions",
            tuple(self.preconditions),
        )
        object.__setattr__(
            self,
            "predicted_subject_target_relations",
            tuple(
                sorted(
                    self.predicted_subject_target_relations,
                    key=lambda item: item.value,
                )
            ),
        )
        object.__setattr__(
            self,
            "collisions",
            tuple(self.collisions),
        )

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False

    @property
    def was_executed(self) -> bool:
        return False


class SpatialManipulationSimulator:

    @staticmethod
    def _replace_subject(
        scene: SpatialScene2D,
        subject_after: SpatialObject2D,
        *,
        predicted_scene_id: str,
    ) -> SpatialScene2D:
        return SpatialScene2D(
            scene_id=predicted_scene_id,
            namespace=scene.namespace,
            belief_context_id=scene.belief_context_id,
            frame_id=scene.frame_id,
            objects=tuple(
                (
                    subject_after
                    if item.object_id
                    == subject_after.object_id
                    else item
                )
                for item in scene.objects
            ),
            observed_at=scene.observed_at,
        )

    @staticmethod
    def _relation_types(
        subject: SpatialObject2D,
        target: SpatialObject2D,
    ) -> Tuple[SpatialRelationType, ...]:
        return SpatialGeometry2D.direct_relation_types(
            subject,
            target,
        )

    @classmethod
    def _collisions(
        cls,
        scene: SpatialScene2D,
        subject_after: SpatialObject2D,
        *,
        exempt_object_ids: Sequence[str] = (),
    ) -> Tuple[SpatialManipulationCollision, ...]:
        exempt = set(exempt_object_ids)
        exempt.add(subject_after.object_id)
        conflicts = []

        for other in scene.objects:
            if other.object_id in exempt:
                continue
            relation_types = cls._relation_types(
                subject_after,
                other,
            )
            forbidden = tuple(
                item
                for item in relation_types
                if item in _FORBIDDEN_OCCUPANCY_RELATIONS
            )
            if forbidden:
                conflicts.append(
                    SpatialManipulationCollision(
                        other_object_id=other.object_id,
                        relation_types=forbidden,
                    )
                )

        return tuple(
            sorted(
                conflicts,
                key=lambda item: item.other_object_id,
            )
        )

    @classmethod
    def simulate(
        cls,
        scene: SpatialScene2D,
        operator: SpatialManipulationOperator,
        *,
        predicted_scene_id: Optional[str] = None,
    ) -> CounterfactualSpatialManipulation:
        if not isinstance(scene, SpatialScene2D):
            raise SpatialManipulationError(
                "scene harus SpatialScene2D"
            )
        if not isinstance(
            operator,
            SpatialManipulationOperator,
        ):
            raise SpatialManipulationError(
                "operator harus SpatialManipulationOperator"
            )
        if len(scene.objects) > MAX_MANIPULATION_SCENE_OBJECTS:
            raise SpatialManipulationError(
                "scene melebihi batas manipulation simulator"
            )

        if predicted_scene_id is None:
            predicted_scene_id = "cf-scene-" + uuid.uuid4().hex
        if (
            not isinstance(predicted_scene_id, str)
            or not predicted_scene_id
        ):
            raise SpatialManipulationError(
                "predicted_scene_id tidak boleh kosong"
            )

        object_map = scene.object_map()
        checks = []

        subject = object_map.get(operator.subject_id)
        checks.append(
            SpatialManipulationCheck(
                kind=SpatialManipulationCheckKind.SUBJECT_EXISTS,
                passed=subject is not None,
                detail=(
                    f"subject {operator.subject_id} ditemukan"
                    if subject is not None
                    else f"subject {operator.subject_id} tidak ditemukan"
                ),
            )
        )

        target = None
        if operator.target_id is not None:
            target = object_map.get(operator.target_id)
            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.TARGET_EXISTS,
                    passed=target is not None,
                    detail=(
                        f"target {operator.target_id} ditemukan"
                        if target is not None
                        else f"target {operator.target_id} tidak ditemukan"
                    ),
                )
            )
            distinct = (
                operator.subject_id
                != operator.target_id
            )
            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.DISTINCT_SUBJECT_TARGET,
                    passed=distinct,
                    detail=(
                        "subject dan target berbeda"
                        if distinct
                        else "subject dan target tidak boleh sama"
                    ),
                )
            )

        if subject is None or (
            operator.target_id is not None
            and target is None
        ) or any(not item.passed for item in checks):
            return CounterfactualSpatialManipulation(
                simulation_id="cf-manip-" + uuid.uuid4().hex,
                source_scene_id=scene.scene_id,
                operator=operator,
                preconditions=tuple(checks),
                feasible=False,
                predicted_scene=None,
                subject_before=subject,
                subject_after=None,
                predicted_subject_target_relations=(),
                collisions=(),
            )

        subject_after = subject
        target_relations = ()
        exempt = ()

        if operator.kind == SpatialManipulationKind.MOVE:
            nonzero = (
                operator.dx != 0.0
                or operator.dy != 0.0
            )
            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.NONZERO_MOVE,
                    passed=nonzero,
                    detail=(
                        "move displacement nonzero"
                        if nonzero
                        else "zero-displacement MOVE ditolak"
                    ),
                )
            )
            if nonzero:
                subject_after = SpatialObject2D(
                    object_id=subject.object_id,
                    pose=SpatialPose2D(
                        subject.pose.x + operator.dx,
                        subject.pose.y + operator.dy,
                    ),
                    extent=subject.extent,
                    labels=subject.labels,
                )

        elif operator.kind == SpatialManipulationKind.ROTATE:
            nonzero = operator.quarter_turns != 0
            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.NONZERO_ROTATION,
                    passed=nonzero,
                    detail=(
                        f"rotation quarter_turns={operator.quarter_turns}"
                        if nonzero
                        else "zero-turn ROTATE ditolak"
                    ),
                )
            )
            representable = (
                operator.quarter_turns in (1, 3)
                and subject.extent.width
                != subject.extent.height
            )
            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.REPRESENTABLE_ROTATION_EFFECT,
                    passed=representable,
                    detail=(
                        "quarter-turn changes axis-aligned extent"
                        if representable
                        else (
                            "rotation effect tidak representable tanpa "
                            "orientation state V2.35"
                        )
                    ),
                )
            )
            if nonzero and representable:
                kind = {
                    1: SpatialLinearTransformKind.ROTATE_90_CCW,
                    3: SpatialLinearTransformKind.ROTATE_270_CCW,
                }[operator.quarter_turns]
                transform = SpatialTransform2D.from_kind(
                    kind,
                    source_frame_id=scene.frame_id,
                )
                subject_after = SpatialObject2D(
                    object_id=subject.object_id,
                    pose=subject.pose,
                    extent=transform.apply_extent(
                        subject.extent
                    ),
                    labels=subject.labels,
                )

        elif operator.kind == SpatialManipulationKind.PLACE_INSIDE:
            assert target is not None
            candidate = SpatialObject2D(
                object_id=subject.object_id,
                pose=SpatialPose2D(
                    target.pose.x + operator.offset_x,
                    target.pose.y + operator.offset_y,
                ),
                extent=subject.extent,
                labels=subject.labels,
            )
            relation_types = cls._relation_types(
                candidate,
                target,
            )
            fits = (
                SpatialRelationType.INSIDE
                in relation_types
            )
            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.FITS_INSIDE_TARGET,
                    passed=fits,
                    detail=(
                        "subject geometry fits inside target"
                        if fits
                        else "subject geometry tidak fit inside target"
                    ),
                )
            )
            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.INSIDE_RELATION_ACHIEVED,
                    passed=fits,
                    detail=(
                        "predicted relation includes INSIDE"
                        if fits
                        else "predicted relation tidak menghasilkan INSIDE"
                    ),
                )
            )
            if fits:
                subject_after = candidate
                target_relations = relation_types
                exempt = (target.object_id,)

        elif operator.kind == SpatialManipulationKind.STACK_ABOVE:
            assert target is not None
            y = (
                target.bounds.max_y
                + operator.gap
                + subject.extent.height / 2.0
            )
            candidate = SpatialObject2D(
                object_id=subject.object_id,
                pose=SpatialPose2D(
                    target.pose.x,
                    y,
                ),
                extent=subject.extent,
                labels=subject.labels,
            )
            relation_types = cls._relation_types(
                candidate,
                target,
            )
            if operator.gap == 0.0:
                achieved = (
                    SpatialRelationType.TOUCHING
                    in relation_types
                    and candidate.pose.y > target.pose.y
                )
            else:
                achieved = (
                    SpatialRelationType.ABOVE
                    in relation_types
                    and SpatialRelationType.DISJOINT
                    in relation_types
                )

            checks.append(
                SpatialManipulationCheck(
                    kind=SpatialManipulationCheckKind.STACK_RELATION_ACHIEVED,
                    passed=achieved,
                    detail=(
                        "predicted stack geometry achieved"
                        if achieved
                        else "predicted stack geometry tidak tercapai"
                    ),
                )
            )
            if achieved:
                subject_after = candidate
                target_relations = relation_types
                exempt = (target.object_id,)

        else:
            raise SpatialManipulationError(
                f"Unsupported manipulation kind: {operator.kind}"
            )

        if any(not item.passed for item in checks):
            return CounterfactualSpatialManipulation(
                simulation_id="cf-manip-" + uuid.uuid4().hex,
                source_scene_id=scene.scene_id,
                operator=operator,
                preconditions=tuple(checks),
                feasible=False,
                predicted_scene=None,
                subject_before=subject,
                subject_after=None,
                predicted_subject_target_relations=(),
                collisions=(),
            )

        collisions = cls._collisions(
            scene,
            subject_after,
            exempt_object_ids=exempt,
        )
        collision_free = not collisions
        checks.append(
            SpatialManipulationCheck(
                kind=SpatialManipulationCheckKind.NO_FORBIDDEN_COLLISION,
                passed=collision_free,
                detail=(
                    "no forbidden occupancy collision"
                    if collision_free
                    else (
                        "forbidden occupancy collision dengan "
                        + ",".join(
                            item.other_object_id
                            for item in collisions
                        )
                    )
                ),
            )
        )

        if not collision_free:
            return CounterfactualSpatialManipulation(
                simulation_id="cf-manip-" + uuid.uuid4().hex,
                source_scene_id=scene.scene_id,
                operator=operator,
                preconditions=tuple(checks),
                feasible=False,
                predicted_scene=None,
                subject_before=subject,
                subject_after=subject_after,
                predicted_subject_target_relations=target_relations,
                collisions=collisions,
            )

        predicted_scene = cls._replace_subject(
            scene,
            subject_after,
            predicted_scene_id=predicted_scene_id,
        )

        return CounterfactualSpatialManipulation(
            simulation_id="cf-manip-" + uuid.uuid4().hex,
            source_scene_id=scene.scene_id,
            operator=operator,
            preconditions=tuple(checks),
            feasible=True,
            predicted_scene=predicted_scene,
            subject_before=subject,
            subject_after=subject_after,
            predicted_subject_target_relations=target_relations,
            collisions=collisions,
        )


def spatial_manipulation_token(
    operator: SpatialManipulationOperator,
    *,
    include_object_ids: bool = False,
    include_numeric_parameters: bool = False,
) -> Tuple:
    """Read-only symbolic bridge for explicit pattern adapters.

    Default token represents manipulation class only.
    """
    if not isinstance(
        operator,
        SpatialManipulationOperator,
    ):
        raise SpatialManipulationError(
            "operator harus SpatialManipulationOperator"
        )

    token = [
        "spatial_manipulation",
        operator.kind.value,
    ]

    if include_object_ids:
        token.extend([
            operator.subject_id,
            operator.target_id,
        ])

    if include_numeric_parameters:
        token.extend([
            operator.dx,
            operator.dy,
            operator.quarter_turns,
            operator.offset_x,
            operator.offset_y,
            operator.gap,
        ])

    return tuple(token)


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialManipulationError,
    SpatialManipulationKind,
    SpatialManipulationCheckKind,
    SpatialManipulationOperator,
    SpatialManipulationCheck,
    SpatialManipulationCollision,
    CounterfactualSpatialManipulation,
    SpatialManipulationSimulator,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "MAX_MANIPULATION_SCENE_OBJECTS",
    "SpatialManipulationError",
    "SpatialManipulationKind",
    "SpatialManipulationCheckKind",
    "SpatialManipulationOperator",
    "SpatialManipulationCheck",
    "SpatialManipulationCollision",
    "CounterfactualSpatialManipulation",
    "SpatialManipulationSimulator",
    "spatial_manipulation_token",
]
