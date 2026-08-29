"""Spatial transformation algebra — V2.34.

CPU-first, standard-library-only exact orthogonal affine transforms over the
V2.33 axis-aligned 2D scene model.

Supported linear transforms are the eight signed-permutation matrices of D4:
- identity;
- rotations 90/180/270 degrees;
- reflections across X, Y, y=x, y=-x.

Each linear transform may also carry a 2D translation, allowing explicit
source-frame -> target-frame maps.

V2.34 deliberately does NOT implement arbitrary-angle rotation, scale, shear,
physics, manipulation actions, or automatic decision-learning coupling.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .spatial import (
    SpatialError,
    SpatialExtent2D,
    SpatialObject2D,
    SpatialPose2D,
    SpatialScene2D,
)


DEFAULT_TRANSFORM_MATCH_TOLERANCE = 1e-9
MAX_TRANSFORM_MATCH_OBJECTS = 256


class SpatialTransformError(SpatialError):
    pass


class SpatialTransformFrameError(SpatialTransformError):
    pass


class SpatialTransformMatchError(SpatialTransformError):
    pass


class SpatialLinearTransformKind(Enum):
    IDENTITY = "identity"
    ROTATE_90_CCW = "rotate_90_ccw"
    ROTATE_180 = "rotate_180"
    ROTATE_270_CCW = "rotate_270_ccw"
    REFLECT_X = "reflect_x"
    REFLECT_Y = "reflect_y"
    REFLECT_Y_EQ_X = "reflect_y_eq_x"
    REFLECT_Y_EQ_NEG_X = "reflect_y_eq_neg_x"


_KIND_TO_MATRIX = {
    SpatialLinearTransformKind.IDENTITY: (1, 0, 0, 1),
    SpatialLinearTransformKind.ROTATE_90_CCW: (0, -1, 1, 0),
    SpatialLinearTransformKind.ROTATE_180: (-1, 0, 0, -1),
    SpatialLinearTransformKind.ROTATE_270_CCW: (0, 1, -1, 0),
    SpatialLinearTransformKind.REFLECT_X: (1, 0, 0, -1),
    SpatialLinearTransformKind.REFLECT_Y: (-1, 0, 0, 1),
    SpatialLinearTransformKind.REFLECT_Y_EQ_X: (0, 1, 1, 0),
    SpatialLinearTransformKind.REFLECT_Y_EQ_NEG_X: (0, -1, -1, 0),
}
_MATRIX_TO_KIND = {
    matrix: kind
    for kind, matrix in _KIND_TO_MATRIX.items()
}


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SpatialTransformError(f"{name} harus finite")
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


def _near(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


@dataclass(frozen=True)
class SpatialTransform2D:
    """Signed-orthogonal affine transform.

    Mapping convention:

        target_point = A @ source_point + t

    where A is a 2x2 signed-permutation matrix.
    """

    a: int
    b: int
    c: int
    d: int
    tx: float
    ty: float
    source_frame_id: str
    target_frame_id: str

    def __post_init__(self):
        matrix = (
            int(self.a),
            int(self.b),
            int(self.c),
            int(self.d),
        )
        if matrix not in _MATRIX_TO_KIND:
            raise SpatialTransformError(
                "V2.34 hanya menerima signed-orthogonal D4 matrix; "
                f"dapat {matrix}"
            )
        for name, value in (
            ("source_frame_id", self.source_frame_id),
            ("target_frame_id", self.target_frame_id),
        ):
            if not isinstance(value, str) or not value:
                raise SpatialTransformError(
                    f"{name} tidak boleh kosong"
                )

        object.__setattr__(self, "a", matrix[0])
        object.__setattr__(self, "b", matrix[1])
        object.__setattr__(self, "c", matrix[2])
        object.__setattr__(self, "d", matrix[3])
        object.__setattr__(self, "tx", _finite(self.tx, "tx"))
        object.__setattr__(self, "ty", _finite(self.ty, "ty"))

    @property
    def matrix(self) -> Tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)

    @property
    def linear_kind(self) -> SpatialLinearTransformKind:
        return _MATRIX_TO_KIND[self.matrix]

    @property
    def determinant(self) -> int:
        return self.a * self.d - self.b * self.c

    @property
    def swaps_axes(self) -> bool:
        return bool(self.b or self.c)

    @property
    def has_translation(self) -> bool:
        return self.tx != 0.0 or self.ty != 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.linear_kind == SpatialLinearTransformKind.IDENTITY
            and not self.has_translation
            and self.source_frame_id == self.target_frame_id
        )

    @property
    def semantic_signature(self) -> str:
        return _signature(
            "spatial_transform",
            {
                "matrix": list(self.matrix),
                "tx": self.tx,
                "ty": self.ty,
                "source_frame_id": self.source_frame_id,
                "target_frame_id": self.target_frame_id,
            },
        )

    def to_descriptor(self) -> Dict:
        return {
            "schema": "agen-spatial-transform-2d-v1",
            "matrix": [
                self.a,
                self.b,
                self.c,
                self.d,
            ],
            "translation": [
                self.tx,
                self.ty,
            ],
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "linear_kind": self.linear_kind.value,
        }

    @classmethod
    def from_descriptor(
        cls,
        descriptor: Dict,
    ) -> "SpatialTransform2D":
        if not isinstance(descriptor, dict):
            raise SpatialTransformError(
                "transform descriptor harus dict"
            )
        if descriptor.get("schema") != "agen-spatial-transform-2d-v1":
            raise SpatialTransformError(
                "transform descriptor schema tidak didukung"
            )

        matrix = descriptor.get("matrix")
        translation = descriptor.get("translation")
        if (
            not isinstance(matrix, (list, tuple))
            or len(matrix) != 4
        ):
            raise SpatialTransformError(
                "descriptor matrix harus 4 elemen"
            )
        if (
            not isinstance(translation, (list, tuple))
            or len(translation) != 2
        ):
            raise SpatialTransformError(
                "descriptor translation harus 2 elemen"
            )

        result = cls(
            a=matrix[0],
            b=matrix[1],
            c=matrix[2],
            d=matrix[3],
            tx=translation[0],
            ty=translation[1],
            source_frame_id=descriptor.get(
                "source_frame_id"
            ),
            target_frame_id=descriptor.get(
                "target_frame_id"
            ),
        )

        declared_kind = descriptor.get("linear_kind")
        if (
            declared_kind is not None
            and declared_kind != result.linear_kind.value
        ):
            raise SpatialTransformError(
                "descriptor linear_kind tidak cocok dengan matrix"
            )
        return result

    @classmethod
    def from_kind(
        cls,
        kind: SpatialLinearTransformKind,
        *,
        source_frame_id: str,
        target_frame_id: Optional[str] = None,
        tx: float = 0.0,
        ty: float = 0.0,
    ) -> "SpatialTransform2D":
        if not isinstance(kind, SpatialLinearTransformKind):
            raise SpatialTransformError(
                "kind harus SpatialLinearTransformKind"
            )
        if target_frame_id is None:
            target_frame_id = source_frame_id
        a, b, c, d = _KIND_TO_MATRIX[kind]
        return cls(
            a=a,
            b=b,
            c=c,
            d=d,
            tx=tx,
            ty=ty,
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
        )

    @classmethod
    def identity(
        cls,
        frame_id: str,
    ) -> "SpatialTransform2D":
        return cls.from_kind(
            SpatialLinearTransformKind.IDENTITY,
            source_frame_id=frame_id,
            target_frame_id=frame_id,
        )

    @classmethod
    def translation(
        cls,
        dx: float,
        dy: float,
        *,
        source_frame_id: str,
        target_frame_id: Optional[str] = None,
    ) -> "SpatialTransform2D":
        return cls.from_kind(
            SpatialLinearTransformKind.IDENTITY,
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
            tx=dx,
            ty=dy,
        )

    def apply_xy(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        x = _finite(x, "x")
        y = _finite(y, "y")
        return (
            _finite(
                self.a * x + self.b * y + self.tx,
                "transformed_x",
            ),
            _finite(
                self.c * x + self.d * y + self.ty,
                "transformed_y",
            ),
        )

    def apply_pose(
        self,
        pose: SpatialPose2D,
    ) -> SpatialPose2D:
        if not isinstance(pose, SpatialPose2D):
            raise SpatialTransformError(
                "pose harus SpatialPose2D"
            )
        x, y = self.apply_xy(pose.x, pose.y)
        return SpatialPose2D(x, y)

    def apply_extent(
        self,
        extent: SpatialExtent2D,
    ) -> SpatialExtent2D:
        if not isinstance(extent, SpatialExtent2D):
            raise SpatialTransformError(
                "extent harus SpatialExtent2D"
            )
        # Absolute signed-permutation matrix maps an axis-aligned rectangle to
        # another axis-aligned rectangle exactly.
        width = (
            abs(self.a) * extent.width
            + abs(self.b) * extent.height
        )
        height = (
            abs(self.c) * extent.width
            + abs(self.d) * extent.height
        )
        return SpatialExtent2D(width, height)

    def apply_object(
        self,
        object_: SpatialObject2D,
    ) -> SpatialObject2D:
        if not isinstance(object_, SpatialObject2D):
            raise SpatialTransformError(
                "object harus SpatialObject2D"
            )
        return SpatialObject2D(
            object_id=object_.object_id,
            pose=self.apply_pose(object_.pose),
            extent=self.apply_extent(object_.extent),
            labels=object_.labels,
        )

    def apply_scene(
        self,
        scene: SpatialScene2D,
        *,
        scene_id: Optional[str] = None,
        observed_at: Optional[int] = None,
    ) -> SpatialScene2D:
        if not isinstance(scene, SpatialScene2D):
            raise SpatialTransformError(
                "scene harus SpatialScene2D"
            )
        if scene.frame_id != self.source_frame_id:
            raise SpatialTransformFrameError(
                "Source scene frame tidak cocok dengan transform: "
                f"{scene.frame_id} != {self.source_frame_id}"
            )
        if scene_id is None:
            scene_id = "scene-" + uuid.uuid4().hex
        return SpatialScene2D(
            scene_id=scene_id,
            namespace=scene.namespace,
            belief_context_id=scene.belief_context_id,
            frame_id=self.target_frame_id,
            objects=tuple(
                self.apply_object(item)
                for item in scene.objects
            ),
            observed_at=(
                scene.observed_at
                if observed_at is None
                else int(observed_at)
            ),
        )

    def inverse(self) -> "SpatialTransform2D":
        # Signed-orthogonal inverse is transpose.
        ia, ib, ic, id_ = (
            self.a,
            self.c,
            self.b,
            self.d,
        )
        itx = -(ia * self.tx + ib * self.ty)
        ity = -(ic * self.tx + id_ * self.ty)
        return SpatialTransform2D(
            a=ia,
            b=ib,
            c=ic,
            d=id_,
            tx=itx,
            ty=ity,
            source_frame_id=self.target_frame_id,
            target_frame_id=self.source_frame_id,
        )

    def then(
        self,
        after: "SpatialTransform2D",
    ) -> "SpatialTransform2D":
        """Compose as `after(self(point))`.

        Frame provenance must line up exactly.
        """
        if not isinstance(after, SpatialTransform2D):
            raise SpatialTransformError(
                "after harus SpatialTransform2D"
            )
        if self.target_frame_id != after.source_frame_id:
            raise SpatialTransformFrameError(
                "Transform composition frame mismatch: "
                f"{self.target_frame_id} != {after.source_frame_id}"
            )

        a = after.a * self.a + after.b * self.c
        b = after.a * self.b + after.b * self.d
        c = after.c * self.a + after.d * self.c
        d = after.c * self.b + after.d * self.d

        tx = (
            after.a * self.tx
            + after.b * self.ty
            + after.tx
        )
        ty = (
            after.c * self.tx
            + after.d * self.ty
            + after.ty
        )

        return SpatialTransform2D(
            a=a,
            b=b,
            c=c,
            d=d,
            tx=tx,
            ty=ty,
            source_frame_id=self.source_frame_id,
            target_frame_id=after.target_frame_id,
        )


@dataclass(frozen=True)
class SpatialTransformMatch:
    transform: SpatialTransform2D
    source_scene_id: str
    target_scene_id: str
    object_count: int
    max_position_error: float
    max_extent_error: float
    tolerance: float

    def __post_init__(self):
        if not isinstance(self.transform, SpatialTransform2D):
            raise SpatialTransformMatchError(
                "transform harus SpatialTransform2D"
            )
        if self.object_count <= 0:
            raise SpatialTransformMatchError(
                "object_count harus positif"
            )
        object.__setattr__(
            self,
            "max_position_error",
            _finite(
                self.max_position_error,
                "max_position_error",
            ),
        )
        object.__setattr__(
            self,
            "max_extent_error",
            _finite(
                self.max_extent_error,
                "max_extent_error",
            ),
        )
        tolerance = _finite(self.tolerance, "tolerance")
        if tolerance < 0.0:
            raise SpatialTransformMatchError(
                "tolerance tidak boleh negatif"
            )
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False


@dataclass(frozen=True)
class SpatialTransformInference:
    source_scene_id: str
    target_scene_id: str
    candidates: Tuple[SpatialTransformMatch, ...]
    tolerance: float
    object_identity_required: bool = True
    labels_required: bool = True

    @property
    def matched(self) -> bool:
        return bool(self.candidates)

    @property
    def unique(self) -> bool:
        return len(self.candidates) == 1

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    @property
    def unique_transform(self) -> Optional[SpatialTransform2D]:
        if not self.unique:
            return None
        return self.candidates[0].transform

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_truth(self) -> bool:
        return False


class SpatialTransformationMatcher:
    """Infer D4 + translation scene transforms from exact object identity.

    Matching is numerical within an explicit tolerance; it is not fuzzy
    perception. Object identity sets must match exactly.
    """

    CANDIDATE_KINDS = tuple(SpatialLinearTransformKind)

    @staticmethod
    def _validate_scene_pair(
        source: SpatialScene2D,
        target: SpatialScene2D,
        *,
        require_namespace: bool,
        require_belief_context: bool,
    ):
        if not isinstance(source, SpatialScene2D):
            raise SpatialTransformMatchError(
                "source harus SpatialScene2D"
            )
        if not isinstance(target, SpatialScene2D):
            raise SpatialTransformMatchError(
                "target harus SpatialScene2D"
            )
        if len(source.objects) > MAX_TRANSFORM_MATCH_OBJECTS:
            raise SpatialTransformMatchError(
                "source scene melebihi batas transform matching"
            )
        if len(target.objects) > MAX_TRANSFORM_MATCH_OBJECTS:
            raise SpatialTransformMatchError(
                "target scene melebihi batas transform matching"
            )
        if require_namespace and source.namespace != target.namespace:
            raise SpatialTransformMatchError(
                "namespace source/target berbeda"
            )
        if (
            require_belief_context
            and source.belief_context_id != target.belief_context_id
        ):
            raise SpatialTransformMatchError(
                "Belief Context source/target berbeda"
            )

        source_ids = {
            item.object_id
            for item in source.objects
        }
        target_ids = {
            item.object_id
            for item in target.objects
        }
        if source_ids != target_ids:
            raise SpatialTransformMatchError(
                "V2.34 transform inference membutuhkan object_id set yang sama"
            )

    @classmethod
    def infer(
        cls,
        source: SpatialScene2D,
        target: SpatialScene2D,
        *,
        tolerance: float = DEFAULT_TRANSFORM_MATCH_TOLERANCE,
        require_labels: bool = True,
        require_namespace: bool = True,
        require_belief_context: bool = True,
    ) -> SpatialTransformInference:
        tolerance = _finite(tolerance, "tolerance")
        if tolerance < 0.0:
            raise SpatialTransformMatchError(
                "tolerance tidak boleh negatif"
            )

        cls._validate_scene_pair(
            source,
            target,
            require_namespace=require_namespace,
            require_belief_context=require_belief_context,
        )

        source_map = source.object_map()
        target_map = target.object_map()
        object_ids = tuple(sorted(source_map))
        anchor_id = object_ids[0]
        anchor_source = source_map[anchor_id]
        anchor_target = target_map[anchor_id]

        matches = []

        for kind in cls.CANDIDATE_KINDS:
            a, b, c, d = _KIND_TO_MATRIX[kind]

            anchor_linear_x = (
                a * anchor_source.pose.x
                + b * anchor_source.pose.y
            )
            anchor_linear_y = (
                c * anchor_source.pose.x
                + d * anchor_source.pose.y
            )
            tx = anchor_target.pose.x - anchor_linear_x
            ty = anchor_target.pose.y - anchor_linear_y

            transform = SpatialTransform2D(
                a=a,
                b=b,
                c=c,
                d=d,
                tx=tx,
                ty=ty,
                source_frame_id=source.frame_id,
                target_frame_id=target.frame_id,
            )

            max_position_error = 0.0
            max_extent_error = 0.0
            valid = True

            for object_id in object_ids:
                source_object = source_map[object_id]
                target_object = target_map[object_id]

                if (
                    require_labels
                    and source_object.labels != target_object.labels
                ):
                    valid = False
                    break

                predicted = transform.apply_object(source_object)

                position_error = max(
                    abs(predicted.pose.x - target_object.pose.x),
                    abs(predicted.pose.y - target_object.pose.y),
                )
                extent_error = max(
                    abs(
                        predicted.extent.width
                        - target_object.extent.width
                    ),
                    abs(
                        predicted.extent.height
                        - target_object.extent.height
                    ),
                )
                max_position_error = max(
                    max_position_error,
                    position_error,
                )
                max_extent_error = max(
                    max_extent_error,
                    extent_error,
                )

                if (
                    position_error > tolerance
                    or extent_error > tolerance
                ):
                    valid = False
                    break

            if valid:
                matches.append(
                    SpatialTransformMatch(
                        transform=transform,
                        source_scene_id=source.scene_id,
                        target_scene_id=target.scene_id,
                        object_count=len(object_ids),
                        max_position_error=max_position_error,
                        max_extent_error=max_extent_error,
                        tolerance=tolerance,
                    )
                )

        matches.sort(
            key=lambda item: (
                item.transform.linear_kind.value,
                item.transform.tx,
                item.transform.ty,
                item.transform.semantic_signature,
            )
        )

        return SpatialTransformInference(
            source_scene_id=source.scene_id,
            target_scene_id=target.scene_id,
            candidates=tuple(matches),
            tolerance=tolerance,
            object_identity_required=True,
            labels_required=bool(require_labels),
        )


def spatial_transform_token(
    transform: SpatialTransform2D,
    *,
    include_translation: bool = False,
    include_frames: bool = False,
) -> Tuple:
    """Read-only symbolic bridge to V2.32 structural patterns.

    Default token captures the transformation CLASS only, so repeated
    ROTATE_90 transforms at different absolute translations can be treated as
    the same structural operator by an explicit pattern adapter.
    """
    if not isinstance(transform, SpatialTransform2D):
        raise SpatialTransformError(
            "transform harus SpatialTransform2D"
        )

    token = [
        "spatial_transform",
        transform.linear_kind.value,
    ]

    if include_translation:
        token.extend([
            transform.tx,
            transform.ty,
        ])

    if include_frames:
        token.extend([
            transform.source_frame_id,
            transform.target_frame_id,
        ])

    return tuple(token)


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialTransformError,
    SpatialTransformFrameError,
    SpatialTransformMatchError,
    SpatialLinearTransformKind,
    SpatialTransform2D,
    SpatialTransformMatch,
    SpatialTransformInference,
    SpatialTransformationMatcher,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "DEFAULT_TRANSFORM_MATCH_TOLERANCE",
    "MAX_TRANSFORM_MATCH_OBJECTS",
    "SpatialTransformError",
    "SpatialTransformFrameError",
    "SpatialTransformMatchError",
    "SpatialLinearTransformKind",
    "SpatialTransform2D",
    "SpatialTransformMatch",
    "SpatialTransformInference",
    "SpatialTransformationMatcher",
    "spatial_transform_token",
]
