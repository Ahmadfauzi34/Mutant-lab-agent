"""Object-centric 2D spatial state and relation algebra — V2.33.

Standard-library-only, CPU-first spatial representation.

V2.33 deliberately implements:
- object identity inside one scene;
- one explicit 2D coordinate frame per scene;
- axis-aligned object geometry;
- deterministic geometric relations;
- safe inverse/symmetric/transitive relation algebra;
- translation-normalized scene signatures;
- bounded operational scene memory.

V2.33 deliberately does NOT implement:
- rotation / orientation semantics;
- coordinate-frame transforms;
- motion, collision dynamics, or manipulation;
- path planning coupling;
- fuzzy/learned spatial perception;
- automatic transfer to Q/world-model/pattern learning.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MAX_SPATIAL_OBJECTS_PER_SCENE = 256
DEFAULT_SPATIAL_SCENE_LIMIT = 512
MAX_SPATIAL_RELATIONS = 65536


class SpatialError(ValueError):
    pass


class SpatialSceneConflict(SpatialError):
    pass


class SpatialRelationConflict(SpatialError):
    pass


class SpatialRelationType(Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    CONTAINS = "contains"
    TOUCHING = "touching"
    OVERLAPS = "overlaps"
    DISJOINT = "disjoint"
    COINCIDENT = "coincident"


class SpatialRelationSource(Enum):
    GEOMETRIC_DIRECT = "geometric_direct"
    EXPLICIT_INPUT = "explicit_input"
    ALGEBRA_DERIVED = "algebra_derived"


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SpatialError(f"{name} harus finite")
    # Collapse signed zero so canonical signatures remain stable.
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
class SpatialPose2D:
    x: float
    y: float

    def __post_init__(self):
        object.__setattr__(self, "x", _finite(self.x, "x"))
        object.__setattr__(self, "y", _finite(self.y, "y"))


@dataclass(frozen=True)
class SpatialExtent2D:
    width: float
    height: float

    def __post_init__(self):
        width = _finite(self.width, "width")
        height = _finite(self.height, "height")
        if width < 0.0 or height < 0.0:
            raise SpatialError("width/height tidak boleh negatif")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)


@dataclass(frozen=True)
class SpatialBounds2D:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self):
        min_x = _finite(self.min_x, "min_x")
        min_y = _finite(self.min_y, "min_y")
        max_x = _finite(self.max_x, "max_x")
        max_y = _finite(self.max_y, "max_y")
        if min_x > max_x or min_y > max_y:
            raise SpatialError("SpatialBounds2D min tidak boleh > max")
        object.__setattr__(self, "min_x", min_x)
        object.__setattr__(self, "min_y", min_y)
        object.__setattr__(self, "max_x", max_x)
        object.__setattr__(self, "max_y", max_y)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass(frozen=True)
class SpatialObject2D:
    object_id: str
    pose: SpatialPose2D
    extent: SpatialExtent2D
    labels: Tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.object_id, str) or not self.object_id:
            raise SpatialError("object_id tidak boleh kosong")
        if not isinstance(self.pose, SpatialPose2D):
            raise SpatialError("pose harus SpatialPose2D")
        if not isinstance(self.extent, SpatialExtent2D):
            raise SpatialError("extent harus SpatialExtent2D")

        labels = tuple(self.labels)
        for label in labels:
            if not isinstance(label, str) or not label:
                raise SpatialError("labels harus string non-empty")
        object.__setattr__(self, "labels", labels)

    @property
    def bounds(self) -> SpatialBounds2D:
        half_w = self.extent.width / 2.0
        half_h = self.extent.height / 2.0
        return SpatialBounds2D(
            self.pose.x - half_w,
            self.pose.y - half_h,
            self.pose.x + half_w,
            self.pose.y + half_h,
        )


@dataclass(frozen=True)
class SpatialScene2D:
    scene_id: str
    namespace: str
    belief_context_id: str
    frame_id: str
    objects: Tuple[SpatialObject2D, ...]
    observed_at: int

    def __post_init__(self):
        for name, value in (
            ("scene_id", self.scene_id),
            ("namespace", self.namespace),
            ("belief_context_id", self.belief_context_id),
            ("frame_id", self.frame_id),
        ):
            if not isinstance(value, str) or not value:
                raise SpatialError(f"{name} tidak boleh kosong")

        objects = tuple(self.objects)
        if not objects:
            raise SpatialError("Spatial scene harus memiliki minimal satu object")
        if len(objects) > MAX_SPATIAL_OBJECTS_PER_SCENE:
            raise SpatialError(
                "Spatial scene melewati batas object "
                f"{MAX_SPATIAL_OBJECTS_PER_SCENE}"
            )

        seen = set()
        for item in objects:
            if not isinstance(item, SpatialObject2D):
                raise SpatialError("scene objects harus SpatialObject2D")
            if item.object_id in seen:
                raise SpatialError(
                    f"Duplicate spatial object_id: {item.object_id}"
                )
            seen.add(item.object_id)

        # Stable object order is part of scene identity, independent of adapter
        # input ordering.
        object.__setattr__(
            self,
            "objects",
            tuple(sorted(objects, key=lambda item: item.object_id)),
        )
        object.__setattr__(self, "observed_at", int(self.observed_at))

    def object_map(self) -> Dict[str, SpatialObject2D]:
        return {
            item.object_id: item
            for item in self.objects
        }


@dataclass(frozen=True)
class SpatialRelation:
    relation_id: str
    relation_type: SpatialRelationType
    subject_id: str
    object_id: str
    source: SpatialRelationSource
    scene_id: Optional[str] = None
    premise_relation_ids: Tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.relation_id, str) or not self.relation_id:
            raise SpatialError("relation_id tidak boleh kosong")
        if not isinstance(self.relation_type, SpatialRelationType):
            raise SpatialError("relation_type tidak valid")
        for name, value in (
            ("subject_id", self.subject_id),
            ("object_id", self.object_id),
        ):
            if not isinstance(value, str) or not value:
                raise SpatialError(f"{name} tidak boleh kosong")
        if self.subject_id == self.object_id:
            raise SpatialRelationConflict(
                "Spatial relation strict/self tidak diizinkan"
            )
        if not isinstance(self.source, SpatialRelationSource):
            raise SpatialError("relation source tidak valid")
        if self.scene_id is not None and (
            not isinstance(self.scene_id, str)
            or not self.scene_id
        ):
            raise SpatialError("scene_id relation harus string/None")
        object.__setattr__(
            self,
            "premise_relation_ids",
            tuple(self.premise_relation_ids),
        )

    @property
    def key(self) -> Tuple[str, SpatialRelationType, str]:
        return (
            self.subject_id,
            self.relation_type,
            self.object_id,
        )

    @property
    def is_observation(self) -> bool:
        return False

    @property
    def is_evidence(self) -> bool:
        return False


class SpatialRelationAlgebra:
    INVERSE = {
        SpatialRelationType.LEFT_OF: SpatialRelationType.RIGHT_OF,
        SpatialRelationType.RIGHT_OF: SpatialRelationType.LEFT_OF,
        SpatialRelationType.ABOVE: SpatialRelationType.BELOW,
        SpatialRelationType.BELOW: SpatialRelationType.ABOVE,
        SpatialRelationType.INSIDE: SpatialRelationType.CONTAINS,
        SpatialRelationType.CONTAINS: SpatialRelationType.INSIDE,
        SpatialRelationType.TOUCHING: SpatialRelationType.TOUCHING,
        SpatialRelationType.OVERLAPS: SpatialRelationType.OVERLAPS,
        SpatialRelationType.DISJOINT: SpatialRelationType.DISJOINT,
        SpatialRelationType.COINCIDENT: SpatialRelationType.COINCIDENT,
    }

    SYMMETRIC = frozenset({
        SpatialRelationType.TOUCHING,
        SpatialRelationType.OVERLAPS,
        SpatialRelationType.DISJOINT,
        SpatialRelationType.COINCIDENT,
    })

    TRANSITIVE = frozenset({
        SpatialRelationType.LEFT_OF,
        SpatialRelationType.RIGHT_OF,
        SpatialRelationType.ABOVE,
        SpatialRelationType.BELOW,
        SpatialRelationType.INSIDE,
        SpatialRelationType.CONTAINS,
    })

    OPPOSITES = {
        SpatialRelationType.LEFT_OF: SpatialRelationType.RIGHT_OF,
        SpatialRelationType.RIGHT_OF: SpatialRelationType.LEFT_OF,
        SpatialRelationType.ABOVE: SpatialRelationType.BELOW,
        SpatialRelationType.BELOW: SpatialRelationType.ABOVE,
        SpatialRelationType.INSIDE: SpatialRelationType.CONTAINS,
        SpatialRelationType.CONTAINS: SpatialRelationType.INSIDE,
    }

    TOPOLOGY_TYPES = frozenset({
        SpatialRelationType.INSIDE,
        SpatialRelationType.CONTAINS,
        SpatialRelationType.TOUCHING,
        SpatialRelationType.OVERLAPS,
        SpatialRelationType.DISJOINT,
        SpatialRelationType.COINCIDENT,
    })

    @staticmethod
    def relation_id(
        relation_type: SpatialRelationType,
        subject_id: str,
        object_id: str,
        *,
        source: SpatialRelationSource,
        scene_id: Optional[str] = None,
        premise_relation_ids: Sequence[str] = (),
    ) -> str:
        return _signature(
            "srel",
            {
                "type": relation_type.value,
                "subject": subject_id,
                "object": object_id,
                "source": source.value,
                "scene_id": scene_id,
                "premises": sorted(premise_relation_ids),
            },
        )

    @classmethod
    def make(
        cls,
        relation_type: SpatialRelationType,
        subject_id: str,
        object_id: str,
        *,
        source: SpatialRelationSource = SpatialRelationSource.EXPLICIT_INPUT,
        scene_id: Optional[str] = None,
        premise_relation_ids: Sequence[str] = (),
    ) -> SpatialRelation:
        return SpatialRelation(
            relation_id=cls.relation_id(
                relation_type,
                subject_id,
                object_id,
                source=source,
                scene_id=scene_id,
                premise_relation_ids=premise_relation_ids,
            ),
            relation_type=relation_type,
            subject_id=subject_id,
            object_id=object_id,
            source=source,
            scene_id=scene_id,
            premise_relation_ids=tuple(premise_relation_ids),
        )

    @classmethod
    def _validate_pair_conflicts(
        cls,
        relation_keys: Iterable[
            Tuple[str, SpatialRelationType, str]
        ],
    ):
        by_pair: Dict[Tuple[str, str], set] = defaultdict(set)
        for subject, relation_type, object_id in relation_keys:
            if subject == object_id:
                raise SpatialRelationConflict(
                    f"Self spatial relation tidak valid: {relation_type.value}"
                )
            by_pair[(subject, object_id)].add(relation_type)

        for pair, relation_types in by_pair.items():
            for relation_type in tuple(relation_types):
                opposite = cls.OPPOSITES.get(relation_type)
                if opposite is not None and opposite in relation_types:
                    raise SpatialRelationConflict(
                        "Spatial relation contradiction pada "
                        f"{pair}: {relation_type.value} vs {opposite.value}"
                    )

            if SpatialRelationType.DISJOINT in relation_types:
                forbidden = (
                    cls.TOPOLOGY_TYPES
                    - {SpatialRelationType.DISJOINT}
                )
                conflict = relation_types & forbidden
                if conflict:
                    raise SpatialRelationConflict(
                        "DISJOINT bertentangan dengan "
                        f"{sorted(item.value for item in conflict)} pada {pair}"
                    )

            if SpatialRelationType.COINCIDENT in relation_types:
                forbidden = {
                    SpatialRelationType.INSIDE,
                    SpatialRelationType.CONTAINS,
                    SpatialRelationType.TOUCHING,
                    SpatialRelationType.OVERLAPS,
                    SpatialRelationType.DISJOINT,
                    SpatialRelationType.LEFT_OF,
                    SpatialRelationType.RIGHT_OF,
                    SpatialRelationType.ABOVE,
                    SpatialRelationType.BELOW,
                }
                conflict = relation_types & forbidden
                if conflict:
                    raise SpatialRelationConflict(
                        "COINCIDENT bertentangan dengan relasi lain pada "
                        f"{pair}: {sorted(item.value for item in conflict)}"
                    )

    @classmethod
    def close(
        cls,
        relations: Sequence[SpatialRelation],
        *,
        max_relations: int = MAX_SPATIAL_RELATIONS,
    ) -> Tuple[SpatialRelation, ...]:
        """Safe algebraic closure.

        Adds inverse/symmetric forms and exact transitive consequences.
        Cyclic strict relations and direct contradictions are rejected.
        """
        if max_relations <= 0:
            raise SpatialError("max_relations harus positif")

        known: Dict[
            Tuple[str, SpatialRelationType, str],
            SpatialRelation,
        ] = {}

        scene_ids = {
            relation.scene_id
            for relation in relations
            if (
                isinstance(relation, SpatialRelation)
                and relation.scene_id is not None
            )
        }
        if len(scene_ids) > 1:
            raise SpatialRelationConflict(
                "Spatial relation closure tidak boleh mencampur scene/frame "
                f"berbeda: {sorted(scene_ids)}"
            )

        for relation in relations:
            if not isinstance(relation, SpatialRelation):
                raise SpatialError("relations harus SpatialRelation")
            existing = known.get(relation.key)
            if existing is None:
                known[relation.key] = relation

        cls._validate_pair_conflicts(known.keys())

        changed = True
        while changed:
            changed = False
            additions: List[SpatialRelation] = []

            # Inverse / symmetric closure.
            for relation in tuple(known.values()):
                inverse_type = cls.INVERSE[relation.relation_type]
                inverse_key = (
                    relation.object_id,
                    inverse_type,
                    relation.subject_id,
                )
                if inverse_key not in known:
                    additions.append(
                        cls.make(
                            inverse_type,
                            relation.object_id,
                            relation.subject_id,
                            source=SpatialRelationSource.ALGEBRA_DERIVED,
                            scene_id=relation.scene_id,
                            premise_relation_ids=(
                                relation.relation_id,
                            ),
                        )
                    )

            # Transitive closure.
            outgoing: Dict[
                Tuple[str, SpatialRelationType],
                List[SpatialRelation],
            ] = defaultdict(list)
            for relation in known.values():
                if relation.relation_type in cls.TRANSITIVE:
                    outgoing[
                        (
                            relation.subject_id,
                            relation.relation_type,
                        )
                    ].append(relation)

            for first in tuple(known.values()):
                if first.relation_type not in cls.TRANSITIVE:
                    continue
                for second in outgoing.get(
                    (
                        first.object_id,
                        first.relation_type,
                    ),
                    (),
                ):
                    if first.subject_id == second.object_id:
                        raise SpatialRelationConflict(
                            "Cycle pada strict transitive spatial relation: "
                            f"{first.relation_type.value}"
                        )
                    key = (
                        first.subject_id,
                        first.relation_type,
                        second.object_id,
                    )
                    if key in known:
                        continue
                    additions.append(
                        cls.make(
                            first.relation_type,
                            first.subject_id,
                            second.object_id,
                            source=SpatialRelationSource.ALGEBRA_DERIVED,
                            scene_id=(
                                first.scene_id
                                if first.scene_id == second.scene_id
                                else None
                            ),
                            premise_relation_ids=(
                                first.relation_id,
                                second.relation_id,
                            ),
                        )
                    )

            for relation in additions:
                if relation.key in known:
                    continue
                known[relation.key] = relation
                changed = True
                if len(known) > max_relations:
                    raise SpatialError(
                        "Spatial relation closure melewati batas "
                        f"{max_relations}"
                    )

            cls._validate_pair_conflicts(known.keys())

        return tuple(
            sorted(
                known.values(),
                key=lambda item: (
                    item.subject_id,
                    item.relation_type.value,
                    item.object_id,
                    item.source.value,
                ),
            )
        )


class SpatialGeometry2D:
    @staticmethod
    def _contains(
        outer: SpatialBounds2D,
        inner: SpatialBounds2D,
    ) -> bool:
        return (
            outer.min_x <= inner.min_x
            and outer.min_y <= inner.min_y
            and outer.max_x >= inner.max_x
            and outer.max_y >= inner.max_y
        )

    @staticmethod
    def _same(
        left: SpatialBounds2D,
        right: SpatialBounds2D,
    ) -> bool:
        return (
            left.min_x == right.min_x
            and left.min_y == right.min_y
            and left.max_x == right.max_x
            and left.max_y == right.max_y
        )

    @classmethod
    def direct_relation_types(
        cls,
        subject: SpatialObject2D,
        object_: SpatialObject2D,
    ) -> Tuple[SpatialRelationType, ...]:
        if subject.object_id == object_.object_id:
            raise SpatialError("Relation query memerlukan dua object berbeda")

        a = subject.bounds
        b = object_.bounds
        result = set()

        if a.max_x < b.min_x:
            result.add(SpatialRelationType.LEFT_OF)
        if a.min_x > b.max_x:
            result.add(SpatialRelationType.RIGHT_OF)
        if a.min_y > b.max_y:
            result.add(SpatialRelationType.ABOVE)
        if a.max_y < b.min_y:
            result.add(SpatialRelationType.BELOW)

        same = cls._same(a, b)
        a_inside_b = cls._contains(b, a) and not same
        a_contains_b = cls._contains(a, b) and not same

        if same:
            result.add(SpatialRelationType.COINCIDENT)
        elif a_inside_b:
            result.add(SpatialRelationType.INSIDE)
        elif a_contains_b:
            result.add(SpatialRelationType.CONTAINS)
        else:
            overlap_x = min(a.max_x, b.max_x) - max(a.min_x, b.min_x)
            overlap_y = min(a.max_y, b.max_y) - max(a.min_y, b.min_y)

            if overlap_x < 0.0 or overlap_y < 0.0:
                result.add(SpatialRelationType.DISJOINT)
            elif overlap_x == 0.0 or overlap_y == 0.0:
                result.add(SpatialRelationType.TOUCHING)
            else:
                result.add(SpatialRelationType.OVERLAPS)

        return tuple(sorted(result, key=lambda item: item.value))

    @classmethod
    def scene_relations(
        cls,
        scene: SpatialScene2D,
    ) -> Tuple[SpatialRelation, ...]:
        objects = scene.objects
        relation_count_upper = len(objects) * max(0, len(objects) - 1)
        if relation_count_upper > MAX_SPATIAL_RELATIONS:
            raise SpatialError("Scene relation upper-bound terlalu besar")

        relations = []
        for subject in objects:
            for object_ in objects:
                if subject.object_id == object_.object_id:
                    continue
                for relation_type in cls.direct_relation_types(
                    subject,
                    object_,
                ):
                    relations.append(
                        SpatialRelationAlgebra.make(
                            relation_type,
                            subject.object_id,
                            object_.object_id,
                            source=SpatialRelationSource.GEOMETRIC_DIRECT,
                            scene_id=scene.scene_id,
                        )
                    )

        # Geometry generates both directions itself; closure here is used as a
        # validator and to ensure exact transitive consequences remain
        # consistent if direct relations ever become adapter-filtered later.
        return SpatialRelationAlgebra.close(relations)


class SpatialSceneCanonicalizer:
    @staticmethod
    def exact_signature(scene: SpatialScene2D) -> str:
        return _signature(
            "scene_exact",
            {
                "namespace": scene.namespace,
                "belief_context_id": scene.belief_context_id,
                "frame_id": scene.frame_id,
                "objects": [
                    {
                        "id": item.object_id,
                        "x": item.pose.x,
                        "y": item.pose.y,
                        "width": item.extent.width,
                        "height": item.extent.height,
                        "labels": list(item.labels),
                    }
                    for item in scene.objects
                ],
            },
        )

    @staticmethod
    def translation_normalized_signature(scene: SpatialScene2D) -> str:
        min_x = min(item.bounds.min_x for item in scene.objects)
        min_y = min(item.bounds.min_y for item in scene.objects)

        return _signature(
            "scene_translation_normalized",
            {
                "namespace": scene.namespace,
                "belief_context_id": scene.belief_context_id,
                "frame_id": scene.frame_id,
                "objects": [
                    {
                        "id": item.object_id,
                        "x": _finite(item.pose.x - min_x, "normalized_x"),
                        "y": _finite(item.pose.y - min_y, "normalized_y"),
                        "width": item.extent.width,
                        "height": item.extent.height,
                        "labels": list(item.labels),
                    }
                    for item in scene.objects
                ],
            },
        )

    @staticmethod
    def relational_signature(
        scene: SpatialScene2D,
        relations: Optional[Sequence[SpatialRelation]] = None,
    ) -> str:
        if relations is None:
            relations = SpatialGeometry2D.scene_relations(scene)

        return _signature(
            "scene_relational",
            {
                "namespace": scene.namespace,
                "belief_context_id": scene.belief_context_id,
                "frame_id": scene.frame_id,
                "relations": sorted(
                    (
                        relation.subject_id,
                        relation.relation_type.value,
                        relation.object_id,
                    )
                    for relation in relations
                ),
            },
        )


class SpatialSceneStore:
    """Bounded operational spatial scene registry.

    Scene storage is current/operational in V2.33. No exact COLD spatial-scene
    archive or temporal object tracking is claimed.
    """

    def __init__(
        self,
        scene_limit: int = DEFAULT_SPATIAL_SCENE_LIMIT,
    ):
        if int(scene_limit) <= 0:
            raise SpatialError("scene_limit harus positif")
        self.scene_limit = int(scene_limit)
        self.scenes: OrderedDict[str, SpatialScene2D] = OrderedDict()
        self.total_scenes_registered = 0

    def register(
        self,
        scene: SpatialScene2D,
    ) -> Dict:
        if not isinstance(scene, SpatialScene2D):
            raise SpatialError("scene harus SpatialScene2D")

        existing = self.scenes.get(scene.scene_id)
        if existing is not None:
            if (
                SpatialSceneCanonicalizer.exact_signature(existing)
                != SpatialSceneCanonicalizer.exact_signature(scene)
            ):
                raise SpatialSceneConflict(
                    f"scene_id dipakai untuk spatial content berbeda: "
                    f"{scene.scene_id}"
                )
            self.scenes.move_to_end(scene.scene_id)
            return {
                "deduplicated": True,
                "scene": existing,
            }

        self.scenes[scene.scene_id] = scene
        self.scenes.move_to_end(scene.scene_id)
        self.total_scenes_registered += 1

        while len(self.scenes) > self.scene_limit:
            self.scenes.popitem(last=False)

        return {
            "deduplicated": False,
            "scene": scene,
        }

    def get(self, scene_id: str) -> SpatialScene2D:
        try:
            scene = self.scenes[scene_id]
        except KeyError as exc:
            raise SpatialError(
                f"Spatial scene tidak ditemukan: {scene_id}"
            ) from exc
        self.scenes.move_to_end(scene_id)
        return scene

    def state(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        current = [
            scene
            for scene in self.scenes.values()
            if (
                (namespace is None or scene.namespace == namespace)
                and (
                    belief_context_id is None
                    or scene.belief_context_id == belief_context_id
                )
            )
        ]
        return {
            "operational_scenes": len(current),
            "total_scenes_registered": self.total_scenes_registered,
            "scene_limit": self.scene_limit,
            "max_objects_per_scene": MAX_SPATIAL_OBJECTS_PER_SCENE,
            "coordinate_model": "single_frame_axis_aligned_2d",
            "temporal_object_tracking": False,
            "cross_frame_transform": True,
            "cross_frame_transform_model": "V2.34_D4_plus_translation",
            "counterfactual_manipulation": True,
            "counterfactual_manipulation_model": (
                "V2.35_MOVE_ROTATE_PLACE_INSIDE_STACK_ABOVE"
            ),
            "manipulation_semantics": False,
            "physical_manipulation_execution": False,
            "bounded_spatial_manipulation_planning": True,
            "spatial_planning_model": "V2.36_BOUNDED_BFS_RELATION_GOALS",
            "ticketed_spatial_execution_feedback": True,
            "spatial_execution_feedback_model": (
                "V2.37_EXTERNAL_DISPATCH_ACTUAL_OBSERVATION"
            ),
            "deviation_triggered_spatial_replanning": True,
            "spatial_replanning_model": (
                "V2.38_EXPLICIT_DEVIATION_BOUNDED_BFS"
            ),
            "deterministic_spatial_recovery_policy": True,
            "spatial_recovery_policy_model": (
                "V2.39_CONTINUE_REPLAN_ABORT_INTERVENTION_HANDOFF"
            ),
            "controlled_replacement_handoff": True,
            "empirical_manipulation_reliability": True,
            "spatial_reliability_model": (
                "V2.40_ACTUAL_FEEDBACK_BETA_WILSON"
            ),
            "reliability_gated_recovery_handoff": True,
            "reliability_aware_spatial_plan_ranking": True,
            "spatial_plan_ranking_model": (
                "V2.41_EQUAL_DEPTH_FULL_COVERAGE_WILSON"
            ),
            "reliability_aware_replacement_plan_ranking": True,
            "spatial_replan_ranking_model": (
                "DERIVED_REPLAN_VIEW_FULL_COVERAGE_WILSON"
            ),
            "reliability_aware_spatial_replanning": False,
            "automatic_reliability_gate": False,
            "automatic_feedback_recovery_side_effect": False,
            "autonomous_spatial_replanning": False,
        }


def make_spatial_scene(
    objects: Sequence[SpatialObject2D],
    *,
    namespace: str,
    belief_context_id: str,
    frame_id: str = "world",
    scene_id: Optional[str] = None,
    observed_at: int = 0,
) -> SpatialScene2D:
    if scene_id is None:
        scene_id = "scene-" + uuid.uuid4().hex
    return SpatialScene2D(
        scene_id=scene_id,
        namespace=namespace,
        belief_context_id=belief_context_id,
        frame_id=frame_id,
        objects=tuple(objects),
        observed_at=int(observed_at),
    )


_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    SpatialError,
    SpatialSceneConflict,
    SpatialRelationConflict,
    SpatialRelationType,
    SpatialRelationSource,
    SpatialPose2D,
    SpatialExtent2D,
    SpatialBounds2D,
    SpatialObject2D,
    SpatialScene2D,
    SpatialRelation,
    SpatialRelationAlgebra,
    SpatialGeometry2D,
    SpatialSceneCanonicalizer,
    SpatialSceneStore,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "MAX_SPATIAL_OBJECTS_PER_SCENE",
    "DEFAULT_SPATIAL_SCENE_LIMIT",
    "MAX_SPATIAL_RELATIONS",
    "SpatialError",
    "SpatialSceneConflict",
    "SpatialRelationConflict",
    "SpatialRelationType",
    "SpatialRelationSource",
    "SpatialPose2D",
    "SpatialExtent2D",
    "SpatialBounds2D",
    "SpatialObject2D",
    "SpatialScene2D",
    "SpatialRelation",
    "SpatialRelationAlgebra",
    "SpatialGeometry2D",
    "SpatialSceneCanonicalizer",
    "SpatialSceneStore",
    "make_spatial_scene",
]
