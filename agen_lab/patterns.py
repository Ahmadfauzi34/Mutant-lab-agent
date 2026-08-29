"""Structural pattern representation and discovery — V2.32.

CPU-first, standard-library-only pattern cognition.

The subsystem deliberately separates:
- raw symbols from canonical structural identity;
- structural support from predictive reliability;
- pattern hypotheses from Evidence/truth;
- read-only discovery/navigation/audit from actual prediction assessment.

V2.32 intentionally supports exact symbolic structure only:
- exact periodic sequence structure;
- exact mirror/palindrome structure.

No fuzzy similarity, neural embedding, spatial transformation, or automatic
Q/world-model training is claimed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class PatternError(ValueError):
    pass


class PatternSourceConflict(PatternError):
    pass


class PatternPredictionError(PatternError):
    pass


class PatternKind(Enum):
    PERIODIC_SEQUENCE = "periodic_sequence"
    MIRROR_SYMMETRY = "mirror_symmetry"


class PatternRelationType(Enum):
    DERIVED_FROM = "derived_from"
    PREDICTED_BY = "predicted_by"


PatternSymbol = object
MAX_PATTERN_SEQUENCE_LENGTH = 4096


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_symbol(value, depth: int = 0):
    """Allow a small language-neutral symbolic domain.

    Tuples are supported so later spatial work can use symbols such as
    ``("LEFT_OF", "A", "B")`` without changing the pattern engine.
    """
    if depth > 8:
        raise PatternError("Pattern symbol nesting terlalu dalam")

    if value is None or isinstance(value, (bool, int, str)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise PatternError("Pattern symbol float harus finite")
        return

    if isinstance(value, tuple):
        for item in value:
            _validate_symbol(item, depth + 1)
        return

    raise PatternError(
        "Pattern symbol harus JSON-scalar atau tuple simbolik; "
        f"dapat {type(value).__module__}.{type(value).__name__}"
    )


def _symbol_key(value):
    _validate_symbol(value)
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(
                _symbol_key(item)
                for item in value
            ),
        )
    raise PatternError("Pattern symbol key tidak didukung")


def _symbol_equal(left, right) -> bool:
    return _symbol_key(left) == _symbol_key(right)


def _sequence_fingerprint(
    sequence: Sequence[PatternSymbol],
) -> str:
    payload = [
        _symbol_key(symbol)
        for symbol in sequence
    ]
    return _stable_signature({
        "typed_sequence": payload,
    })


def canonicalize_symbol_sequence(
    sequence: Sequence[PatternSymbol],
) -> Tuple[str, ...]:
    """Canonical equality structure by first occurrence in O(n).

    ``A B A B`` and ``7 3 7 3`` both become
    ``("$0", "$1", "$0", "$1")``. Python-equal but type-distinct values
    such as ``True`` and ``1`` remain distinct symbols.
    """
    raw = tuple(sequence)
    if not raw:
        raise PatternError("Pattern sequence tidak boleh kosong")
    if len(raw) > MAX_PATTERN_SEQUENCE_LENGTH:
        raise PatternError(
            "Pattern sequence melewati batas "
            f"{MAX_PATTERN_SEQUENCE_LENGTH}"
        )

    key_to_index = {}
    canonical: List[str] = []

    for symbol in raw:
        key = _symbol_key(symbol)
        match_index = key_to_index.get(key)
        if match_index is None:
            match_index = len(key_to_index)
            key_to_index[key] = match_index
        canonical.append(f"${match_index}")

    return tuple(canonical)


def _minimal_exact_period(
    canonical: Tuple[str, ...],
) -> Optional[int]:
    """Return the shortest exact period in O(n), partial final cycle allowed."""
    n = len(canonical)
    if n < 2:
        return None

    prefix = [0] * n
    for index in range(1, n):
        cursor = prefix[index - 1]
        while (
            cursor > 0
            and canonical[index]
            != canonical[cursor]
        ):
            cursor = prefix[cursor - 1]
        if canonical[index] == canonical[cursor]:
            cursor += 1
        prefix[index] = cursor

    period = n - prefix[-1]
    if period <= 0 or n < 2 * period:
        return None

    # Border-derived period is exact over the observed finite prefix.
    return period

def _stable_signature(payload: Dict) -> str:
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class StructuralPatternCandidate:
    kind: PatternKind
    canonical_form: Tuple[str, ...]
    period: Optional[int]
    sequence_length: int
    structural_coverage: float
    compression_gain: float
    semantic_signature: str

    def __post_init__(self):
        if self.sequence_length <= 0:
            raise PatternError("sequence_length harus positif")
        if not self.canonical_form:
            raise PatternError("canonical_form tidak boleh kosong")
        if not (0.0 <= self.structural_coverage <= 1.0):
            raise PatternError("structural_coverage harus 0..1")
        if self.period is not None and self.period <= 0:
            raise PatternError("period harus positif atau None")

    @property
    def predictive(self) -> bool:
        return self.kind == PatternKind.PERIODIC_SEQUENCE


@dataclass(frozen=True)
class StructuralPatternDefinition:
    pattern_id: str
    semantic_signature: str
    namespace: str
    belief_context_id: str
    kind: PatternKind
    canonical_form: Tuple[str, ...]
    period: Optional[int]
    created_at: int

    def __post_init__(self):
        for name, value in (
            ("pattern_id", self.pattern_id),
            ("semantic_signature", self.semantic_signature),
            ("namespace", self.namespace),
            ("belief_context_id", self.belief_context_id),
        ):
            if not isinstance(value, str) or not value:
                raise PatternError(f"{name} tidak boleh kosong")
        if not self.canonical_form:
            raise PatternError("canonical_form tidak boleh kosong")


@dataclass
class StructuralPatternHypothesis:
    definition: StructuralPatternDefinition
    support_count: int = 0
    prediction_trials: int = 0
    prediction_correct: int = 0
    structural_strength_sum: float = 0.0
    recent_source_instance_ids: List[str] = field(default_factory=list)
    first_observed_at: Optional[int] = None
    last_observed_at: Optional[int] = None

    @property
    def prediction_failures(self) -> int:
        return max(
            0,
            self.prediction_trials - self.prediction_correct,
        )

    @property
    def prediction_reliability(self) -> float:
        # Laplace/Beta(1,1) smoothing. This is predictive reliability, not
        # truth confidence and not evidence support.
        return (
            self.prediction_correct + 1.0
        ) / (
            self.prediction_trials + 2.0
        )

    @property
    def average_structural_strength(self) -> float:
        if self.support_count <= 0:
            return 0.0
        return self.structural_strength_sum / self.support_count

    def register_support(
        self,
        instance_id: str,
        structural_strength: float,
        observed_at: int,
        source_history_limit: int = 64,
    ):
        self.support_count += 1
        self.structural_strength_sum += max(
            0.0,
            min(1.0, float(structural_strength)),
        )
        if self.first_observed_at is None:
            self.first_observed_at = int(observed_at)
        self.last_observed_at = int(observed_at)
        self.recent_source_instance_ids.append(instance_id)
        if len(self.recent_source_instance_ids) > source_history_limit:
            del self.recent_source_instance_ids[
                : len(self.recent_source_instance_ids)
                - source_history_limit
            ]

    def assess_prediction(self, correct: bool):
        self.prediction_trials += 1
        if correct:
            self.prediction_correct += 1


@dataclass(frozen=True)
class StructuralPatternInstance:
    instance_id: str
    namespace: str
    belief_context_id: str
    sequence: Tuple[PatternSymbol, ...]
    canonical_sequence: Tuple[str, ...]
    observed_at: int
    source_id: Optional[str]
    discovered_pattern_ids: Tuple[str, ...]


@dataclass
class StructuralPatternPrediction:
    prediction_id: str
    pattern_id: str
    semantic_signature: str
    namespace: str
    belief_context_id: str
    prefix: Tuple[PatternSymbol, ...]
    expected_symbol: PatternSymbol
    expected_canonical_variable: str
    generated_at: int
    empirical_support_count: int
    prediction_trials_at_generation: int
    prediction_reliability_at_generation: float
    structural_strength_at_generation: float
    assessed: bool = False
    actual_symbol: Optional[PatternSymbol] = None
    correct: Optional[bool] = None

    @property
    def is_experience(self) -> bool:
        return False

    @property
    def is_truth(self) -> bool:
        return False


@dataclass(frozen=True)
class StructuralPatternPredictionAssessment:
    prediction_id: str
    pattern_id: str
    expected_symbol: PatternSymbol
    actual_symbol: PatternSymbol
    correct: bool
    assessed_at: int
    reliability_before: float
    reliability_after: float

    @property
    def is_evidence(self) -> bool:
        return False


@dataclass(frozen=True)
class PatternRelation:
    relation_id: str
    relation_type: PatternRelationType
    from_node_id: str
    to_node_id: str
    created_at: int


class StructuralPatternEngine:
    """Pure deterministic structural discovery and matching."""

    @staticmethod
    def discover(
        sequence: Sequence[PatternSymbol],
    ) -> Tuple[StructuralPatternCandidate, ...]:
        raw = tuple(sequence)
        canonical = canonicalize_symbol_sequence(raw)
        n = len(canonical)
        candidates: List[StructuralPatternCandidate] = []

        # Exact periodicity, including partial final cycle (ABABA).
        period = _minimal_exact_period(canonical)

        if period is not None:
            motif = canonical[:period]
            compression_gain = max(
                0.0,
                float(n - (period + 1)),
            )
            strength = max(
                0.0,
                min(
                    1.0,
                    compression_gain / max(1.0, float(n)),
                ),
            )
            signature = _stable_signature({
                "kind": PatternKind.PERIODIC_SEQUENCE.value,
                "canonical_form": list(motif),
                "period": period,
            })
            candidates.append(
                StructuralPatternCandidate(
                    kind=PatternKind.PERIODIC_SEQUENCE,
                    canonical_form=motif,
                    period=period,
                    sequence_length=n,
                    structural_coverage=1.0,
                    compression_gain=compression_gain,
                    semantic_signature=signature,
                )
            )

        # Exact structural mirror. The full equality template is retained so
        # no fuzzy or approximate symmetry is implied.
        if (
            n >= 3
            and canonical == tuple(reversed(canonical))
        ):
            compression_gain = max(
                0.0,
                float((n // 2) - 1),
            )
            signature = _stable_signature({
                "kind": PatternKind.MIRROR_SYMMETRY.value,
                "canonical_form": list(canonical),
                "period": None,
            })
            candidates.append(
                StructuralPatternCandidate(
                    kind=PatternKind.MIRROR_SYMMETRY,
                    canonical_form=canonical,
                    period=None,
                    sequence_length=n,
                    structural_coverage=1.0,
                    compression_gain=compression_gain,
                    semantic_signature=signature,
                )
            )

        return tuple(candidates)

    @staticmethod
    def match_periodic_prefix(
        sequence: Sequence[PatternSymbol],
        definition: StructuralPatternDefinition,
    ) -> Optional[Tuple[PatternSymbol, str]]:
        if definition.kind != PatternKind.PERIODIC_SEQUENCE:
            return None
        period = definition.period
        if period is None or period <= 0:
            return None

        raw = tuple(sequence)
        if len(raw) < period:
            return None

        canonical = canonicalize_symbol_sequence(raw)
        motif = definition.canonical_form

        if len(motif) != period:
            return None

        if not all(
            canonical[index] == motif[index % period]
            for index in range(len(canonical))
        ):
            return None

        phase = len(raw) % period
        expected_variable = motif[phase]

        for index, variable in enumerate(canonical):
            if variable == expected_variable:
                return raw[index], expected_variable

        return None


class StructuralPatternStore:
    """Bounded operational structural-pattern memory.

    The abstraction/hypothesis store is durable through normal checkpoint and
    portable-state serialization. Exact source pattern instances are bounded
    operational memory in V2.32; no COLD pattern-instance archive is claimed.
    """

    def __init__(
        self,
        pattern_limit: int = 4096,
        instance_limit: int = 4096,
        prediction_limit: int = 2048,
        relation_limit: int = 8192,
        source_index_limit: Optional[int] = None,
    ):
        resolved_source_index_limit = (
            max(16, int(instance_limit) * 2)
            if source_index_limit is None
            else int(source_index_limit)
        )
        for name, value in (
            ("pattern_limit", pattern_limit),
            ("instance_limit", instance_limit),
            ("prediction_limit", prediction_limit),
            ("relation_limit", relation_limit),
            ("source_index_limit", resolved_source_index_limit),
        ):
            if int(value) <= 0:
                raise PatternError(f"{name} harus positif")

        self.pattern_limit = int(pattern_limit)
        self.instance_limit = int(instance_limit)
        self.prediction_limit = int(prediction_limit)
        self.relation_limit = int(relation_limit)
        self.source_index_limit = resolved_source_index_limit

        self.patterns: OrderedDict[
            str, StructuralPatternHypothesis
        ] = OrderedDict()
        self.instances: OrderedDict[
            str, StructuralPatternInstance
        ] = OrderedDict()
        self.predictions: OrderedDict[
            str, StructuralPatternPrediction
        ] = OrderedDict()
        self.relations: OrderedDict[
            str, PatternRelation
        ] = OrderedDict()

        self.source_index: OrderedDict[
            Tuple[str, str, str],
            Tuple[str, str],
        ] = OrderedDict()
        self.total_instances_seen = 0
        self.total_predictions_generated = 0
        self.total_prediction_assessments = 0

    @staticmethod
    def _pattern_id(
        *,
        semantic_signature: str,
        namespace: str,
        belief_context_id: str,
    ) -> str:
        digest = hashlib.sha256(
            _canonical_json({
                "semantic_signature": semantic_signature,
                "namespace": namespace,
                "belief_context_id": belief_context_id,
            }).encode("utf-8")
        ).hexdigest()
        return f"pattern:{digest[:24]}"

    @staticmethod
    def _relation_id(
        relation_type: PatternRelationType,
        from_node_id: str,
        to_node_id: str,
    ) -> str:
        digest = hashlib.sha256(
            _canonical_json({
                "type": relation_type.value,
                "from": from_node_id,
                "to": to_node_id,
            }).encode("utf-8")
        ).hexdigest()
        return f"prel:{digest[:24]}"

    def _drop_node_relations(self, node_id: str):
        for relation_id in [
            rid
            for rid, relation in self.relations.items()
            if (
                relation.from_node_id == node_id
                or relation.to_node_id == node_id
            )
        ]:
            self.relations.pop(relation_id, None)

    def _bounded_insert(
        self,
        mapping: OrderedDict,
        key: str,
        value,
        limit: int,
        *,
        drop_relations: bool,
    ):
        mapping[key] = value
        mapping.move_to_end(key)
        while len(mapping) > limit:
            old_key, _ = mapping.popitem(last=False)
            if drop_relations:
                self._drop_node_relations(old_key)

    def _add_relation(
        self,
        relation_type: PatternRelationType,
        from_node_id: str,
        to_node_id: str,
        created_at: int,
    ):
        relation_id = self._relation_id(
            relation_type,
            from_node_id,
            to_node_id,
        )
        relation = PatternRelation(
            relation_id=relation_id,
            relation_type=relation_type,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            created_at=int(created_at),
        )
        self._bounded_insert(
            self.relations,
            relation_id,
            relation,
            self.relation_limit,
            drop_relations=False,
        )

    def observe_sequence(
        self,
        sequence: Sequence[PatternSymbol],
        *,
        namespace: str,
        belief_context_id: str,
        observed_at: int,
        source_id: Optional[str] = None,
    ) -> Dict:
        if not isinstance(namespace, str) or not namespace:
            raise PatternError("namespace tidak boleh kosong")
        if not isinstance(belief_context_id, str) or not belief_context_id:
            raise PatternError("belief_context_id tidak boleh kosong")

        raw = tuple(sequence)
        canonical = canonicalize_symbol_sequence(raw)
        candidates = StructuralPatternEngine.discover(raw)

        if source_id is not None:
            if not isinstance(source_id, str) or not source_id:
                raise PatternError("source_id harus string non-empty atau None")
            source_key = (
                belief_context_id,
                namespace,
                source_id,
            )
            existing_info = self.source_index.get(source_key)
            if existing_info is not None:
                existing_id, existing_fingerprint = existing_info
                incoming_fingerprint = _sequence_fingerprint(raw)
                if existing_fingerprint != incoming_fingerprint:
                    raise PatternSourceConflict(
                        "source_id pattern yang sama membawa sequence berbeda"
                    )
                self.source_index.move_to_end(source_key)
                existing = self.instances.get(existing_id)
                if existing is None:
                    # Exact operational instance aged out, but bounded source
                    # identity still prevents retry double-counting and detects
                    # changed payloads through the typed fingerprint.
                    return {
                        "deduplicated": True,
                        "instance": None,
                        "discovered_patterns": (),
                        "source_history_evicted": True,
                    }
                return {
                    "deduplicated": True,
                    "instance": existing,
                    "discovered_patterns": tuple(
                        self.patterns[pattern_id]
                        for pattern_id in existing.discovered_pattern_ids
                        if pattern_id in self.patterns
                    ),
                    "source_history_evicted": False,
                }

        instance_id = "pinst-" + uuid.uuid4().hex
        pattern_ids: List[str] = []
        hypotheses: List[StructuralPatternHypothesis] = []

        for candidate in candidates:
            pattern_id = self._pattern_id(
                semantic_signature=candidate.semantic_signature,
                namespace=namespace,
                belief_context_id=belief_context_id,
            )
            hypothesis = self.patterns.get(pattern_id)

            if hypothesis is None:
                definition = StructuralPatternDefinition(
                    pattern_id=pattern_id,
                    semantic_signature=candidate.semantic_signature,
                    namespace=namespace,
                    belief_context_id=belief_context_id,
                    kind=candidate.kind,
                    canonical_form=candidate.canonical_form,
                    period=candidate.period,
                    created_at=int(observed_at),
                )
                hypothesis = StructuralPatternHypothesis(
                    definition=definition,
                )
                self._bounded_insert(
                    self.patterns,
                    pattern_id,
                    hypothesis,
                    self.pattern_limit,
                    drop_relations=True,
                )
            else:
                self.patterns.move_to_end(pattern_id)

            structural_strength = (
                candidate.compression_gain
                / max(1.0, float(candidate.sequence_length))
            )
            hypothesis.register_support(
                instance_id,
                structural_strength,
                int(observed_at),
            )
            pattern_ids.append(pattern_id)
            hypotheses.append(hypothesis)

        instance = StructuralPatternInstance(
            instance_id=instance_id,
            namespace=namespace,
            belief_context_id=belief_context_id,
            sequence=raw,
            canonical_sequence=canonical,
            observed_at=int(observed_at),
            source_id=source_id,
            discovered_pattern_ids=tuple(pattern_ids),
        )
        self._bounded_insert(
            self.instances,
            instance_id,
            instance,
            self.instance_limit,
            drop_relations=True,
        )
        self.total_instances_seen += 1

        if source_id is not None:
            source_key = (
                belief_context_id,
                namespace,
                source_id,
            )
            self.source_index[source_key] = (
                instance_id,
                _sequence_fingerprint(raw),
            )
            self.source_index.move_to_end(source_key)
            while len(self.source_index) > self.source_index_limit:
                self.source_index.popitem(last=False)

        for pattern_id in pattern_ids:
            if pattern_id not in self.patterns:
                continue
            self._add_relation(
                PatternRelationType.DERIVED_FROM,
                pattern_id,
                instance_id,
                int(observed_at),
            )

        return {
            "deduplicated": False,
            "instance": instance,
            "discovered_patterns": tuple(hypotheses),
            "source_history_evicted": False,
        }

    def hypotheses(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        kind: Optional[PatternKind] = None,
    ) -> Tuple[StructuralPatternHypothesis, ...]:
        values = []
        for hypothesis in self.patterns.values():
            definition = hypothesis.definition
            if namespace is not None and definition.namespace != namespace:
                continue
            if (
                belief_context_id is not None
                and definition.belief_context_id != belief_context_id
            ):
                continue
            if kind is not None and definition.kind != kind:
                continue
            values.append(hypothesis)
        return tuple(values)

    def predict_next(
        self,
        sequence: Sequence[PatternSymbol],
        *,
        namespace: str,
        belief_context_id: str,
        generated_at: int,
    ) -> Optional[StructuralPatternPrediction]:
        raw = tuple(sequence)
        canonicalize_symbol_sequence(raw)

        matches = []
        for hypothesis in self.hypotheses(
            namespace=namespace,
            belief_context_id=belief_context_id,
            kind=PatternKind.PERIODIC_SEQUENCE,
        ):
            match = StructuralPatternEngine.match_periodic_prefix(
                raw,
                hypothesis.definition,
            )
            if match is None:
                continue
            expected_symbol, expected_variable = match
            matches.append(
                (
                    hypothesis.prediction_reliability,
                    hypothesis.prediction_trials,
                    hypothesis.support_count,
                    hypothesis.average_structural_strength,
                    -int(hypothesis.definition.period or 0),
                    hypothesis.definition.pattern_id,
                    hypothesis,
                    expected_symbol,
                    expected_variable,
                )
            )

        if not matches:
            return None

        matches.sort(reverse=True, key=lambda item: item[:6])
        (
            _,
            _,
            _,
            _,
            _,
            _,
            hypothesis,
            expected_symbol,
            expected_variable,
        ) = matches[0]

        prediction_id = "ppred-" + uuid.uuid4().hex
        prediction = StructuralPatternPrediction(
            prediction_id=prediction_id,
            pattern_id=hypothesis.definition.pattern_id,
            semantic_signature=hypothesis.definition.semantic_signature,
            namespace=namespace,
            belief_context_id=belief_context_id,
            prefix=raw,
            expected_symbol=expected_symbol,
            expected_canonical_variable=expected_variable,
            generated_at=int(generated_at),
            empirical_support_count=hypothesis.support_count,
            prediction_trials_at_generation=hypothesis.prediction_trials,
            prediction_reliability_at_generation=(
                hypothesis.prediction_reliability
            ),
            structural_strength_at_generation=(
                hypothesis.average_structural_strength
            ),
        )

        self._bounded_insert(
            self.predictions,
            prediction_id,
            prediction,
            self.prediction_limit,
            drop_relations=True,
        )
        self.total_predictions_generated += 1
        self._add_relation(
            PatternRelationType.PREDICTED_BY,
            prediction_id,
            hypothesis.definition.pattern_id,
            int(generated_at),
        )
        return prediction

    def assess_prediction(
        self,
        prediction_id: str,
        actual_symbol: PatternSymbol,
        *,
        assessed_at: int,
    ) -> StructuralPatternPredictionAssessment:
        _validate_symbol(actual_symbol)

        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            raise PatternPredictionError(
                f"Pattern prediction tidak ditemukan: {prediction_id}"
            )
        if prediction.assessed:
            raise PatternPredictionError(
                f"Pattern prediction sudah dinilai: {prediction_id}"
            )

        hypothesis = self.patterns.get(prediction.pattern_id)
        if hypothesis is None:
            raise PatternPredictionError(
                "Pattern hypothesis prediction sudah tidak tersedia"
            )

        before = hypothesis.prediction_reliability
        correct = _symbol_equal(
            prediction.expected_symbol,
            actual_symbol,
        )
        hypothesis.assess_prediction(correct)
        after = hypothesis.prediction_reliability

        prediction.assessed = True
        prediction.actual_symbol = actual_symbol
        prediction.correct = bool(correct)
        self.predictions.move_to_end(prediction_id)
        self.patterns.move_to_end(prediction.pattern_id)
        self.total_prediction_assessments += 1

        return StructuralPatternPredictionAssessment(
            prediction_id=prediction_id,
            pattern_id=prediction.pattern_id,
            expected_symbol=prediction.expected_symbol,
            actual_symbol=actual_symbol,
            correct=bool(correct),
            assessed_at=int(assessed_at),
            reliability_before=before,
            reliability_after=after,
        )

    def relational_completion(
        self,
        node_id: str,
        *,
        max_depth: int = 2,
        relation_types: Optional[
            Iterable[PatternRelationType]
        ] = None,
    ) -> Dict:
        if max_depth < 0 or max_depth > 8:
            raise PatternError("max_depth relational completion harus 0..8")

        allowed = (
            None
            if relation_types is None
            else set(relation_types)
        )

        adjacency: Dict[str, List[Tuple[str, PatternRelation]]] = {}
        for relation in self.relations.values():
            if (
                allowed is not None
                and relation.relation_type not in allowed
            ):
                continue
            adjacency.setdefault(
                relation.from_node_id,
                [],
            ).append(
                (
                    relation.to_node_id,
                    relation,
                )
            )
            adjacency.setdefault(
                relation.to_node_id,
                [],
            ).append(
                (
                    relation.from_node_id,
                    relation,
                )
            )

        known_nodes = (
            set(self.patterns)
            | set(self.instances)
            | set(self.predictions)
        )
        if node_id not in known_nodes:
            raise PatternError(
                f"Pattern graph node tidak ditemukan: {node_id}"
            )

        visited = {node_id}
        frontier = deque([(node_id, 0)])
        reached = []
        traversed_relation_ids = set()

        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                continue

            for neighbor, relation in adjacency.get(current, ()):
                traversed_relation_ids.add(relation.relation_id)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                reached.append({
                    "node_id": neighbor,
                    "depth": depth + 1,
                    "via_relation_id": relation.relation_id,
                    "via_relation_type": relation.relation_type.value,
                })
                frontier.append((neighbor, depth + 1))

        return {
            "source_node_id": node_id,
            "max_depth": max_depth,
            "reached": tuple(reached),
            "reached_count": len(reached),
            "traversed_relation_ids": tuple(
                sorted(traversed_relation_ids)
            ),
            "learning_mutation": False,
        }

    def topology_audit(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
        include_predictions: bool = False,
        hub_threshold: int = 4,
    ) -> Dict:
        if hub_threshold <= 0:
            raise PatternError("hub_threshold harus positif")

        pattern_ids = {
            pid
            for pid, hypothesis in self.patterns.items()
            if (
                (namespace is None or hypothesis.definition.namespace == namespace)
                and (
                    belief_context_id is None
                    or hypothesis.definition.belief_context_id
                    == belief_context_id
                )
            )
        }
        instance_ids = {
            iid
            for iid, instance in self.instances.items()
            if (
                (namespace is None or instance.namespace == namespace)
                and (
                    belief_context_id is None
                    or instance.belief_context_id == belief_context_id
                )
            )
        }
        prediction_ids = (
            {
                pid
                for pid, prediction in self.predictions.items()
                if (
                    include_predictions
                    and (
                        namespace is None
                        or prediction.namespace == namespace
                    )
                    and (
                        belief_context_id is None
                        or prediction.belief_context_id
                        == belief_context_id
                    )
                )
            }
            if include_predictions
            else set()
        )

        nodes = pattern_ids | instance_ids | prediction_ids
        relations = [
            relation
            for relation in self.relations.values()
            if (
                relation.from_node_id in nodes
                and relation.to_node_id in nodes
            )
        ]

        parent = {node: node for node in nodes}
        rank = {node: 0 for node in nodes}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            a = find(left)
            b = find(right)
            if a == b:
                return
            if rank[a] < rank[b]:
                a, b = b, a
            parent[b] = a
            if rank[a] == rank[b]:
                rank[a] += 1

        degree = {node: 0 for node in nodes}
        relation_type_counts: Dict[str, int] = {}
        for relation in relations:
            union(
                relation.from_node_id,
                relation.to_node_id,
            )
            degree[relation.from_node_id] += 1
            degree[relation.to_node_id] += 1
            key = relation.relation_type.value
            relation_type_counts[key] = (
                relation_type_counts.get(key, 0) + 1
            )

        components: Dict[str, List[str]] = {}
        for node in nodes:
            components.setdefault(
                find(node),
                [],
            ).append(node)

        component_count = len(components)
        edge_count = len(relations)
        node_count = len(nodes)
        cycle_rank = max(
            0,
            edge_count - node_count + component_count,
        )
        isolated = sorted(
            node
            for node, count in degree.items()
            if count == 0
        )
        hubs = tuple(
            sorted(
                (
                    (node, count)
                    for node, count in degree.items()
                    if count >= hub_threshold
                ),
                key=lambda item: (-item[1], item[0]),
            )
        )

        return {
            "node_count": node_count,
            "pattern_count": len(pattern_ids),
            "instance_count": len(instance_ids),
            "prediction_count": len(prediction_ids),
            "relation_count": edge_count,
            "relation_type_counts": relation_type_counts,
            "connected_components": component_count,
            "cycle_rank": cycle_rank,
            "isolated_nodes": tuple(isolated),
            "hub_nodes": hubs,
            "hub_threshold": hub_threshold,
            "learning_mutation": False,
            "semantic_note": (
                "Topology metrics are structural diagnostics only; "
                "not truth confidence, intelligence, or belief-context drift."
            ),
        }

    def state(
        self,
        *,
        namespace: Optional[str] = None,
        belief_context_id: Optional[str] = None,
    ) -> Dict:
        hypotheses = self.hypotheses(
            namespace=namespace,
            belief_context_id=belief_context_id,
        )
        return {
            "patterns": len(hypotheses),
            "periodic_patterns": sum(
                1
                for item in hypotheses
                if item.definition.kind
                == PatternKind.PERIODIC_SEQUENCE
            ),
            "mirror_patterns": sum(
                1
                for item in hypotheses
                if item.definition.kind
                == PatternKind.MIRROR_SYMMETRY
            ),
            "operational_instances": sum(
                1
                for item in self.instances.values()
                if (
                    (namespace is None or item.namespace == namespace)
                    and (
                        belief_context_id is None
                        or item.belief_context_id == belief_context_id
                    )
                )
            ),
            "operational_predictions": sum(
                1
                for item in self.predictions.values()
                if (
                    (namespace is None or item.namespace == namespace)
                    and (
                        belief_context_id is None
                        or item.belief_context_id == belief_context_id
                    )
                )
            ),
            "total_instances_seen": self.total_instances_seen,
            "total_predictions_generated": self.total_predictions_generated,
            "total_prediction_assessments": (
                self.total_prediction_assessments
            ),
            "limits": {
                "patterns": self.pattern_limit,
                "instances": self.instance_limit,
                "predictions": self.prediction_limit,
                "relations": self.relation_limit,
                "source_index": self.source_index_limit,
            },
            "source_identities_retained": len(self.source_index),
        }


# Canonical semantic type identity for pickle + portable state.
_CANONICAL_MODULE = "agen_kognitif_v2_28"
for _type in (
    PatternError,
    PatternSourceConflict,
    PatternPredictionError,
    PatternKind,
    PatternRelationType,
    StructuralPatternCandidate,
    StructuralPatternDefinition,
    StructuralPatternHypothesis,
    StructuralPatternInstance,
    StructuralPatternPrediction,
    StructuralPatternPredictionAssessment,
    PatternRelation,
    StructuralPatternEngine,
    StructuralPatternStore,
):
    _type.__module__ = _CANONICAL_MODULE


__all__ = [
    "PatternError",
    "PatternSourceConflict",
    "PatternPredictionError",
    "PatternKind",
    "PatternRelationType",
    "StructuralPatternCandidate",
    "StructuralPatternDefinition",
    "StructuralPatternHypothesis",
    "StructuralPatternInstance",
    "StructuralPatternPrediction",
    "StructuralPatternPredictionAssessment",
    "PatternRelation",
    "StructuralPatternEngine",
    "StructuralPatternStore",
    "canonicalize_symbol_sequence",
    "MAX_PATTERN_SEQUENCE_LENGTH",
]
