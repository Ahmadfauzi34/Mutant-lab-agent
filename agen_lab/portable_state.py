"""Language-neutral portable cognitive state — V2.41.

The portable state is a single SQLite database.  It contains the existing COLD
archive tables plus an explicit JSON object-graph schema for durable cognitive
runtime state.  It never stores pickle or executable Python bytecode.

The graph format is intentionally language-neutral:
- primitive JSON values;
- stable semantic type IDs (``agen/<TypeName>``), not Python module paths;
- explicit node/reference identity for shared objects and cycles;
- explicit tuple/set/ordered-map kinds;
- additive SQLite tables prefixed ``portable_state_``.

A Rust/WASM implementation can therefore read the database without importing
Python.  Python restoration remains available as the reference implementation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid

from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PORTABLE_STATE_MAGIC = "AGEN_LANGUAGE_NEUTRAL_COGNITIVE_STATE"
PORTABLE_STATE_SCHEMA_VERSION = 1
PORTABLE_GRAPH_SCHEMA = "agen-object-graph-json-v1"
PORTABLE_TYPE_PREFIX = "agen/"


class PortableStateError(RuntimeError):
    pass


class PortableStateSchemaError(PortableStateError):
    pass


class PortableStateTypeError(PortableStateError):
    pass


def _core_version() -> str:
    module = sys.modules.get("agen_kognitif_v2_28")
    if module is None:
        raise PortableStateError(
            "Canonical compatibility module belum tersedia"
        )
    return str(module.CORE_VERSION)


def _canonical_module():
    module = sys.modules.get("agen_kognitif_v2_28")
    if module is None:
        raise PortableStateError(
            "Canonical compatibility module belum tersedia"
        )
    return module


def _semantic_type_id(cls) -> str:
    module_name = getattr(cls, "__module__", "")
    if module_name != "agen_kognitif_v2_28":
        raise PortableStateTypeError(
            "Portable state hanya menerima cognitive type canonical; "
            f"dapat {module_name}.{getattr(cls, '__name__', '?')}"
        )
    return PORTABLE_TYPE_PREFIX + cls.__name__


def _resolve_semantic_type(type_id: str):
    if (
        not isinstance(type_id, str)
        or not type_id.startswith(PORTABLE_TYPE_PREFIX)
    ):
        raise PortableStateTypeError(
            f"Portable type id tidak valid: {type_id!r}"
        )

    name = type_id[len(PORTABLE_TYPE_PREFIX):]
    if (
        not name
        or not name.replace("_", "").isalnum()
    ):
        raise PortableStateTypeError(
            f"Portable type id tidak aman: {type_id!r}"
        )

    module = _canonical_module()
    try:
        cls = getattr(module, name)
    except AttributeError as exc:
        raise PortableStateTypeError(
            f"Portable type tidak dikenal: {type_id}"
        ) from exc

    if not isinstance(cls, type):
        raise PortableStateTypeError(
            f"Portable type bukan class/enum: {type_id}"
        )
    return cls


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash_rows(rows: Iterable[Tuple[int, str, Optional[str], str]]) -> str:
    digest = hashlib.sha256()
    for node_id, kind, type_id, payload_json in rows:
        digest.update(str(int(node_id)).encode("ascii"))
        digest.update(b"\x1f")
        digest.update(kind.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update((type_id or "").encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(payload_json.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


class PortableObjectGraphCodec:
    """Encode/decode the cognitive object graph without pickle.

    The root agent's ``epistemic_archive`` attribute is deliberately external:
    the SQLite file being written *is* that durable COLD archive.  On restore a
    new EpistemicArchiveManager is bound to a runtime copy of the portable DB.
    """

    ROOT_EXTERNAL_ATTRS = frozenset({"epistemic_archive"})
    MAX_NODES = 1_000_000

    def __init__(self):
        self._object_to_node: Dict[int, int] = {}
        self._nodes: List[Dict] = []
        self._external_object_ids: Dict[int, str] = {}

    def encode(self, root) -> Dict:
        self._object_to_node = {}
        self._nodes = []
        self._external_object_ids = {}
        if hasattr(root, "epistemic_archive"):
            self._external_object_ids[id(root.epistemic_archive)] = "cold_archive"
        root_value = self._encode_value(root, is_root=True)
        if not (
            isinstance(root_value, dict)
            and "$ref" in root_value
        ):
            raise PortableStateSchemaError(
                "Portable graph root harus object reference"
            )
        return {
            "graph_schema": PORTABLE_GRAPH_SCHEMA,
            "root": root_value,
            "nodes": list(self._nodes),
        }

    def _reserve(self, value, kind: str, type_id: Optional[str] = None) -> Tuple[int, Dict]:
        object_id = id(value)
        known = self._object_to_node.get(object_id)
        if known is not None:
            return known, {}

        node_id = len(self._nodes) + 1
        if node_id > self.MAX_NODES:
            raise PortableStateSchemaError(
                "Portable graph melewati batas node"
            )

        node = {
            "node_id": node_id,
            "kind": kind,
            "type_id": type_id,
            "payload": None,
        }
        self._object_to_node[object_id] = node_id
        self._nodes.append(node)
        return node_id, node

    def _encode_value(self, value, *, is_root: bool = False):
        if value is None or isinstance(value, (bool, int, str)):
            return value

        if isinstance(value, float):
            if not math.isfinite(value):
                raise PortableStateSchemaError(
                    "NaN/Infinity tidak diizinkan dalam portable state"
                )
            return value

        if isinstance(value, Enum):
            return {
                "$enum": {
                    "type_id": _semantic_type_id(type(value)),
                    "value": self._encode_value(value.value),
                }
            }

        if isinstance(value, Path):
            return {"$path": str(value)}

        if isinstance(value, (bytes, bytearray)):
            return {
                "$bytes_b64": base64.b64encode(bytes(value)).decode("ascii")
            }

        object_id = id(value)
        external_name = self._external_object_ids.get(object_id)
        if external_name is not None:
            return {"$external": external_name}
        known = self._object_to_node.get(object_id)
        if known is not None:
            return {"$ref": known}

        if isinstance(value, list):
            node_id, node = self._reserve(value, "list")
            node["payload"] = {
                "items": [self._encode_value(item) for item in value]
            }
            return {"$ref": node_id}

        if isinstance(value, tuple):
            node_id, node = self._reserve(value, "tuple")
            node["payload"] = {
                "items": [self._encode_value(item) for item in value]
            }
            return {"$ref": node_id}

        if isinstance(value, set):
            node_id, node = self._reserve(value, "set")
            node["payload"] = {
                "items": [self._encode_value(item) for item in value]
            }
            return {"$ref": node_id}

        if isinstance(value, frozenset):
            node_id, node = self._reserve(value, "frozenset")
            node["payload"] = {
                "items": [self._encode_value(item) for item in value]
            }
            return {"$ref": node_id}

        if isinstance(value, OrderedDict):
            node_id, node = self._reserve(value, "ordered_dict")
            node["payload"] = {
                "entries": [
                    [self._encode_value(key), self._encode_value(item)]
                    for key, item in value.items()
                ]
            }
            return {"$ref": node_id}

        if isinstance(value, dict):
            node_id, node = self._reserve(value, "dict")
            node["payload"] = {
                "entries": [
                    [self._encode_value(key), self._encode_value(item)]
                    for key, item in value.items()
                ]
            }
            return {"$ref": node_id}

        if hasattr(value, "__dict__"):
            type_id = _semantic_type_id(type(value))
            node_id, node = self._reserve(value, "object", type_id)
            attrs = []
            for name, item in vars(value).items():
                if (
                    is_root
                    and name in self.ROOT_EXTERNAL_ATTRS
                ):
                    continue
                if not isinstance(name, str):
                    raise PortableStateSchemaError(
                        "Nama attribute portable harus string"
                    )
                if callable(item):
                    raise PortableStateTypeError(
                        f"Callable runtime state tidak portable: {type_id}.{name}"
                    )
                attrs.append([name, self._encode_value(item)])
            node["payload"] = {"attrs": attrs}
            return {"$ref": node_id}

        raise PortableStateTypeError(
            "Runtime value tidak punya representasi portable: "
            f"{type(value).__module__}.{type(value).__name__}"
        )

    @classmethod
    def decode(cls, graph: Dict):
        if not isinstance(graph, dict):
            raise PortableStateSchemaError("Graph payload bukan object")
        if graph.get("graph_schema") != PORTABLE_GRAPH_SCHEMA:
            raise PortableStateSchemaError(
                "Portable graph schema tidak didukung"
            )

        nodes = graph.get("nodes")
        root_value = graph.get("root")
        if not isinstance(nodes, list):
            raise PortableStateSchemaError("nodes harus list")
        if len(nodes) > cls.MAX_NODES:
            raise PortableStateSchemaError("Terlalu banyak graph nodes")

        node_map: Dict[int, Dict] = {}
        for raw in nodes:
            if not isinstance(raw, dict):
                raise PortableStateSchemaError("Node bukan object")
            node_id = raw.get("node_id")
            if not isinstance(node_id, int) or node_id <= 0:
                raise PortableStateSchemaError("node_id tidak valid")
            if node_id in node_map:
                raise PortableStateSchemaError("node_id duplikat")
            node_map[node_id] = raw

        memo: Dict[int, object] = {}
        tuple_in_progress = set()

        def decode_value(value):
            if value is None or isinstance(value, (bool, int, float, str)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise PortableStateSchemaError("Float non-finite")
                return value

            if not isinstance(value, dict):
                raise PortableStateSchemaError(
                    f"Encoded value tidak valid: {value!r}"
                )

            if set(value) == {"$ref"}:
                return decode_node(value["$ref"])

            if set(value) == {"$external"}:
                external_name = value["$external"]
                if external_name != "cold_archive":
                    raise PortableStateSchemaError(
                        f"External portable reference tidak dikenal: {external_name!r}"
                    )
                return None

            if set(value) == {"$path"}:
                path_value = value["$path"]
                if not isinstance(path_value, str):
                    raise PortableStateSchemaError("$path bukan string")
                return Path(path_value)

            if set(value) == {"$bytes_b64"}:
                raw = value["$bytes_b64"]
                if not isinstance(raw, str):
                    raise PortableStateSchemaError("$bytes_b64 bukan string")
                try:
                    return base64.b64decode(raw.encode("ascii"), validate=True)
                except Exception as exc:
                    raise PortableStateSchemaError("Base64 invalid") from exc

            if set(value) == {"$enum"}:
                spec = value["$enum"]
                if not isinstance(spec, dict):
                    raise PortableStateSchemaError("$enum invalid")
                enum_cls = _resolve_semantic_type(spec.get("type_id"))
                if not issubclass(enum_cls, Enum):
                    raise PortableStateTypeError("type_id enum bukan Enum")
                return enum_cls(decode_value(spec.get("value")))

            raise PortableStateSchemaError(
                f"Encoded special value tidak dikenal: {sorted(value)}"
            )

        def decode_node(node_id):
            if not isinstance(node_id, int) or node_id <= 0:
                raise PortableStateSchemaError("Reference node id invalid")
            if node_id in memo:
                return memo[node_id]
            raw = node_map.get(node_id)
            if raw is None:
                raise PortableStateSchemaError(
                    f"Dangling portable reference: {node_id}"
                )

            kind = raw.get("kind")
            type_id = raw.get("type_id")
            payload = raw.get("payload")
            if not isinstance(payload, dict):
                raise PortableStateSchemaError("Node payload harus object")

            if kind == "list":
                result = []
                memo[node_id] = result
                result.extend(decode_value(item) for item in payload.get("items", []))
                return result

            if kind == "set":
                result = set()
                memo[node_id] = result
                for item in payload.get("items", []):
                    result.add(decode_value(item))
                return result

            if kind == "dict":
                result = {}
                memo[node_id] = result
                for entry in payload.get("entries", []):
                    if not isinstance(entry, list) or len(entry) != 2:
                        raise PortableStateSchemaError("Dict entry invalid")
                    result[decode_value(entry[0])] = decode_value(entry[1])
                return result

            if kind == "ordered_dict":
                result = OrderedDict()
                memo[node_id] = result
                for entry in payload.get("entries", []):
                    if not isinstance(entry, list) or len(entry) != 2:
                        raise PortableStateSchemaError("OrderedDict entry invalid")
                    result[decode_value(entry[0])] = decode_value(entry[1])
                return result

            if kind in {"tuple", "frozenset"}:
                if node_id in tuple_in_progress:
                    raise PortableStateSchemaError(
                        "Cycle melalui immutable tuple/frozenset tidak didukung"
                    )
                tuple_in_progress.add(node_id)
                items = [decode_value(item) for item in payload.get("items", [])]
                tuple_in_progress.remove(node_id)
                result = tuple(items) if kind == "tuple" else frozenset(items)
                memo[node_id] = result
                return result

            if kind == "object":
                obj_cls = _resolve_semantic_type(type_id)
                if issubclass(obj_cls, Enum):
                    raise PortableStateTypeError("Enum harus encoded sebagai $enum")
                try:
                    result = obj_cls.__new__(obj_cls)
                except Exception as exc:
                    raise PortableStateTypeError(
                        f"Tidak dapat allocate {type_id}"
                    ) from exc
                memo[node_id] = result
                attrs = payload.get("attrs", [])
                if not isinstance(attrs, list):
                    raise PortableStateSchemaError("Object attrs harus list")
                for entry in attrs:
                    if not isinstance(entry, list) or len(entry) != 2:
                        raise PortableStateSchemaError("Object attr invalid")
                    name, encoded = entry
                    if not isinstance(name, str):
                        raise PortableStateSchemaError("Attribute name bukan string")
                    object.__setattr__(result, name, decode_value(encoded))
                return result

            raise PortableStateSchemaError(
                f"Portable node kind tidak dikenal: {kind!r}"
            )

        root = decode_value(root_value)
        return root


class PortableCognitiveStateManager:
    """Save/load one language-neutral SQLite cognitive-state database."""

    MANIFEST_TABLE = "portable_state_manifest"
    NODE_TABLE = "portable_state_nodes"

    @classmethod
    def _ensure_portable_schema(cls, db: sqlite3.Connection):
        db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {cls.MANIFEST_TABLE} (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                manifest_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS {cls.NODE_TABLE} (
                node_id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                type_id TEXT,
                payload_json TEXT NOT NULL
            );
            """
        )

    @classmethod
    def save(cls, agent, path) -> Dict:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            # Not mandatory, but make the intended artifact obvious.
            pass

        codec = PortableObjectGraphCodec()
        graph = codec.encode(agent)

        rows: List[Tuple[int, str, Optional[str], str]] = []
        for node in graph["nodes"]:
            payload_json = _canonical_json(node["payload"])
            rows.append((
                int(node["node_id"]),
                str(node["kind"]),
                node.get("type_id"),
                payload_json,
            ))
        rows.sort(key=lambda row: row[0])
        graph_sha256 = _hash_rows(rows)

        archive_bytes = agent.epistemic_archive.snapshot_bytes()
        cold_snapshot_sha256 = hashlib.sha256(archive_bytes).hexdigest()

        manifest = {
            "magic": PORTABLE_STATE_MAGIC,
            "portable_schema_version": PORTABLE_STATE_SCHEMA_VERSION,
            "graph_schema": PORTABLE_GRAPH_SCHEMA,
            "core_version": _core_version(),
            "format": "sqlite3",
            "language_neutral": True,
            "python_pickle": False,
            "root_node_id": int(graph["root"]["$ref"]),
            "node_count": len(rows),
            "graph_sha256": graph_sha256,
            "cold_snapshot_sha256_before_portable_tables": cold_snapshot_sha256,
            "cold_snapshot_bytes_before_portable_tables": len(archive_bytes),
            "domain_name": agent.domain.name,
            "belief_context_id": agent.belief_contexts.current_id,
            "belief_time": agent.belief_contexts.now,
            "interaction_clock": agent.interaction_clock,
            "objective_profile_instance_id": agent.objective_profile.instance_id,
            "type_identity_note": (
                "semantic agen/<TypeName>; independent of Python module paths"
            ),
        }

        temp = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
        try:
            temp.write_bytes(archive_bytes)
            db = sqlite3.connect(str(temp), timeout=30.0)
            try:
                cls._ensure_portable_schema(db)
                db.execute(f"DELETE FROM {cls.NODE_TABLE}")
                db.execute(f"DELETE FROM {cls.MANIFEST_TABLE}")
                db.executemany(
                    f"""
                    INSERT INTO {cls.NODE_TABLE}(node_id, kind, type_id, payload_json)
                    VALUES(?,?,?,?)
                    """,
                    rows,
                )
                db.execute(
                    f"INSERT INTO {cls.MANIFEST_TABLE}(singleton, manifest_json) VALUES(1,?)",
                    (_canonical_json(manifest),),
                )
                db.commit()
                integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity != "ok":
                    raise PortableStateError(
                        f"SQLite portable state integrity gagal: {integrity}"
                    )
            finally:
                db.close()
            os.replace(temp, target)
        except Exception:
            try:
                temp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        return dict(manifest)

    @classmethod
    def inspect(cls, path) -> Dict:
        target = Path(path)
        try:
            db = sqlite3.connect(
                f"file:{target}?mode=ro",
                uri=True,
                timeout=30.0,
            )
            try:
                row = db.execute(
                    f"SELECT manifest_json FROM {cls.MANIFEST_TABLE} WHERE singleton=1"
                ).fetchone()
                if row is None:
                    raise PortableStateSchemaError(
                        "Portable manifest tidak ditemukan"
                    )
                manifest = json.loads(row[0])
                integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                db.close()
        except PortableStateError:
            raise
        except Exception as exc:
            raise PortableStateError(
                "Portable cognitive-state database tidak dapat dibaca"
            ) from exc

        cls._validate_manifest(manifest)
        result = dict(manifest)
        result["sqlite_integrity"] = integrity
        result["path"] = str(target)
        return result

    @classmethod
    def _validate_manifest(cls, manifest: Dict):
        if not isinstance(manifest, dict):
            raise PortableStateSchemaError("Manifest bukan object")
        if manifest.get("magic") != PORTABLE_STATE_MAGIC:
            raise PortableStateSchemaError("Portable state magic tidak cocok")
        if manifest.get("portable_schema_version") != PORTABLE_STATE_SCHEMA_VERSION:
            raise PortableStateSchemaError(
                "Portable state schema tidak didukung"
            )
        if manifest.get("graph_schema") != PORTABLE_GRAPH_SCHEMA:
            raise PortableStateSchemaError("Portable graph schema tidak cocok")
        if manifest.get("python_pickle") is not False:
            raise PortableStateSchemaError("Portable state tidak boleh mengandung pickle")
        if manifest.get("language_neutral") is not True:
            raise PortableStateSchemaError("State tidak ditandai language-neutral")
        core_version = manifest.get("core_version")
        if not isinstance(core_version, str) or not core_version:
            raise PortableStateSchemaError("Portable core_version tidak valid")

    @classmethod
    def load(cls, path):
        target = Path(path)
        try:
            db = sqlite3.connect(
                f"file:{target}?mode=ro",
                uri=True,
                timeout=30.0,
            )
            try:
                manifest_row = db.execute(
                    f"SELECT manifest_json FROM {cls.MANIFEST_TABLE} WHERE singleton=1"
                ).fetchone()
                if manifest_row is None:
                    raise PortableStateSchemaError("Portable manifest tidak ditemukan")
                manifest = json.loads(manifest_row[0])
                cls._validate_manifest(manifest)
                current_core_version = _core_version()
                compatible_core_versions = {current_core_version}
                if current_core_version == "2.32":
                    # One-step language-neutral migration from frozen V2.31.
                    # Runtime repair adds an EMPTY pattern store and never
                    # fabricates historical pattern observations.
                    compatible_core_versions.add("2.31")
                elif current_core_version == "2.33":
                    # Explicit language-neutral migration from frozen V2.32
                    # and V2.31. V2.31 receives EMPTY pattern + spatial stores;
                    # V2.32 receives EMPTY spatial store. No historical pattern
                    # or spatial observations are fabricated.
                    compatible_core_versions.update({
                        "2.32",
                        "2.31",
                    })
                elif current_core_version == "2.34":
                    # V2.34 transformation algebra is stateless. Frozen
                    # V2.33/V2.32/V2.31 durable states are accepted explicitly;
                    # missing pattern/spatial state follows existing empty
                    # backfill rules and no transform history is fabricated.
                    compatible_core_versions.update({
                        "2.33",
                        "2.32",
                        "2.31",
                    })
                elif current_core_version == "2.35":
                    # V2.35 manipulation simulation is stateless. Frozen
                    # V2.34/V2.33/V2.32/V2.31 durable states remain explicit
                    # compatibility inputs; no counterfactual manipulation
                    # history is synthesized.
                    compatible_core_versions.update({
                        "2.34",
                        "2.33",
                        "2.32",
                        "2.31",
                    })
                elif current_core_version == "2.36":
                    # V2.36 planning is stateless. Frozen V2.35 through V2.31
                    # portable states remain explicit compatibility inputs;
                    # no plan/search/execution history is synthesized.
                    compatible_core_versions.update({
                        "2.35",
                        "2.34",
                        "2.33",
                        "2.32",
                        "2.31",
                    })
                elif current_core_version == "2.37":
                    # V2.37 adds bounded execution tickets/actual observation
                    # feedback. Frozen V2.36 through V2.31 portable states are
                    # explicit compatibility inputs and receive EMPTY V2.37
                    # execution history.
                    compatible_core_versions.update({
                        "2.36",
                        "2.35",
                        "2.34",
                        "2.33",
                        "2.32",
                        "2.31",
                    })
                elif current_core_version == "2.38":
                    # V2.38 adds durable bounded replan-attempt records. Frozen
                    # V2.37 through V2.31 portable states remain explicit
                    # compatibility inputs and receive EMPTY replan history.
                    compatible_core_versions.update({
                        "2.37",
                        "2.36",
                        "2.35",
                        "2.34",
                        "2.33",
                        "2.32",
                        "2.31",
                    })
                elif current_core_version == "2.39":
                    # V2.39 adds durable bounded recovery-policy decisions.
                    # Frozen V2.38 through V2.31 states remain explicit
                    # compatibility inputs and receive EMPTY recovery history.
                    compatible_core_versions.update({
                        "2.38",
                        "2.37",
                        "2.36",
                        "2.35",
                        "2.34",
                        "2.33",
                        "2.32",
                        "2.31",
                    })
                elif current_core_version == "2.40":
                    # V2.40 adds durable bounded reliability aggregates and
                    # confidence assessments. Frozen V2.39 through V2.31
                    # states receive EMPTY reliability history.
                    compatible_core_versions.update({
                        "2.39", "2.38", "2.37", "2.36", "2.35",
                        "2.34", "2.33", "2.32", "2.31",
                    })
                elif current_core_version == "2.41":
                    # V2.41 ranking is stateless/read-only. Frozen V2.40
                    # through V2.31 states remain explicit compatibility
                    # inputs; no ranking/reliability history is synthesized.
                    compatible_core_versions.update({
                        "2.40", "2.39", "2.38", "2.37", "2.36",
                        "2.35", "2.34", "2.33", "2.32", "2.31",
                    })
                elif current_core_version == "2.42":
                    # V2.42 replan ranking is also stateless/read-only. Frozen
                    # V2.41 through V2.31 states remain explicit compatibility
                    # inputs; no derived ranking state is synthesized.
                    compatible_core_versions.update({
                        "2.41", "2.40", "2.39", "2.38", "2.37",
                        "2.36", "2.35", "2.34", "2.33", "2.32", "2.31",
                    })

                if manifest.get("core_version") not in compatible_core_versions:
                    raise PortableStateSchemaError(
                        "Portable cognitive semantics version tidak didukung: "
                        f"{manifest.get('core_version')} not in "
                        f"{sorted(compatible_core_versions)}"
                    )

                rows = db.execute(
                    f"""
                    SELECT node_id, kind, type_id, payload_json
                    FROM {cls.NODE_TABLE}
                    ORDER BY node_id
                    """
                ).fetchall()
                integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity != "ok":
                    raise PortableStateError(
                        f"SQLite portable state integrity gagal: {integrity}"
                    )
            finally:
                db.close()
        except PortableStateError:
            raise
        except Exception as exc:
            raise PortableStateError("Portable state gagal dibaca") from exc

        expected_count = manifest.get("node_count")
        if expected_count != len(rows):
            raise PortableStateSchemaError(
                f"Portable node count mismatch: {len(rows)} != {expected_count}"
            )
        actual_graph_hash = _hash_rows(rows)
        if actual_graph_hash != manifest.get("graph_sha256"):
            raise PortableStateSchemaError("Portable graph SHA-256 mismatch")

        graph_nodes = []
        for node_id, kind, type_id, payload_json in rows:
            try:
                payload = json.loads(payload_json)
            except Exception as exc:
                raise PortableStateSchemaError(
                    f"Node payload JSON invalid: {node_id}"
                ) from exc
            graph_nodes.append({
                "node_id": int(node_id),
                "kind": kind,
                "type_id": type_id,
                "payload": payload,
            })

        graph = {
            "graph_schema": manifest["graph_schema"],
            "root": {"$ref": int(manifest["root_node_id"])},
            "nodes": graph_nodes,
        }
        agent = PortableObjectGraphCodec.decode(graph)

        canonical = _canonical_module()
        expected_agent_cls = getattr(canonical, "IntegratedCognitiveAgent")
        if not isinstance(agent, expected_agent_cls):
            raise PortableStateTypeError(
                "Portable root bukan IntegratedCognitiveAgent"
            )

        # Runtime gets its own mutable copy.  Portable metadata tables are
        # dropped from the runtime copy so the COLD archive remains exactly the
        # cognitive archive, while the source .db stays immutable/readable.
        fd, sidecar_name = tempfile.mkstemp(
            prefix="agen_portable_runtime_",
            suffix=".sqlite3",
        )
        os.close(fd)
        sidecar = Path(sidecar_name)
        shutil.copyfile(target, sidecar)
        side_db = sqlite3.connect(str(sidecar), timeout=30.0)
        try:
            side_db.execute(f"DROP TABLE IF EXISTS {cls.NODE_TABLE}")
            side_db.execute(f"DROP TABLE IF EXISTS {cls.MANIFEST_TABLE}")
            side_db.commit()
        finally:
            side_db.close()

        archive_cls = getattr(canonical, "EpistemicArchiveManager")
        archive = archive_cls(sidecar)
        archive._owned_path = True
        object.__setattr__(agent, "epistemic_archive", archive)

        if not hasattr(agent, "_repair_runtime_links"):
            raise PortableStateTypeError(
                "Portable agent tidak punya runtime-link repair"
            )
        agent._repair_runtime_links()
        return agent, dict(manifest)


__all__ = [
    "PORTABLE_STATE_MAGIC",
    "PORTABLE_STATE_SCHEMA_VERSION",
    "PORTABLE_GRAPH_SCHEMA",
    "PortableStateError",
    "PortableStateSchemaError",
    "PortableStateTypeError",
    "PortableObjectGraphCodec",
    "PortableCognitiveStateManager",
]
