"""Domain graph + spatio-temporal planning — physical extraction M6A.

Owns domain knowledge-graph primitives and CPU spatio-temporal planning.
No dependency on the compatibility kernel.
"""
from __future__ import annotations

import heapq
import math

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

SAFETY_CLEARANCE = 1.5
COLLISION_MARGIN = 1.0

class RelationType(str, Enum):
    REQUIRES = "requires"
    USES = "uses"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


RELATION_WEIGHTS = {
    RelationType.REQUIRES: 1.0,
    RelationType.USES: 0.8,
    RelationType.SUPPORTS: 0.6,
    RelationType.CONTRADICTS: 0.1,
}

@dataclass(frozen=True)
class Concept:
    name: str
    concept_type: str


@dataclass(frozen=True)
class Relation:
    subject: str
    relation: RelationType
    object: str


class Domain:
    def __init__(self, name: str):
        self.name = name


class KnowledgeBase:
    def __init__(self, domain: Domain):
        self.domain = domain
        self.concepts: Dict[str, Concept] = {}
        self.relations: List[Relation] = []

    def add(self, concept: Concept):
        self.concepts[concept.name] = concept

    def connect(self, subject: str, relation: RelationType, object: str):
        if subject not in self.concepts or object not in self.concepts:
            raise ValueError(f"Konsep tidak dikenal: {subject} -> {object}")
        rel = Relation(subject, relation, object)
        if rel not in self.relations:
            self.relations.append(rel)

    # PERBAIKAN #6: RELATION_WEIGHTS kini benar-benar dipakai.
    # Menghitung kekuatan jalur terkuat (max-product) dari konsep sumber
    # ke konsep target mengikuti arah relasi. 0.0 jika tak terhubung.
    def relation_strength(self, source: str, target: str) -> float:
        if source == target:
            return 1.0
        if source not in self.concepts or target not in self.concepts:
            return 0.0

        adj: Dict[str, List[Tuple[str, float]]] = {}
        for r in self.relations:
            adj.setdefault(r.subject, []).append((r.object, RELATION_WEIGHTS[r.relation]))

        best: Dict[str, float] = {source: 1.0}
        queue = [source]
        while queue:
            curr = queue.pop(0)
            for nxt, w in adj.get(curr, []):
                cand = best[curr] * w
                if cand > best.get(nxt, 0.0):
                    best[nxt] = cand
                    queue.append(nxt)
        return best.get(target, 0.0)


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def manhattan(self, other: "Point") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class SpaceTimeNode:
    p: Point
    t: int


@dataclass
class MovingObstacle:
    id: str
    trajectory: List[Point]
    is_looping: bool = True

    def position_at(self, t: int) -> Point:
        if not self.trajectory:
            raise ValueError("Trajectory rintangan kosong")
        if self.is_looping:
            return self.trajectory[t % len(self.trajectory)]
        return self.trajectory[min(t, len(self.trajectory) - 1)]


class SpatioTemporalCostmap:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.static_obstacles: Set[Point] = set()
        self.moving_obstacles: List[MovingObstacle] = []

    def add_static_obstacle(self, p: Point):
        if 0 <= p.x < self.width and 0 <= p.y < self.height:
            self.static_obstacles.add(p)

    def add_moving_obstacle(self, obs: MovingObstacle):
        self.moving_obstacles.append(obs)

    def is_occupied_at(self, p: Point, t: int) -> bool:
        if p.x < 0 or p.x >= self.width or p.y < 0 or p.y >= self.height:
            return True
        if p in self.static_obstacles:
            return True
        for obs in self.moving_obstacles:
            if obs.position_at(t) == p:
                return True
        return False

    def is_vertex_or_swap_conflict(self, p_curr: Point, p_next: Point, t: int) -> bool:
        if self.is_occupied_at(p_next, t + 1):
            return True
        for obs in self.moving_obstacles:
            if obs.position_at(t) == p_next and obs.position_at(t + 1) == p_curr:
                return True
        return False

    def get_safety_cost(self, p: Point, t: int) -> float:
        cost = 0.0
        for obs in self.moving_obstacles:
            dist = p.euclidean(obs.position_at(t))
            # PERBAIKAN #4: radius penalti memakai konstanta terpadu
            if dist <= SAFETY_CLEARANCE:
                cost += max(0.0, SAFETY_CLEARANCE - dist) * 5.0
        return cost


class SpatioTemporalPlanner:
    def __init__(self, costmap: SpatioTemporalCostmap):
        self.costmap = costmap

    def plan_spacetime_path(self, start: Point, goal: Point, start_time: int = 0, max_steps: int = 60) -> Optional[List[SpaceTimeNode]]:
        if self.costmap.is_occupied_at(start, start_time):
            return None

        frontier = []
        counter = 0
        start_node = SpaceTimeNode(start, start_time)
        heapq.heappush(frontier, (start.manhattan(goal), counter, start_node))

        came_from: Dict[SpaceTimeNode, SpaceTimeNode] = {}
        cost_so_far: Dict[SpaceTimeNode, float] = {start_node: 0.0}

        actions = [
            Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0),
            Point(0, 0)  # WAIT IN PLACE
        ]

        while frontier:
            _, _, current = heapq.heappop(frontier)

            if current.p == goal:
                path = []
                curr = current
                while curr != start_node:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start_node)
                path.reverse()
                return path

            if current.t >= start_time + max_steps:
                continue

            for act in actions:
                next_p = current.p + act
                next_t = current.t + 1

                if self.costmap.is_vertex_or_swap_conflict(current.p, next_p, current.t):
                    continue

                move_cost = 0.5 if act == Point(0, 0) else 1.0
                safety_penalty = self.costmap.get_safety_cost(next_p, next_t)
                new_cost = cost_so_far[current] + move_cost + safety_penalty

                next_node = SpaceTimeNode(next_p, next_t)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + next_p.manhattan(goal)
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, next_node))
                    came_from[next_node] = current

        return None


@dataclass(frozen=True)
class RouteAction:
    name: str
    start: Point
    goal: Point



_CANONICAL_PICKLE_MODULE = "agen_kognitif_v2_28"
_PICKLE_COMPAT_CLASSES = (
    RelationType,
    Concept,
    Relation,
    Domain,
    KnowledgeBase,
    Point,
    SpaceTimeNode,
    MovingObstacle,
    SpatioTemporalCostmap,
    SpatioTemporalPlanner,
    RouteAction,
)
for _cls in _PICKLE_COMPAT_CLASSES:
    _cls.__module__ = _CANONICAL_PICKLE_MODULE
del _cls

__all__ = ['RelationType', 'Concept', 'Relation', 'Domain', 'KnowledgeBase', 'Point', 'SpaceTimeNode', 'MovingObstacle', 'SpatioTemporalCostmap', 'SpatioTemporalPlanner', 'RouteAction', 'RELATION_WEIGHTS', 'SAFETY_CLEARANCE', 'COLLISION_MARGIN']
