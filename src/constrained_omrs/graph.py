from __future__ import annotations

from collections.abc import Iterable

import numpy as np

Edge = tuple[int, int]


def canonical_edge(i: int, j: int) -> Edge:
    if i == j:
        raise ValueError("Self-edges are not allowed.")
    return (i, j) if i < j else (j, i)


def incidence_matrix(num_robots: int, edges: Iterable[Edge]) -> np.ndarray:
    """Return an oriented node-edge incidence matrix.

    Every canonical edge ``(i,j)`` is oriented ``i -> j``. The orientation is
    arbitrary, provided the same convention is used for the associated relative
    displacement and barrier gradient.
    """
    edge_list = [canonical_edge(int(i), int(j)) for i, j in edges]
    E = np.zeros((num_robots, len(edge_list)), dtype=float)
    for k, (i, j) in enumerate(edge_list):
        E[i, k] = 1.0
        E[j, k] = -1.0
    return E


def active_pairs(active: np.ndarray) -> list[Edge]:
    ids = np.flatnonzero(np.asarray(active, dtype=bool))
    return [
        (int(ids[a]), int(ids[b]))
        for a in range(len(ids))
        for b in range(a + 1, len(ids))
    ]


def connected_components(vertices: Iterable[int], edges: Iterable[Edge]) -> list[set[int]]:
    """Return connected components of an undirected graph."""
    vertices = {int(v) for v in vertices}
    adjacency = {v: set() for v in vertices}
    for raw_i, raw_j in edges:
        i, j = canonical_edge(int(raw_i), int(raw_j))
        if i in vertices and j in vertices:
            adjacency[i].add(j)
            adjacency[j].add(i)

    components: list[set[int]] = []
    unvisited = set(vertices)
    while unvisited:
        root = min(unvisited)
        component = {root}
        stack = [root]
        unvisited.remove(root)
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def is_connected(vertices: Iterable[int], edges: Iterable[Edge]) -> bool:
    vertices = list(vertices)
    if len(vertices) <= 1:
        return True
    return len(connected_components(vertices, edges)) == 1
