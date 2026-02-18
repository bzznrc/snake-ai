"""Square-grid generation helpers."""

from __future__ import annotations

import random
from typing import Callable, TypeVar

T = TypeVar("T")


def _grow_connected_random_walk_shape(
    start: T,
    min_sections: int,
    max_sections: int,
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
    rng: random.Random | None = None,
) -> list[T]:
    rng = rng or random
    if min_sections < 1:
        raise ValueError("min_sections must be >= 1")
    if max_sections < min_sections:
        raise ValueError("max_sections must be >= min_sections")

    target_sections = rng.randint(int(min_sections), int(max_sections))
    shape = [start]
    current = start

    for _ in range(target_sections - 1):
        candidates = list(neighbor_candidates_fn(current))
        rng.shuffle(candidates)

        extended = False
        for candidate in candidates:
            if is_candidate_valid_fn(candidate, shape):
                shape.append(candidate)
                current = candidate
                extended = True
                break
        if not extended:
            break

    return shape


def spawn_connected_random_walk_shapes(
    shape_count: int,
    min_sections: int,
    max_sections: int,
    sample_start_fn: Callable[[], T | None],
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
    rng: random.Random | None = None,
) -> list[list[T]]:
    """Build multiple connected random-walk shapes from sampled starts."""

    rng = rng or random
    shapes: list[list[T]] = []
    for _ in range(int(shape_count)):
        start = sample_start_fn()
        if start is None:
            continue
        shape = _grow_connected_random_walk_shape(
            start=start,
            min_sections=min_sections,
            max_sections=max_sections,
            neighbor_candidates_fn=neighbor_candidates_fn,
            is_candidate_valid_fn=is_candidate_valid_fn,
            rng=rng,
        )
        if shape:
            shapes.append(shape)
    return shapes
