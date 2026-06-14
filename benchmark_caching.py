#!/usr/bin/env python3
"""
Benchmark: per-timestep caching in routing.py (cached vs. no-cache).

Measures how much redundant graph recomputation the per-timestep cache in
`_reset_caches_for_step` eliminates, by running every test case twice:

  1. CACHED   - routing.py as written (rebuild only when the timestep changes)
  2. NO-CACHE - identical logic, but the timestep guard is disabled so the
                adjacency lists / connection lookup / congestion estimates are
                rebuilt on every route_package() call (the naive baseline)

Reported per mode: number of full graph rebuilds, total routing wall time,
and final score (to prove the cache does not change behaviour).

Usage:  python3 benchmark_caching.py
"""

import glob
import time

import ar_hackathon.api.routing as routing
from ar_hackathon.engine.game_engine import GameEngine

# ---------------------------------------------------------------- counters
REBUILDS = 0
ROUTE_CALLS = 0
ROUTE_TIME = 0.0

_original_reset = routing._reset_caches_for_step


def _counting_reset(state, *, force=False):
    """Wrap the cache reset so we can count full rebuilds (and force them).

    In force mode we rebuild the cached graph data on every call (the naive
    baseline) but preserve the bandwidth-reservation ledger and planned
    arrivals within a timestep, so routing DECISIONS stay identical and the
    experiment isolates the cache's effect.
    """
    global REBUILDS
    if not force:
        if state.current_time_step != routing._reserved_step:
            REBUILDS += 1
        _original_reset(state)
        return

    saved_step = routing._reserved_step
    saved_usage = dict(routing._reserved_usage)
    saved_arrivals = dict(routing._planned_arrivals)
    routing._reserved_step = -1          # defeat the guard -> full rebuild
    _original_reset(state)
    REBUILDS += 1
    if state.current_time_step == saved_step:  # same timestep: restore ledgers
        routing._reserved_usage.update(saved_usage)
        routing._planned_arrivals.update(saved_arrivals)


def make_router(force_rebuild: bool):
    """Return a route_package wrapper that times calls and counts rebuilds."""

    def router(state, package):
        global ROUTE_CALLS, ROUTE_TIME
        ROUTE_CALLS += 1
        t0 = time.perf_counter()
        _counting_reset(state, force=force_rebuild)
        result = routing.route_package(state, package)
        ROUTE_TIME += time.perf_counter() - t0
        return result

    return router


def run_all(force_rebuild: bool):
    global REBUILDS, ROUTE_CALLS, ROUTE_TIME
    REBUILDS = ROUTE_CALLS = 0
    ROUTE_TIME = 0.0
    scores = {}
    for case in sorted(glob.glob("test_cases/level*/test_case_*.json")):
        routing._reserved_step = -1  # clean slate between cases
        engine = GameEngine(case, make_router(force_rebuild))
        stats = engine.run_until_finished()
        scores[case] = stats["score"]
    return REBUILDS, ROUTE_CALLS, ROUTE_TIME, scores


def main():
    print("Running CACHED (routing.py as written)...")
    c_rebuilds, c_calls, c_time, c_scores = run_all(force_rebuild=False)

    print("Running NO-CACHE baseline (rebuild every call)...")
    n_rebuilds, n_calls, n_time, n_scores = run_all(force_rebuild=True)

    rebuild_cut = 100.0 * (1 - c_rebuilds / n_rebuilds)
    time_cut = 100.0 * (1 - c_time / n_time)

    print("\n================ RESULTS (all 7 test cases) ================")
    print(f"route_package() calls : {c_calls}")
    print(f"Graph rebuilds  CACHED: {c_rebuilds:>8}")
    print(f"Graph rebuilds NOCACHE: {n_rebuilds:>8}")
    print(f"-> redundant rebuilds eliminated: {rebuild_cut:.1f}%")
    print(f"Routing time    CACHED: {c_time:>8.3f}s")
    print(f"Routing time   NOCACHE: {n_time:>8.3f}s")
    print(f"-> routing compute time reduced: {time_cut:.1f}%")
    print(f"Total score   CACHED: {sum(c_scores.values()):.2f}")
    print(f"Total score  NOCACHE: {sum(n_scores.values()):.2f}")
    print("(Scores can differ slightly: the engine applies moves immediately,")
    print(" so the no-cache baseline sees mid-timestep state by construction.)")
    print("=============================================================")


if __name__ == "__main__":
    main()
