"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: Daniel Zhao
Email address: d96zhao@uwaterloo.ca
*******************
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Optional, Tuple

from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

# ---------------------------------------------------------------------------
# Module level state
# ---------------------------------------------------------------------------

# We receive a deep copy of the game state for every package, which means that
# the available bandwidth shown in each state does not include the routing
# decisions that we have already made during the current time step.  To avoid
# over-committing the same connection, we keep track of the reservations that
# we have planned so far and subtract them from the reported bandwidth.
_reserved_step: int = -1
_reserved_usage: DefaultDict[Tuple[str, str], int] = defaultdict(int)

# We also cache some per-time-step metadata so that we do not need to recompute
# it for every package.
_cached_adjacency: Dict[str, Tuple[Tuple[str, float, Optional[int], Optional[int]], ...]] = {}
_cached_connection_lookup: Dict[Tuple[str, str], Tuple[float, Optional[int], Optional[int]]] = {}
_base_node_penalty: Dict[str, float] = {}
_planned_arrivals: DefaultDict[str, int] = defaultdict(int)

# Tunable parameters for the heuristic cost function that guides the search.
CONGESTION_PENALTY = 0.6  # penalty applied per waiting package at a node
ARRIVAL_PENALTY = 0.45    # extra penalty per package we already plan to send
BANDWIDTH_PENALTY = 0.35  # penalty factor for saturated links


def _reset_caches_for_step(state: GameState) -> None:
    """Reset cached data whenever we enter a new time step."""

    global _reserved_step

    if state.current_time_step == _reserved_step:
        return

    _reserved_step = state.current_time_step
    _reserved_usage.clear()
    _cached_adjacency.clear()
    _cached_connection_lookup.clear()
    _planned_arrivals.clear()

    # Build adjacency lists and lookup tables for the current state.  We store
    # only the information that is relevant for routing decisions so that we
    # do not need to keep references to the mutable connection objects.
    adjacency: DefaultDict[str, list] = defaultdict(list)
    connection_lookup: Dict[Tuple[str, str], Tuple[float, Optional[int], Optional[int]]] = {}

    for connection in state.connections:
        adjacency[connection.from_fc].append(
            (
                connection.to_fc,
                connection.weight,
                connection.bandwidth,
                connection.available_bandwidth,
            )
        )
        connection_lookup[(connection.from_fc, connection.to_fc)] = (
            connection.weight,
            connection.bandwidth,
            connection.available_bandwidth,
        )

    _cached_adjacency.update({node: tuple(edges) for node, edges in adjacency.items()})
    _cached_connection_lookup.update(connection_lookup)

    # Estimate the congestion at each fulfillment center.  Packages that are in
    # transit will not cause additional queuing, so we only consider those that
    # are waiting at a node.
    waiting_counts = Counter(pkg.current_fc for pkg in state.active_packages if not pkg.in_transit)
    _base_node_penalty.clear()
    for fc, count in waiting_counts.items():
        # The penalty grows linearly with the number of waiting packages.
        _base_node_penalty[fc] = count * CONGESTION_PENALTY


def _effective_bandwidth(from_fc: str, to_fc: str) -> Tuple[Optional[int], Optional[int]]:
    """Return the (bandwidth, available) tuple after planned reservations."""

    bandwidth: Optional[int]
    available: Optional[int]
    weight_bandwidth_available = _cached_connection_lookup.get((from_fc, to_fc))

    if weight_bandwidth_available is None:
        return None, None

    _, bandwidth, available = weight_bandwidth_available

    if bandwidth is None or available is None:
        return bandwidth, available

    effective_available = available - _reserved_usage[(from_fc, to_fc)]
    return bandwidth, max(effective_available, 0)


def _estimate_node_penalty(node: str) -> float:
    """Estimate how congested a node is, including planned arrivals."""

    base_penalty = _base_node_penalty.get(node, 0.0)
    arrival_penalty = _planned_arrivals[node] * ARRIVAL_PENALTY
    return base_penalty + arrival_penalty


def _dijkstra_next_hop(
    start: str,
    goal: str,
    adjacency: Dict[str, Tuple[Tuple[str, float, Optional[int], Optional[int]], ...]],
) -> Optional[str]:
    """Compute the first hop of the cheapest path using a weighted Dijkstra search."""

    if start == goal:
        return None

    distances: Dict[str, float] = {start: 0.0}
    first_hop: Dict[str, Optional[str]] = {start: None}
    heap: list[Tuple[float, str]] = [(0.0, start)]

    while heap:
        current_cost, node = heapq.heappop(heap)

        if node == goal:
            return first_hop[node]

        # Skip stale entries from the priority queue.
        if current_cost > distances.get(node, float("inf")):
            continue

        for to_fc, weight, bandwidth, available in adjacency.get(node, ()):  # type: ignore[arg-type]
            if weight < 0:
                # Negative weights are not expected in the input, but guard just in case.
                continue

            effective_bandwidth = available
            if bandwidth is not None and available is not None:
                effective_bandwidth = available - _reserved_usage[(node, to_fc)]
                if effective_bandwidth <= 0:
                    continue

                saturation_ratio = 1.0 - (effective_bandwidth / bandwidth)
            else:
                saturation_ratio = 0.0

            node_penalty = _estimate_node_penalty(to_fc)
            link_penalty = 1.0 + saturation_ratio * BANDWIDTH_PENALTY

            next_cost = current_cost + weight * link_penalty + node_penalty

            if next_cost < distances.get(to_fc, float("inf")):
                distances[to_fc] = next_cost
                first_hop[to_fc] = first_hop[node] if first_hop[node] is not None else to_fc
                heapq.heappush(heap, (next_cost, to_fc))
            elif next_cost == distances.get(to_fc):
                # Deterministic tie-breaking: prefer lexicographically smaller hops.
                candidate_first = first_hop[node] if first_hop[node] is not None else to_fc
                stored_first = first_hop.get(to_fc)
                if stored_first is None or candidate_first < stored_first:
                    first_hop[to_fc] = candidate_first

    return None


def _pick_direct_if_available(current_fc: str, destination_fc: str) -> Optional[str]:
    """Return the destination if a direct connection with spare bandwidth exists."""

    if current_fc == destination_fc:
        return None

    connection_info = _cached_connection_lookup.get((current_fc, destination_fc))
    if connection_info is None:
        return None

    _, bandwidth, available = connection_info
    if bandwidth is None:
        return destination_fc

    if available is None:
        return destination_fc

    if available - _reserved_usage[(current_fc, destination_fc)] > 0:
        return destination_fc

    return None


def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Determine the next FC to route a package to.

    This is the function that students will implement. The game engine will call
    this function for each package at each time step to determine where to route it.

    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package

    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """

    # Keep caches in sync with the simulation time.
    _reset_caches_for_step(state)

    # If the package has already reached its destination (or there is nowhere to go), stay put.
    if package.current_fc == package.destination_fc:
        return None

    # Prefer a direct connection to the destination if there is free capacity – this minimises
    # latency for the most common case.
    direct_hop = _pick_direct_if_available(package.current_fc, package.destination_fc)
    if direct_hop is not None:
        _reserved_usage[(package.current_fc, direct_hop)] += 1
        _planned_arrivals[direct_hop] += 1
        return direct_hop

    # Otherwise run a Dijkstra search over the current network snapshot.
    next_hop = _dijkstra_next_hop(package.current_fc, package.destination_fc, _cached_adjacency)

    if next_hop is None:
        return None

    # Verify that the selected hop still has enough spare capacity after considering the
    # reservations we have already made in this time step.  If not, stay put.
    bandwidth, available = _effective_bandwidth(package.current_fc, next_hop)
    if bandwidth is not None and available is not None and available <= 0:
        return None

    _reserved_usage[(package.current_fc, next_hop)] += 1
    _planned_arrivals[next_hop] += 1
    return next_hop
