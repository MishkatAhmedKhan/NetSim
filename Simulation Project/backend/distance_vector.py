"""
Distance Vector routing logic.
Ported from src/simulation/DistanceVector.ts
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

INF = 999999


def initialize_dv_table(
    node_id: str,
    links: List[dict]
) -> Dict[str, dict]:
    """
    Mirrors initializeDVTable() in DistanceVector.ts.
    Returns a RoutingTableDV: { destId -> DVEntry }
    """
    table: Dict[str, dict] = {}

    # Distance to self is 0
    table[node_id] = {
        "destinationId": node_id,
        "nextHopId": node_id,
        "cost": 0
    }

    # Initial distances to immediate neighbors
    for link in links:
        is_down = link.get("status") == "down"
        initial_cost = INF if is_down else link["cost"]

        if link["sourceId"] == node_id:
            neighbor = link["targetId"]
            table[neighbor] = {
                "destinationId": neighbor,
                "nextHopId": None if is_down else neighbor,
                "cost": initial_cost
            }
        elif link["targetId"] == node_id:
            neighbor = link["sourceId"]
            table[neighbor] = {
                "destinationId": neighbor,
                "nextHopId": None if is_down else neighbor,
                "cost": initial_cost
            }

    return table


def process_dv_update_with_logs(
    nodes: List[dict],
    current_table: Dict[str, dict],
    source_node: dict,
    target_node: dict,
    neighbor_payload: Dict[str, dict],
    link_cost: int
) -> Tuple[Dict[str, dict], bool, List[str]]:
    """
    Mirrors processDVUpdateWithLogs() in DistanceVector.ts.
    Returns (new_table, changed, updates_log).
    """
    new_table = {k: dict(v) for k, v in current_table.items()}
    changed = False
    updates_log: List[str] = []

    node_names = {n["id"]: n["name"] for n in nodes}

    def get_dest_name(dest_id: str) -> str:
        return node_names.get(dest_id, dest_id)

    source_id = source_node["id"]
    target_name = target_node["name"]

    for dest_id, neighbor_entry in neighbor_payload.items():
        advertised_cost = neighbor_entry["cost"]
        new_potential_cost = INF if advertised_cost == INF else link_cost + advertised_cost

        our_entry = new_table.get(dest_id) or {
            "destinationId": dest_id,
            "nextHopId": None,
            "cost": INF
        }

        should_update = (
            our_entry["cost"] == INF
            or new_potential_cost < our_entry["cost"]
            or (our_entry["nextHopId"] == source_id and our_entry["cost"] != new_potential_cost)
        )

        if should_update:
            actually_changed = (
                our_entry["cost"] != new_potential_cost
                or (our_entry["nextHopId"] != source_id and new_potential_cost != INF)
            )
            if actually_changed:
                dest_name = get_dest_name(dest_id)
                old_cost_str = "∞" if our_entry["cost"] == INF else str(our_entry["cost"])
                new_cost_str = "∞" if new_potential_cost == INF else str(new_potential_cost)
                adv_str = "∞" if advertised_cost == INF else str(advertised_cost)

                updates_log.append(
                    f"[{target_name}] updated route to {dest_name} via {source_node['name']}. "
                    f"Cost: {old_cost_str} → {new_cost_str} ({link_cost} + {adv_str})"
                )

                new_table[dest_id] = {
                    "destinationId": dest_id,
                    "nextHopId": None if new_potential_cost == INF else source_id,
                    "cost": new_potential_cost
                }
                changed = True

    return new_table, changed, updates_log


def process_dv_update(
    current_table: Dict[str, dict],
    neighbor_id: str,
    neighbor_payload: Dict[str, dict],
    link_cost: int
) -> Tuple[Dict[str, dict], bool]:
    """
    Mirrors processDVUpdate() in DistanceVector.ts.
    Returns (new_table, changed).
    """
    new_table = {k: dict(v) for k, v in current_table.items()}
    changed = False

    for dest_id, neighbor_entry in neighbor_payload.items():
        advertised_cost = neighbor_entry["cost"]
        new_potential_cost = INF if advertised_cost == INF else link_cost + advertised_cost

        our_entry = new_table.get(dest_id) or {
            "destinationId": dest_id,
            "nextHopId": None,
            "cost": INF
        }

        should_update = (
            our_entry["cost"] == INF
            or new_potential_cost < our_entry["cost"]
            or (our_entry["nextHopId"] == neighbor_id and our_entry["cost"] != new_potential_cost)
        )

        if should_update:
            actually_changed = (
                our_entry["cost"] != new_potential_cost
                or (our_entry["nextHopId"] != neighbor_id and new_potential_cost != INF)
            )
            if actually_changed:
                new_table[dest_id] = {
                    "destinationId": dest_id,
                    "nextHopId": None if new_potential_cost == INF else neighbor_id,
                    "cost": new_potential_cost
                }
                changed = True

    return new_table, changed
