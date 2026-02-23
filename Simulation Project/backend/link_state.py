"""
Link State routing logic (Dijkstra's algorithm).
Ported from src/simulation/LinkState.ts
"""
from __future__ import annotations
from typing import Dict, List, Optional

from distance_vector import INF


def generate_lsp(node_id: str, seq_num: int, links: List[dict]) -> dict:
    """
    Mirrors generateLSP() in LinkState.ts.
    Returns an LSP dict: { sourceId, sequenceNumber, links: [{targetId, cost}] }
    """
    local_links = []
    for link in links:
        if (link["sourceId"] == node_id or link["targetId"] == node_id) \
                and link.get("status") != "down" \
                and abs(link["cost"]) != INF:
            target_id = link["targetId"] if link["sourceId"] == node_id else link["sourceId"]
            local_links.append({"targetId": target_id, "cost": link["cost"]})

    return {
        "sourceId": node_id,
        "sequenceNumber": seq_num,
        "links": local_links
    }


def calculate_dijkstra(source_id: str, lsdb: Dict[str, dict]) -> Dict[str, dict]:
    """
    Mirrors calculateDijkstra() in LinkState.ts.
    Returns RoutingTableDV: { destId -> DVEntry }
    """
    distances: Dict[str, int] = {}
    previous: Dict[str, Optional[str]] = {}
    unvisited: set = set()

    # Initialize
    for node_id in lsdb:
        distances[node_id] = 0 if node_id == source_id else INF
        previous[node_id] = None
        unvisited.add(node_id)

    # Edge case: add source if not in LSDB
    if source_id not in distances:
        distances[source_id] = 0
        unvisited.add(source_id)

    while unvisited:
        # Find node with minimum distance
        u = None
        current_min = INF
        for node in unvisited:
            if distances.get(node, INF) < current_min:
                current_min = distances[node]
                u = node

        if u is None or distances.get(u, INF) == INF:
            break  # Remaining nodes are unreachable

        unvisited.remove(u)

        u_lsp = lsdb.get(u)
        if u_lsp:
            for neighbor in u_lsp.get("links", []):
                v = neighbor["targetId"]
                if v not in unvisited:
                    continue
                alt_cost = distances[u] + neighbor["cost"]
                if alt_cost < distances.get(v, INF):
                    distances[v] = alt_cost
                    previous[v] = u

    # Construct routing table — resolve immediate next-hop
    routing_table: Dict[str, dict] = {}

    for dest_id, dist in distances.items():
        if dist == INF:
            routing_table[dest_id] = {
                "destinationId": dest_id,
                "nextHopId": None,
                "cost": INF
            }
            continue

        # Traverse previous map backward to find first hop
        curr = dest_id
        first_hop = curr
        while previous.get(curr) is not None and previous[curr] != source_id:
            curr = previous[curr]
            first_hop = curr

        routing_table[dest_id] = {
            "destinationId": dest_id,
            "nextHopId": dest_id if dest_id == source_id else first_hop,
            "cost": dist
        }

    return routing_table
