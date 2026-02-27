"""
ARP (Address Resolution Protocol) simulation logic.
Ported from src/store/useArpStore.ts and src/simulation/ArpModels.ts
"""
from __future__ import annotations
import math
import random
import uuid
from typing import Dict, List, Optional, Tuple


# ─── Constants ───────────────────────────────────────────────────────────────

BROADCAST_MAC = "FF:FF:FF:FF:FF:FF"

HOST_NAMES = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
SWITCH_NAMES = ['SW1', 'SW2', 'SW3', 'SW4', 'SW5', 'SW6']
ROUTER_NAMES = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']


# ─── MAC Address Generator ────────────────────────────────────────────────────

def generate_mac() -> str:
    """Generates a random MAC address."""
    parts = [f"{random.randint(0, 255):02X}" for _ in range(6)]
    return ":".join(parts)


# ─── IP / Subnet Helpers ──────────────────────────────────────────────────────

def ip_to_number(ip: str) -> int:
    """Converts an IP string (e.g. '192.168.1.1') to an unsigned 32-bit integer."""
    parts = list(map(int, ip.split('.')))
    return ((parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]) & 0xFFFFFFFF


def network_of(ip: str, mask: str) -> int:
    """Returns the network address as an integer (IP AND mask)."""
    return ip_to_number(ip) & ip_to_number(mask)


def same_subnet(ip1: str, ip2: str, mask: str) -> bool:
    """Returns True if ip1 and ip2 are on the same subnet given mask."""
    return network_of(ip1, mask) == network_of(ip2, mask)


# ─── BFS Helpers ─────────────────────────────────────────────────────────────

def bfs_path(start_id: str, end_id: str, links: List[dict]) -> List[str]:
    """BFS shortest path between two nodes in the graph."""
    visited = {start_id}
    queue = [[start_id]]
    while queue:
        path = queue.pop(0)
        current = path[-1]
        if current == end_id:
            return path
        for link in links:
            neighbor = None
            if link["sourceId"] == current:
                neighbor = link["targetId"]
            elif link["targetId"] == current:
                neighbor = link["sourceId"]
            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []


def find_link_between(a: str, b: str, links: List[dict]) -> Optional[dict]:
    """Returns the link between nodes a and b, or None if not found."""
    for link in links:
        if (link["sourceId"] == a and link["targetId"] == b) or \
           (link["sourceId"] == b and link["targetId"] == a):
            return link
    return None


def get_neighbor_ids(node_id: str, links: List[dict]) -> List[dict]:
    """Returns list of {neighborId, linkId} dicts for all neighbors of node_id."""
    result = []
    for link in links:
        if link["sourceId"] == node_id:
            result.append({"neighborId": link["targetId"], "linkId": link["id"]})
        elif link["targetId"] == node_id:
            result.append({"neighborId": link["sourceId"], "linkId": link["id"]})
    return result


# ─── Probability Engine ───────────────────────────────────────────────────────

def check_probability(mode: str, p: float) -> bool:
    """Evaluates a probability check using the specified distribution mode."""
    from probability import check_probability as _check
    return _check(mode, p)


# ─── Node Factory ─────────────────────────────────────────────────────────────

def create_arp_node(node_type: str, x: float, y: float, existing_nodes: List[dict]) -> dict:
    """
    Creates a new ARP node dict.
    Mirrors useArpStore.addNode() in useArpStore.ts.
    """
    existing_of_type = [n for n in existing_nodes if n["type"] == node_type]
    name_list = HOST_NAMES if node_type == "host" else (SWITCH_NAMES if node_type == "switch" else ROUTER_NAMES)
    name = name_list[len(existing_of_type)] if len(existing_of_type) < len(name_list) \
        else f"{node_type[:1].upper()}{len(existing_of_type) + 1}"

    host_count = sum(1 for n in existing_nodes if n["type"] == "host")
    router_count = sum(1 for n in existing_nodes if n["type"] == "router")

    ip = ""
    if node_type == "host":
        ip = f"192.168.1.{10 + host_count}"
    elif node_type == "router":
        ip = f"192.168.1.{1 + router_count}"

    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "type": node_type,
        "position": {"x": x, "y": y},
        "macAddress": generate_mac(),
        "ipAddress": ip,
        "subnetMask": "255.255.255.0",
        "gateway": "192.168.1.1",
        "interfaces": [],
    }


# ─── ARP Core Logic ───────────────────────────────────────────────────────────

def create_broadcast_packets(
    packet_template: dict,
    from_node_id: str,
    links: List[dict],
    exclude_link_id: Optional[str] = None,
    global_loss_rate: float = 0.0,
    dist_mode: str = "Uniform",
) -> List[dict]:
    """
    Creates ARP broadcast packets sent to all neighbors (except excluded link).
    Mirrors createBroadcastPackets() in useArpStore.ts.
    """
    neighbors = get_neighbor_ids(from_node_id, links)
    packets = []
    for entry in neighbors:
        link_id = entry["linkId"]
        neighbor_id = entry["neighborId"]
        if link_id == exclude_link_id:
            continue
        link = next((l for l in links if l["id"] == link_id), None)
        if not link or link.get("status") == "down":
            continue
        link_loss = link.get("packetLossRate", 0)
        effective_loss = max(link_loss, global_loss_rate)
        will_drop = check_probability(dist_mode, effective_loss)
        packets.append({
            "id": str(uuid.uuid4()),
            "type": packet_template["type"],
            "senderMAC": packet_template["senderMAC"],
            "senderIP": packet_template["senderIP"],
            "targetMAC": packet_template["targetMAC"],
            "targetIP": packet_template["targetIP"],
            "linkId": link_id,
            "sourceNodeId": from_node_id,
            "targetNodeId": neighbor_id,
            "progress": 0,
            "status": "in-transit",
            "willDrop": will_drop,
            "originNodeId": packet_template["originNodeId"],
            "isBroadcast": True,
            "ultimateDestIP": packet_template.get("ultimateDestIP"),
        })
    return packets


def initiate_arp_request(
    source_node_id: str,
    target_ip: str,
    nodes: List[dict],
    links: List[dict],
    arp_caches: Dict[str, List[dict]],
    tick_counter: int,
    packet_loss_rate: float = 0.0,
    packet_dist: str = "Uniform",
) -> Tuple[List[dict], List[dict]]:
    """
    Initiates an ARP request from source_node_id to resolve target_ip.
    Mirrors sendArpRequest() in useArpStore.ts.
    Returns (new_packets, new_logs).
    """
    source_node = next((n for n in nodes if n["id"] == source_node_id), None)
    if not source_node:
        return [], []

    # Subnet check
    src_network = network_of(source_node["ipAddress"], source_node["subnetMask"])
    dst_network = network_of(target_ip, source_node["subnetMask"])
    is_local = src_network == dst_network

    arp_for_ip = target_ip
    ultimate_dest_ip = None

    if not is_local:
        if not source_node.get("gateway"):
            return [], [{
                "id": str(uuid.uuid4()),
                "tick": tick_counter,
                "message": f"{source_node['name']}: Destination {target_ip} is remote but NO gateway configured!",
                "type": "error"
            }]
        arp_for_ip = source_node["gateway"]
        ultimate_dest_ip = target_ip

    # Check ARP cache
    cache = arp_caches.get(source_node_id, [])
    cached = next((e for e in cache if e["ip"] == arp_for_ip and e["ttl"] > 0), None)
    if cached:
        log = {
            "id": str(uuid.uuid4()),
            "tick": tick_counter,
            "message": f"{source_node['name']}: ARP cache HIT for {arp_for_ip} → {cached['mac']}",
            "type": "info"
        }
        return [], [log]

    # ARP cache MISS → broadcast REQUEST
    packet_template = {
        "type": "REQUEST",
        "senderMAC": source_node["macAddress"],
        "senderIP": source_node["ipAddress"],
        "targetMAC": BROADCAST_MAC,
        "targetIP": arp_for_ip,
        "originNodeId": source_node_id,
        "ultimateDestIP": ultimate_dest_ip,
    }
    packets = create_broadcast_packets(
        packet_template, source_node_id, links,
        global_loss_rate=packet_loss_rate, dist_mode=packet_dist
    )
    logs = [{
        "id": str(uuid.uuid4()),
        "tick": tick_counter,
        "message": f"{source_node['name']}: ARP Request — Who has {arp_for_ip}? Tell {source_node['ipAddress']} (broadcast)",
        "type": "info"
    }]
    return packets, logs


def send_gratuitous_arp(
    node_id: str,
    nodes: List[dict],
    links: List[dict],
    tick_counter: int,
    packet_loss_rate: float = 0.0,
    packet_dist: str = "Uniform",
) -> Tuple[List[dict], List[dict]]:
    """
    Sends a Gratuitous ARP from the node.
    Mirrors sendGratuitousArp() in useArpStore.ts.
    Returns (new_packets, new_logs).
    """
    node = next((n for n in nodes if n["id"] == node_id), None)
    if not node or not node.get("ipAddress"):
        return [], []

    packet_template = {
        "type": "GRATUITOUS",
        "senderMAC": node["macAddress"],
        "senderIP": node["ipAddress"],
        "targetMAC": BROADCAST_MAC,
        "targetIP": node["ipAddress"],
        "originNodeId": node_id,
        "ultimateDestIP": None,
    }
    packets = create_broadcast_packets(
        packet_template, node_id, links,
        global_loss_rate=packet_loss_rate, dist_mode=packet_dist
    )
    logs = [{
        "id": str(uuid.uuid4()),
        "tick": tick_counter,
        "message": f"{node['name']}: Gratuitous ARP — {node['ipAddress']} is at {node['macAddress']}",
        "type": "info"
    }]
    return packets, logs


def update_arp_cache(
    node_id: str,
    ip: str,
    mac: str,
    arp_caches: Dict[str, List[dict]],
    tick_counter: int,
    cache_ttl: int = 300,
) -> None:
    """
    Updates the ARP cache for a node with a new IP→MAC mapping.
    Mirrors updateCache() helper in useArpStore.ts simulationTick().
    Mutates arp_caches in place.
    """
    cache = list(arp_caches.get(node_id, []))
    existing_idx = next((i for i, e in enumerate(cache) if e["ip"] == ip), -1)
    if existing_idx >= 0:
        cache[existing_idx] = {
            **cache[existing_idx],
            "mac": mac,
            "createdAtTick": tick_counter,
            "ttl": cache_ttl,
        }
    else:
        cache.append({
            "ip": ip,
            "mac": mac,
            "type": "dynamic",
            "createdAtTick": tick_counter,
            "ttl": cache_ttl,
        })
    arp_caches[node_id] = cache


def advance_arp_tick(
    nodes: List[dict],
    links: List[dict],
    active_packets: List[dict],
    arp_caches: Dict[str, List[dict]],
    switch_mac_tables: Dict[str, List[dict]],
    tick_counter: int,
    cache_ttl: int = 300,
    packet_loss_rate: float = 0.0,
    link_failure_rate: float = 0.0,
    packet_dist: str = "Uniform",
    link_fail_dist: str = "Uniform",
    link_recover_dist: str = "Uniform",
) -> dict:
    """
    Advances the ARP simulation by one tick.
    Mirrors simulationTick() in useArpStore.ts.
    Returns dict with updated state.
    """
    new_packets: List[dict] = []
    new_logs: List[dict] = []
    updated_caches = {k: list(v) for k, v in arp_caches.items()}
    updated_switch_tables = {k: list(v) for k, v in switch_mac_tables.items()}
    surviving_packets: List[dict] = []

    # Link weather system
    updated_links = list(links)
    if link_failure_rate > 0:
        for i, link in enumerate(updated_links):
            link = dict(link)
            if link.get("status") == "down":
                dt = (link.get("downTicks") or 0) + 1
                recover_prob = min(1.0, 0.4 + dt * 0.1)
                if check_probability(link_recover_dist, recover_prob):
                    link["status"] = "up"
                    link["downTicks"] = 0
                else:
                    link["downTicks"] = dt
            else:
                if check_probability(link_fail_dist, link_failure_rate):
                    link["status"] = "down"
                    link["downTicks"] = 1
            updated_links[i] = link

    node_map = {n["id"]: n for n in nodes}

    for pkt in active_packets:
        pkt = dict(pkt)
        if pkt["status"] != "in-transit":
            # Keep briefly-visible dropped packets
            if pkt["status"] == "dropped":
                hold = (pkt.get("dropHoldTicks") or 0) - 1
                if hold > 0:
                    pkt["dropHoldTicks"] = hold
                    surviving_packets.append(pkt)
            continue

        link = next((l for l in updated_links if l["id"] == pkt["linkId"]), None)
        if not link:
            continue

        # If link went down mid-transit, drop
        if link.get("status") == "down":
            pkt["status"] = "dropped"
            pkt["dropHoldTicks"] = 15
            surviving_packets.append(pkt)
            new_logs.append({
                "id": str(uuid.uuid4()), "tick": tick_counter,
                "message": f"DROPPED: {pkt['type']} packet — link is DOWN",
                "type": "error"
            })
            continue

        speed = 1.0 / max(1, link.get("latency", 10))
        new_progress = min(1.0, pkt["progress"] + speed)

        if new_progress >= 1.0:
            if pkt.get("willDrop"):
                pkt["progress"] = 1.0
                pkt["status"] = "dropped"
                pkt["dropHoldTicks"] = 15
                surviving_packets.append(pkt)
                target_name = node_map.get(pkt["targetNodeId"], {}).get("name", "?")
                new_logs.append({
                    "id": str(uuid.uuid4()), "tick": tick_counter,
                    "message": f"DROPPED: ARP {pkt['type']} on link to {target_name}",
                    "type": "error"
                })
                continue

            target_node = node_map.get(pkt["targetNodeId"])
            if not target_node:
                continue

            pkt["progress"] = 1.0
            pkt["status"] = "delivered"

            # Learn sender MAC→IP in ARP cache if target is host/router
            if target_node["type"] in ("host", "router"):
                if pkt.get("senderIP") and pkt.get("senderMAC") and pkt["senderMAC"] != BROADCAST_MAC:
                    update_arp_cache(target_node["id"], pkt["senderIP"], pkt["senderMAC"],
                                     updated_caches, tick_counter, cache_ttl)

                # ARP REQUEST: if this node owns the targetIP, send REPLY
                if pkt["type"] == "REQUEST" and target_node.get("ipAddress") == pkt["targetIP"]:
                    reply_path = bfs_path(target_node["id"], pkt["originNodeId"], updated_links)
                    if len(reply_path) >= 2:
                        next_hop = reply_path[1]
                        reply_link = find_link_between(target_node["id"], next_hop, updated_links)
                        if reply_link:
                            new_packets.append({
                                "id": str(uuid.uuid4()),
                                "type": "REPLY",
                                "senderMAC": target_node["macAddress"],
                                "senderIP": target_node["ipAddress"],
                                "targetMAC": pkt["senderMAC"],
                                "targetIP": pkt["senderIP"],
                                "linkId": reply_link["id"],
                                "sourceNodeId": target_node["id"],
                                "targetNodeId": next_hop,
                                "progress": 0,
                                "status": "in-transit",
                                "willDrop": check_probability(packet_dist, packet_loss_rate),
                                "originNodeId": target_node["id"],
                                "isBroadcast": False,
                                "ultimateDestIP": None,
                            })
                            new_logs.append({
                                "id": str(uuid.uuid4()), "tick": tick_counter,
                                "message": f"{target_node['name']}: I have {pkt['targetIP']}! Sending ARP Reply to {pkt['senderIP']}",
                                "type": "success"
                            })
        else:
            pkt["progress"] = new_progress
            surviving_packets.append(pkt)

    # Decrement TTLs in ARP caches
    for node_id, cache in updated_caches.items():
        for entry in cache:
            if entry["type"] == "dynamic" and entry["ttl"] != float("inf"):
                entry["ttl"] = max(0, entry["ttl"] - 1)

    return {
        "links": updated_links,
        "activePackets": surviving_packets + new_packets,
        "arpCaches": updated_caches,
        "switchMacTables": updated_switch_tables,
        "logs": new_logs,
        "tickCounter": tick_counter + 1,
    }


# ─── ARP Cache Inspection Utilities ─────────────────────────────────────────

def get_cache_summary(arp_caches: Dict[str, List[dict]], nodes: List[dict]) -> List[dict]:
    """
    Returns a human-readable summary of all ARP caches across all nodes.
    Useful for debugging and UI display.
    """
    node_map = {n["id"]: n for n in nodes}
    summary = []
    for node_id, cache in arp_caches.items():
        node = node_map.get(node_id)
        node_name = node["name"] if node else node_id
        for entry in cache:
            summary.append({
                "nodeId": node_id,
                "nodeName": node_name,
                "ip": entry["ip"],
                "mac": entry["mac"],
                "ttl": entry["ttl"],
                "type": entry.get("type", "dynamic"),
                "createdAtTick": entry.get("createdAtTick", 0),
            })
    return summary


def get_stale_entries(
    arp_caches: Dict[str, List[dict]],
    nodes: List[dict],
    ttl_threshold: int = 10,
) -> List[dict]:
    """
    Returns all ARP cache entries with TTL below the given threshold.
    Used to warn about entries that are about to expire.
    """
    all_entries = get_cache_summary(arp_caches, nodes)
    return [e for e in all_entries if isinstance(e["ttl"], (int, float)) and e["ttl"] <= ttl_threshold]


def count_cache_hits_per_node(arp_caches: Dict[str, List[dict]]) -> Dict[str, int]:
    """Returns a dict mapping node_id -> number of active (ttl > 0) cache entries."""
    return {
        node_id: sum(1 for e in cache if e.get("ttl", 0) > 0)
        for node_id, cache in arp_caches.items()
    }


def clear_expired_entries(arp_caches: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """
    Removes all cache entries with TTL == 0.
    Returns the cleaned caches dict.
    """
    return {
        node_id: [e for e in cache if e.get("ttl", 0) > 0]
        for node_id, cache in arp_caches.items()
    }


def find_ip_in_caches(
    target_ip: str,
    arp_caches: Dict[str, List[dict]],
) -> List[dict]:
    """
    Searches all ARP caches for entries matching the given IP.
    Returns a list of {nodeId, mac, ttl} dicts.
    """
    results = []
    for node_id, cache in arp_caches.items():
        for entry in cache:
            if entry["ip"] == target_ip:
                results.append({
                    "nodeId": node_id,
                    "mac": entry["mac"],
                    "ttl": entry["ttl"],
                })
    return results


# ─── ARP Conflict Detection ───────────────────────────────────────────────────

def detect_ip_conflicts(nodes: List[dict]) -> List[dict]:
    """Detects nodes with duplicate IP addresses. Returns list of conflict reports."""
    ip_map: Dict[str, List[dict]] = {}
    for node in nodes:
        ip = node.get("ipAddress", "")
        if not ip:
            continue
        if ip not in ip_map:
            ip_map[ip] = []
        ip_map[ip].append(node)
    conflicts = []
    for ip, conflicting in ip_map.items():
        if len(conflicting) > 1:
            conflicts.append({
                "ip": ip,
                "nodeIds": [n["id"] for n in conflicting],
                "nodeNames": [n["name"] for n in conflicting],
            })
    return conflicts


def detect_mac_conflicts(nodes: List[dict]) -> List[dict]:
    """Detects nodes with duplicate MAC addresses. Returns list of conflict reports."""
    mac_map: Dict[str, List[dict]] = {}
    for node in nodes:
        mac = node.get("macAddress", "")
        if not mac or mac == BROADCAST_MAC:
            continue
        if mac not in mac_map:
            mac_map[mac] = []
        mac_map[mac].append(node)
    conflicts = []
    for mac, conflicting in mac_map.items():
        if len(conflicting) > 1:
            conflicts.append({
                "mac": mac,
                "nodeIds": [n["id"] for n in conflicting],
                "nodeNames": [n["name"] for n in conflicting],
            })
    return conflicts


def detect_cache_inconsistency(
    arp_caches: Dict[str, List[dict]],
    nodes: List[dict],
) -> List[dict]:
    """
    Checks whether any ARP cache has a stale MAC mapping that no longer
    matches the actual MAC of the node with that IP.
    """
    ip_to_mac = {n["ipAddress"]: n["macAddress"] for n in nodes if n.get("ipAddress")}
    issues = []
    for node_id, cache in arp_caches.items():
        for entry in cache:
            actual_mac = ip_to_mac.get(entry["ip"])
            if actual_mac and entry["mac"] != actual_mac:
                issues.append({
                    "nodeId": node_id,
                    "cachedIP": entry["ip"],
                    "cachedMAC": entry["mac"],
                    "actualMAC": actual_mac,
                })
    return issues


# ─── Topology Inspection Helpers ──────────────────────────────────────────────

def get_reachable_nodes(
    source_id: str,
    links: List[dict],
    nodes: List[dict],
) -> List[dict]:
    """Returns all nodes reachable from source_id using BFS over active (non-down) links."""
    active_links = [l for l in links if l.get("status", "up") != "down"]
    visited = {source_id}
    queue = [source_id]
    reachable_ids = []
    while queue:
        current = queue.pop(0)
        for link in active_links:
            neighbor = None
            if link["sourceId"] == current:
                neighbor = link["targetId"]
            elif link["targetId"] == current:
                neighbor = link["sourceId"]
            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                reachable_ids.append(neighbor)
                queue.append(neighbor)
    node_map = {n["id"]: n for n in nodes}
    return [node_map[nid] for nid in reachable_ids if nid in node_map]


def is_fully_connected(nodes: List[dict], links: List[dict]) -> bool:
    """Returns True if all nodes are reachable from the first node."""
    if not nodes:
        return True
    reachable = get_reachable_nodes(nodes[0]["id"], links, nodes)
    return len(reachable) == len(nodes) - 1


def get_isolated_nodes(nodes: List[dict], links: List[dict]) -> List[dict]:
    """Returns nodes with no active (non-down) links connected to them."""
    active_links = [l for l in links if l.get("status", "up") != "down"]
    connected_ids = set()
    for link in active_links:
        connected_ids.add(link["sourceId"])
        connected_ids.add(link["targetId"])
    return [n for n in nodes if n["id"] not in connected_ids]


def get_link_utilization(active_packets: List[dict], links: List[dict]) -> List[dict]:
    """Returns per-link count of in-transit packets, useful for visualizing congestion."""
    link_map = {l["id"]: 0 for l in links}
    for pkt in active_packets:
        if pkt.get("status") == "in-transit":
            lid = pkt.get("linkId")
            if lid in link_map:
                link_map[lid] += 1
    return [{"linkId": lid, "packetCount": count} for lid, count in link_map.items()]


def export_arp_snapshot(
    nodes: List[dict],
    links: List[dict],
    arp_caches: Dict[str, List[dict]],
    tick_counter: int,
) -> dict:
    """
    Exports a full snapshot of the ARP simulation state.
    Useful for saving/restoring state or debugging.
    """
    return {
        "tick": tick_counter,
        "nodes": nodes,
        "links": links,
        "arpCaches": arp_caches,
        "conflicts": {
            "ip": detect_ip_conflicts(nodes),
            "mac": detect_mac_conflicts(nodes),
            "cacheInconsistencies": detect_cache_inconsistency(arp_caches, nodes),
        },
        "reachability": {
            "fullyConnected": is_fully_connected(nodes, links),
            "isolatedNodes": [n["name"] for n in get_isolated_nodes(nodes, links)],
        },
    }

# ARP reply mechanism and dynamic ARP cache update implemented

# ARP cache timeout and stale entry eviction logic added
