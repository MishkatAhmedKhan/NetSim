"""
DHCP / DORA simulation logic.
Ported from src/store/useDhcpStore.ts and src/simulation/DhcpModels.ts
"""
from __future__ import annotations
import math
import random
import uuid
from typing import Dict, List, Optional, Tuple

# ─── Shared probability engine ────────────────────────────────────────────────
from probability import check_probability


# ─── Seeded PRNG (Mulberry32 equivalent) ─────────────────────────────────────

def mulberry32(seed: int):
    """
    Returns a seeded PRNG function that mimics the Mulberry32 JS function.
    Results will differ from JS due to bit-width differences, but the structure is identical.
    """
    state = [seed]

    def rng() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ (t + ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rng


# ─── Probability Helpers ──────────────────────────────────────────────────────

def poisson_sample(lam: float, rng) -> int:
    """Knuth's algorithm for Poisson sampling."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng()
        if p <= L:
            break
    return k - 1


def exponential_sample(lam: float, rng) -> float:
    """Inverse transform sampling for Exponential distribution."""
    return -math.log(1 - rng()) / lam


# ─── MAC Address Generator ────────────────────────────────────────────────────

def generate_dhcp_mac(rng) -> str:
    """Generates a deterministic MAC using a PRNG (mirrors JS generateMAC)."""
    def hex_byte():
        return f"{int(rng() * 256):02x}"
    return f"AA:BB:{hex_byte()}:{hex_byte()}:{hex_byte()}:{hex_byte()}"


# ─── IP Utilities ─────────────────────────────────────────────────────────────

def ip_to_num(ip: str) -> int:
    """Converts dotted IP string to unsigned 32-bit integer."""
    parts = list(map(int, ip.split('.')))
    result = 0
    for p in parts:
        result = (result << 8) + p
    return result & 0xFFFFFFFF


def num_to_ip(num: int) -> str:
    """Converts unsigned 32-bit integer to dotted IP string."""
    return f"{(num >> 24) & 255}.{(num >> 16) & 255}.{(num >> 8) & 255}.{num & 255}"


# ─── Node naming ─────────────────────────────────────────────────────────────

CLIENT_NAMES = [f'PC{i}' for i in range(1, 21)]
SERVER_NAMES = ['SRV1', 'SRV2', 'SRV3', 'SRV4']
ROUTER_NAMES = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
SWITCH_NAMES = ['SW1', 'SW2', 'SW3', 'SW4', 'SW5', 'SW6']


# ─── Link Helpers ─────────────────────────────────────────────────────────────

def find_link_between(a: str, b: str, links: List[dict]) -> Optional[dict]:
    """Finds the link between nodes a and b."""
    for link in links:
        if (link["sourceId"] == a and link["targetId"] == b) or \
           (link["sourceId"] == b and link["targetId"] == a):
            return link
    return None


def is_node_reachable_from(start_id: str, end_id: str, links: List[dict], excluded: set) -> bool:
    """BFS reachability check, ignoring nodes in excluded set."""
    if start_id == end_id:
        return True
    visited = {start_id} | excluded
    queue = [start_id]
    while queue:
        current = queue.pop(0)
        for link in links:
            if link["sourceId"] == current:
                neighbor = link["targetId"]
            elif link["targetId"] == current:
                neighbor = link["sourceId"]
            else:
                continue
            if neighbor == end_id:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def find_server_in_subnet(client_id: str, nodes: List[dict], links: List[dict]) -> Optional[dict]:
    """
    Finds a DHCP server reachable from the client within the same subnet
    (BFS stops at routers — they block broadcasts).
    Mirrors findServerInSubnet() in useDhcpStore.ts.
    """
    visited = {client_id}
    queue = [client_id]
    node_map = {n["id"]: n for n in nodes}

    while queue:
        current = queue.pop(0)
        current_node = node_map.get(current)
        if not current_node:
            continue
        for link in links:
            if link["sourceId"] == current:
                neighbor_id = link["targetId"]
            elif link["targetId"] == current:
                neighbor_id = link["sourceId"]
            else:
                continue
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            neighbor = node_map.get(neighbor_id)
            if not neighbor:
                continue
            if neighbor["type"] == "server":
                return neighbor
            if neighbor["type"] != "router":  # Don't cross routers
                queue.append(neighbor_id)
    return None


def find_any_reachable_server(client_id: str, nodes: List[dict], links: List[dict]) -> Optional[dict]:
    """Finds any server anywhere in the topology reachable from client_id."""
    visited = {client_id}
    queue = [client_id]
    node_map = {n["id"]: n for n in nodes}

    while queue:
        current = queue.pop(0)
        for link in links:
            if link["sourceId"] == current:
                neighbor_id = link["targetId"]
            elif link["targetId"] == current:
                neighbor_id = link["sourceId"]
            else:
                continue
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            neighbor = node_map.get(neighbor_id)
            if not neighbor:
                continue
            if neighbor["type"] == "server":
                return neighbor
            queue.append(neighbor_id)
    return None


# ─── IP Pool Allocation ───────────────────────────────────────────────────────

def allocate_ip_from_pool(server_config: dict, leases: List[dict], rng) -> Optional[str]:
    """
    Allocates an available IP from the DHCP server pool using PRNG.
    Mirrors allocateIpFromPool() in useDhcpStore.ts.
    """
    start = ip_to_num(server_config["ipPoolStart"])
    end = ip_to_num(server_config["ipPoolEnd"])
    pool_size = end - start + 1

    active_lease_ips = {
        l["ip"] for l in leases if l["status"] in ("offered", "active")
    }

    # Try up to pool_size times to find a free one
    for _ in range(pool_size):
        offset = int(rng() * pool_size)
        candidate_ip = num_to_ip(start + offset)
        if candidate_ip not in active_lease_ips:
            return candidate_ip

    return None  # Pool exhausted


# ─── Node Factory ─────────────────────────────────────────────────────────────

def create_dhcp_node(node_type: str, x: float, y: float, existing_nodes: List[dict], prng_seed: int) -> dict:
    """
    Creates a new DHCP topology node.
    Mirrors addNode() in useDhcpStore.ts.
    """
    rng = mulberry32(prng_seed + len(existing_nodes) * 7 + id(existing_nodes))
    mac = generate_dhcp_mac(rng)

    existing_of_type = [n for n in existing_nodes if n["type"] == node_type]
    if node_type == "client":
        name = CLIENT_NAMES[len(existing_of_type) % len(CLIENT_NAMES)]
    elif node_type == "server":
        name = SERVER_NAMES[len(existing_of_type) % len(SERVER_NAMES)]
    elif node_type == "switch":
        name = SWITCH_NAMES[len(existing_of_type) % len(SWITCH_NAMES)]
    else:
        name = ROUTER_NAMES[len(existing_of_type) % len(ROUTER_NAMES)]

    default_ip = None
    if node_type in ("router", "server"):
        while True:
            last_octet = random.randint(1, 254)
            candidate = f"192.168.1.{last_octet}"
            if not any(n.get("ipAddress") == candidate for n in existing_nodes):
                default_ip = candidate
                break

    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "type": node_type,
        "position": {"x": x, "y": y},
        "macAddress": mac,
        "ipAddress": default_ip,
        "leaseExpiry": None,
        "isRelayAgent": False,
    }


# ─── DORA Sequence Initiator ──────────────────────────────────────────────────

def start_dora(
    client_id: str,
    nodes: List[dict],
    links: List[dict],
    dora_sequences: List[dict],
    leases: List[dict],
    tick_counter: int,
    prng_seed: int,
    packet_loss_rate: float = 0.0,
    packet_dist: str = "Uniform",
) -> dict:
    """
    Initiates a DORA sequence for the given client.
    Mirrors startDora() in useDhcpStore.ts.
    Returns dict with new_messages, new_sequences, new_logs.
    """
    client = next((n for n in nodes if n["id"] == client_id and n["type"] == "client"), None)
    if not client:
        return {"messages": [], "sequences": [], "logs": []}

    # Already has active sequence?
    if any(s["clientId"] == client_id and s["status"] == "in-progress" for s in dora_sequences):
        return {"messages": [], "sequences": [], "logs": []}

    server = find_server_in_subnet(client_id, nodes, links)
    any_server = server or find_any_reachable_server(client_id, nodes, links)

    if not any_server:
        return {
            "messages": [], "sequences": [],
            "logs": [{
                "id": str(uuid.uuid4()), "tick": tick_counter,
                "message": f"{client['name']}: No DHCP server reachable!",
                "type": "error"
            }]
        }

    # Generate XID from PRNG
    rng = mulberry32(prng_seed + tick_counter + ord(client_id[0]))
    xid = int(rng() * 0xFFFFFFFF)

    # Find the initial link toward the server (or any reachable server)
    link = find_link_between(client_id, any_server["id"], links)
    if not link:
        # Find first-hop link BFS
        from collections import deque
        visited = {client_id}
        queue = deque([[client_id]])
        first_link = None
        while queue:
            path = queue.popleft()
            current = path[-1]
            for l in links:
                if l["sourceId"] == current:
                    nbr = l["targetId"]
                    lnk = l
                elif l["targetId"] == current:
                    nbr = l["sourceId"]
                    lnk = l
                else:
                    continue
                if nbr not in visited:
                    if len(path) == 1:  # First hop from client
                        first_link = lnk
                        break
                    visited.add(nbr)
                    queue.append(path + [nbr])
            if first_link:
                break
        link = first_link
    if not link:
        return {"messages": [], "sequences": [], "logs": []}

    is_isolated = not server
    effective_loss = max(packet_loss_rate, link.get("packetLossRate", 0))
    # Use the shared probability engine with the selected distribution mode
    is_dropped = is_isolated or check_probability(packet_dist, effective_loss)

    next_hop = link["targetId"] if link["sourceId"] == client_id else link["sourceId"]

    discover_msg = {
        "id": str(uuid.uuid4()),
        "xid": xid,
        "type": "DISCOVER",
        "srcIP": "0.0.0.0",
        "destIP": "255.255.255.255",
        "srcMAC": client["macAddress"],
        "destMAC": "FF:FF:FF:FF:FF:FF",
        "yiaddr": "0.0.0.0",
        "leaseTime": 0,
        "subnetMask": "",
        "gateway": "",
        "dns": "",
        "linkId": link["id"],
        "sourceNodeId": client_id,
        "targetNodeId": next_hop,
        "finalTargetNodeId": any_server["id"],
        "progress": 0,
        "status": "in-transit",
        "willDrop": is_dropped,
        "srcPort": 68,
        "destPort": 67,
    }

    sequence = {
        "id": str(uuid.uuid4()),
        "xid": xid,
        "clientId": client_id,
        "clientName": client["name"],
        "serverId": any_server["id"],
        "serverName": any_server["name"],
        "messages": [discover_msg],
        "status": "in-progress",
        "retryCount": 0,
        "startTick": tick_counter,
        "retryAtTick": None,
        "lastDroppedType": None,
        "currentBackoff": 4,
        "offeredIp": None,
    }

    return {
        "messages": [discover_msg],
        "sequences": [sequence],
        "logs": [{
            "id": str(uuid.uuid4()),
            "tick": tick_counter,
            "message": f"{client['name']} → DISCOVER (XID: 0x{xid:08X})",
            "type": "info"
        }]
    }


# ─── Auto Arrivals ────────────────────────────────────────────────────────────

def start_auto_arrivals(
    nodes: List[dict],
    dora_sequences: List[dict],
    arrival_lambda: float,
    tick_counter: int,
    prng_seed: int,
    links: List[dict],
    leases: List[dict],
    packet_loss_rate: float = 0.0,
    packet_dist: str = "Uniform",
) -> dict:
    """
    Triggers Poisson-distributed client DHCP arrivals.
    Mirrors startAutoArrivals() in useDhcpStore.ts.
    Returns {messages, sequences, logs}.
    """
    rng = mulberry32(prng_seed + tick_counter * 13)
    arrivals = poisson_sample(arrival_lambda, rng)

    available_clients = [
        n for n in nodes
        if n["type"] == "client"
        and not n.get("ipAddress")
        and not any(s["clientId"] == n["id"] and s["status"] == "in-progress" for s in dora_sequences)
    ]

    all_messages = []
    all_sequences = list(dora_sequences)
    all_logs = []

    count = min(arrivals, len(available_clients))
    for i in range(count):
        idx = int(exponential_sample(1, rng) * len(available_clients)) % len(available_clients)
        client = available_clients[idx]
        result = start_dora(
            client["id"], nodes, links, all_sequences, leases,
            tick_counter, prng_seed, packet_loss_rate, packet_dist
        )
        all_messages.extend(result["messages"])
        all_sequences.extend(result["sequences"])
        all_logs.extend(result["logs"])
        available_clients.pop(idx)

    return {"messages": all_messages, "sequences": all_sequences, "logs": all_logs}


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def run_monte_carlo(
    total_runs: int,
    packet_loss_percent: float,
    max_retries: int,
    num_clients: int,
    packet_dist: str = "Uniform",
) -> dict:
    """
    Runs a Monte Carlo simulation of the DHCP DORA process.
    Mirrors runMonteCarlo() in useDhcpStore.ts.
    Returns a MonteCarloResult dict.
    """
    loss_rate = packet_loss_percent / 100.0
    success_count = 0
    fail_count = 0
    total_retries = 0
    histogram: Dict[int, int] = {}

    for _ in range(total_runs):
        retries = 0
        success = False

        for attempt in range(max_retries + 1):
            # Each DORA exchange requires 4 messages: DISCOVER, OFFER, REQUEST, ACK
            # Use the shared probability engine with the specified distribution mode
            all_delivered = not any(check_probability(packet_dist, loss_rate) for _ in range(4))
            if all_delivered:
                success = True
                retries = attempt
                break
            retries = attempt + 1

        if success:
            success_count += 1
        else:
            fail_count += 1

        total_retries += retries
        histogram[retries] = histogram.get(retries, 0) + 1

    avg_retries = total_retries / total_runs if total_runs > 0 else 0.0
    success_rate = (success_count / total_runs) * 100 if total_runs > 0 else 0.0

    histogram_list = sorted(
        [{"retries": k, "count": v} for k, v in histogram.items()],
        key=lambda x: x["retries"]
    )

    return {
        "totalRuns": total_runs,
        "successCount": success_count,
        "failCount": fail_count,
        "successRate": success_rate,
        "avgRetries": avg_retries,
        "histogram": histogram_list,
        "packetLossPercent": packet_loss_percent,
    }
