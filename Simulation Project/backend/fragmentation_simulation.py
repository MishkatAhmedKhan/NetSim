"""
IP Fragmentation simulation logic.
Ported from src/store/useFragmentationStore.ts
"""
from __future__ import annotations
import random
import uuid
from typing import Dict, List, Optional, Tuple

# ─── Shared probability engine ────────────────────────────────────────────────
from probability import check_probability

STARTING_IDENTIFICATION = 1000


# ─── Path Finding ─────────────────────────────────────────────────────────────

def bfs_all_paths(start_id: str, end_id: str, links: List[dict], max_paths: int = 10) -> List[List[str]]:
    """
    BFS to find all simple paths between two nodes (capped at max_paths).
    Mirrors bfsAllPaths() in useFragmentationStore.ts.
    """
    results = []
    stack = [{"node": start_id, "path": [start_id]}]
    while stack and len(results) < max_paths:
        entry = stack.pop()
        node, path = entry["node"], entry["path"]
        if node == end_id:
            results.append(path)
            continue
        neighbors = []
        for link in links:
            if link["sourceId"] == node:
                neighbors.append(link["targetId"])
            elif link["targetId"] == node:
                neighbors.append(link["sourceId"])
        for nbr in neighbors:
            if nbr not in path:
                stack.append({"node": nbr, "path": path + [nbr]})
    return results


def bfs_random_path(start_id: str, end_id: str, links: List[dict]) -> List[str]:
    """Picks a random path among all available paths."""
    all_paths = bfs_all_paths(start_id, end_id, links)
    if not all_paths:
        return []
    return random.choice(all_paths)


def bfs_shortest_path(start_id: str, end_id: str, links: List[dict]) -> List[str]:
    """Picks the shortest path among all available paths."""
    all_paths = bfs_all_paths(start_id, end_id, links)
    if not all_paths:
        return []
    return min(all_paths, key=len)


# ─── ICMP Packet Factory ──────────────────────────────────────────────────────

def create_icmp_packet(
    from_node_id: str,
    to_node_id: str,
    links: List[dict],
    original_id: int,
    icmp_type: str,
    icmp_message: str,
) -> Optional[dict]:
    """
    Creates an ICMP error packet travelling back from from_node_id to to_node_id.
    Mirrors createICMPPacket() in useFragmentationStore.ts.
    """
    reverse_path = bfs_shortest_path(from_node_id, to_node_id, links)
    if len(reverse_path) < 2:
        return None

    next_hop = reverse_path[1]
    link = next(
        (l for l in links
         if (l["sourceId"] == from_node_id and l["targetId"] == next_hop) or
            (l["targetId"] == from_node_id and l["sourceId"] == next_hop)),
        None
    )
    if not link:
        return None

    return {
        "packetId": str(uuid.uuid4()),
        "originalId": original_id,
        "sourceId": from_node_id,
        "targetId": to_node_id,
        "currentNodeId": from_node_id,
        "nextHopId": next_hop,
        "linkId": link["id"],
        "progress": 0,
        "speedMultiplier": 1.0,
        "route": reverse_path,
        "headerLength": 20,
        "payloadLength": 28,
        "totalLength": 48,
        "offset": 0,
        "mfFlag": 0,
        "dfFlag": 0,
        "isICMP": True,
        "icmpType": icmp_type,
        "icmpMessage": icmp_message,
        "status": "in-transit",
    }


# ─── Node Factory ─────────────────────────────────────────────────────────────

def create_frag_node(node_type: str, x: float, y: float, existing_nodes: List[dict]) -> dict:
    """
    Creates a new fragmentation topology node.
    Mirrors addNode() in useFragmentationStore.ts.
    """
    count_of_type = sum(1 for n in existing_nodes if n["type"] == node_type)
    prefix = "H" if node_type == "host" else "R"
    return {
        "id": str(uuid.uuid4()),
        "name": f"{prefix}{count_of_type + 1}",
        "type": node_type,
        "position": {"x": x, "y": y},
    }


def create_frag_link(source_id: str, target_id: str, mtu: int) -> dict:
    """Creates a new fragmentation link with default probabilistic properties."""
    return {
        "id": str(uuid.uuid4()),
        "sourceId": source_id,
        "targetId": target_id,
        "mtu": mtu,
        "dropRate": 0.0,
        "ber": 0.0,
        "queueCapacity": 10,
        "jitter": 0.0,
    }


# ─── Packet Send ──────────────────────────────────────────────────────────────

def send_packet(
    source_id: str,
    target_id: str,
    payload_size: int,
    df_flag: int,
    links: List[dict],
    packet_counter: int,
) -> Tuple[List[dict], int]:
    """
    Creates one or more IP packets/fragments for the given source→target transmission.
    Mirrors sendPacket() in useFragmentationStore.ts.
    Returns (new_packets, updated_packet_counter).
    """
    path = bfs_shortest_path(source_id, target_id, links)
    if len(path) < 2:
        return [], packet_counter

    next_hop_id = path[1]
    link = next(
        (l for l in links
         if (l["sourceId"] == source_id and l["targetId"] == next_hop_id) or
            (l["targetId"] == source_id and l["sourceId"] == next_hop_id)),
        None
    )
    if not link:
        return [], packet_counter

    header_length = 20
    total_length = payload_size + header_length
    original_id = packet_counter
    new_packets = []

    if total_length > link["mtu"]:
        if df_flag == 1:
            # DF set — can't fragment, will be dropped at transit
            new_packets.append({
                "packetId": str(uuid.uuid4()),
                "originalId": original_id,
                "sourceId": source_id,
                "targetId": target_id,
                "currentNodeId": source_id,
                "nextHopId": next_hop_id,
                "linkId": link["id"],
                "progress": 0,
                "speedMultiplier": 1.0,
                "route": path,
                "headerLength": header_length,
                "payloadLength": payload_size,
                "totalLength": total_length,
                "offset": 0,
                "mfFlag": 0,
                "dfFlag": 1,
                "status": "in-transit",
            })
        else:
            # Fragment at source
            max_payload = (link["mtu"] - header_length) // 8 * 8
            remaining = payload_size
            offset_bytes = 0
            frag_idx = 0

            while remaining > 0:
                this_size = min(max_payload, remaining)
                remaining -= this_size
                is_last = remaining == 0

                frag_route = bfs_random_path(source_id, target_id, links) or path
                frag_route_idx = frag_route.index(source_id) if source_id in frag_route else -1
                frag_next_hop = frag_route[frag_route_idx + 1] if 0 <= frag_route_idx < len(frag_route) - 1 else next_hop_id

                frag_link = next(
                    (l for l in links
                     if (l["sourceId"] == source_id and l["targetId"] == frag_next_hop) or
                        (l["targetId"] == source_id and l["sourceId"] == frag_next_hop)),
                    link
                )

                new_packets.append({
                    "packetId": str(uuid.uuid4()),
                    "originalId": original_id,
                    "sourceId": source_id,
                    "targetId": target_id,
                    "currentNodeId": source_id,
                    "nextHopId": frag_next_hop,
                    "linkId": frag_link["id"],
                    "progress": -(frag_idx * 0.35),
                    "speedMultiplier": 1.0,
                    "route": frag_route,
                    "headerLength": header_length,
                    "payloadLength": this_size,
                    "totalLength": this_size + header_length,
                    "offset": offset_bytes // 8,
                    "mfFlag": 0 if is_last else 1,
                    "dfFlag": 0,
                    "fragMath": {
                        "mtu": frag_link["mtu"],
                        "headerLen": header_length,
                        "maxPayload": max_payload,
                        "offsetBytes": offset_bytes,
                    },
                    "status": "in-transit",
                })
                offset_bytes += this_size
                frag_idx += 1
    else:
        # No fragmentation needed
        new_packets.append({
            "packetId": str(uuid.uuid4()),
            "originalId": original_id,
            "sourceId": source_id,
            "targetId": target_id,
            "currentNodeId": source_id,
            "nextHopId": next_hop_id,
            "linkId": link["id"],
            "progress": 0,
            "speedMultiplier": 1.0,
            "route": path,
            "headerLength": header_length,
            "payloadLength": payload_size,
            "totalLength": total_length,
            "offset": 0,
            "mfFlag": 0,
            "dfFlag": df_flag,
            "status": "in-transit",
        })

    return new_packets, packet_counter + 1


# ─── Simulation Tick ──────────────────────────────────────────────────────────

def simulation_tick(
    nodes: List[dict],
    links: List[dict],
    active_packets: List[dict],
    reassembly_buffers: List[dict],
    drop_animations: List[dict],
    tick_counter: int,
    simulation_speed: float = 1.0,
    reassembly_timeout: int = 200,
    drop_dist: str = "Uniform",
) -> dict:
    """
    Advances the fragmentation simulation by one tick.
    Mirrors simulationTick() in useFragmentationStore.ts.
    Returns updated simulation state dict.
    """
    current_tick = tick_counter + 1

    if not active_packets and not any(b["status"] == "buffering" for b in reassembly_buffers):
        return {
            "activePackets": [],
            "droppedPackets": [],
            "deliveredPackets": [],
            "reassemblyBuffers": reassembly_buffers,
            "dropAnimations": drop_animations,
            "tickCounter": current_tick,
            "isSimulationRunning": False,
        }

    updated_packets: List[dict] = []
    new_dropped: List[dict] = []
    newly_generated: List[dict] = []
    new_buffered: List[dict] = []
    new_drop_animations: List[dict] = list(drop_animations)
    new_delivered_icmp: List[dict] = []

    node_map = {n["id"]: n for n in nodes}
    link_map = {l["id"]: l for l in links}

    def get_packet_position(pkt: dict) -> dict:
        link = link_map.get(pkt["linkId"])
        if link:
            src = node_map.get(link["sourceId"])
            tgt = node_map.get(link["targetId"])
            if src and tgt:
                p = pkt["progress"]
                if pkt.get("nextHopId") == link["sourceId"]:
                    p = 1 - p
                return {
                    "x": src["position"]["x"] + (tgt["position"]["x"] - src["position"]["x"]) * p,
                    "y": src["position"]["y"] + (tgt["position"]["y"] - src["position"]["y"]) * p,
                }
        node = node_map.get(pkt.get("currentNodeId", ""))
        return {"x": node["position"]["x"], "y": node["position"]["y"]} if node else {"x": 0, "y": 0}

    def record_drop(pkt: dict) -> None:
        pos = get_packet_position(pkt)
        new_drop_animations.append({
            "id": str(uuid.uuid4()),
            "x": pos["x"],
            "y": pos["y"],
            "reason": pkt.get("dropReason", "Dropped"),
            "originalId": pkt["originalId"],
            "createdTick": current_tick,
        })

    for packet in active_packets:
        packet = dict(packet)

        if packet["status"] == "in-transit":
            packet["progress"] = packet["progress"] + 0.008 * simulation_speed

            if packet["progress"] >= 1.0:
                packet["currentNodeId"] = packet["nextHopId"]
                packet["progress"] = 0.0

                # BER check on arrival
                transit_link = link_map.get(packet["linkId"])
                if transit_link and transit_link.get("ber", 0) > 0:
                    ber = transit_link["ber"]
                    p_error = 1 - (1 - ber) ** (packet["totalLength"] * 8)
                    if random.random() < p_error:
                        packet["status"] = "dropped"
                        packet["dropReason"] = f"Bit Error (BER={ber}, P={p_error * 100:.2f}%)"
                        new_dropped.append(packet)
                        record_drop(packet)
                        continue

                # Per-link drop rate check — uses the selected distribution mode
                if transit_link and transit_link.get("dropRate", 0) > 0 and check_probability(drop_dist, transit_link["dropRate"]):
                    packet["status"] = "dropped"
                    pct = transit_link["dropRate"] * 100
                    packet["dropReason"] = f"Random Loss ({pct:.0f}% drop rate)"
                    new_dropped.append(packet)
                    record_drop(packet)
                    continue

                packet["status"] = "processing"

            updated_packets.append(packet)

        elif packet["status"] == "processing":
            # ICMP packets at destination
            if packet.get("isICMP") and packet["currentNodeId"] == packet["targetId"]:
                packet["status"] = "delivered"
                new_delivered_icmp.append(packet)
                continue

            # Reached destination host
            if packet["currentNodeId"] == packet["targetId"]:
                packet["status"] = "buffered"
                new_buffered.append(packet)
                continue

            # Follow pre-assigned route
            route = packet.get("route", [])
            try:
                route_idx = route.index(packet["currentNodeId"])
            except ValueError:
                route_idx = -1

            if route_idx < 0 or route_idx >= len(route) - 1:
                packet["status"] = "dropped"
                packet["dropReason"] = "No Route to Destination"
                new_dropped.append(packet)
                record_drop(packet)
                continue

            next_hop_id = route[route_idx + 1]
            next_link = next(
                (l for l in links
                 if (l["sourceId"] == packet["currentNodeId"] and l["targetId"] == next_hop_id) or
                    (l["targetId"] == packet["currentNodeId"] and l["sourceId"] == next_hop_id)),
                None
            )

            if not next_link:
                packet["status"] = "dropped"
                packet["dropReason"] = "Link Not Found"
                new_dropped.append(packet)
                record_drop(packet)
                continue

            # Queue overflow check
            fragments_on_link = sum(
                1 for p in active_packets
                if p.get("linkId") == next_link["id"] and p.get("status") == "in-transit"
            )
            if fragments_on_link >= next_link.get("queueCapacity", 10):
                packet["status"] = "dropped"
                packet["dropReason"] = f"Queue Full ({next_link.get('queueCapacity', 10)} max)"
                new_dropped.append(packet)
                record_drop(packet)
                continue

            # MTU check at intermediate router
            if packet["totalLength"] > next_link["mtu"]:
                if packet.get("dfFlag") == 1:
                    packet["status"] = "dropped"
                    packet["dropReason"] = f"DF Set, Exceeds MTU {next_link['mtu']}"
                    new_dropped.append(packet)
                    record_drop(packet)

                    if not packet.get("isICMP"):
                        icmp = create_icmp_packet(
                            packet["currentNodeId"], packet["sourceId"],
                            links, packet["originalId"],
                            "Type 3, Code 4", f"Fragmentation Needed (MTU {next_link['mtu']})"
                        )
                        if icmp:
                            newly_generated.append(icmp)
                    continue

                # Fragment here
                max_payload = (next_link["mtu"] - packet["headerLength"]) // 8 * 8
                remaining_payload = packet["payloadLength"]
                current_offset_bytes = packet["offset"] * 8
                frag_index = 0

                while remaining_payload > 0:
                    this_size = min(max_payload, remaining_payload)
                    remaining_payload -= this_size
                    is_last = remaining_payload == 0
                    combined_mf = packet.get("mfFlag", 0) if is_last else 1

                    frag_route = bfs_random_path(packet["currentNodeId"], packet["targetId"], links) or route
                    frag_route_idx = frag_route.index(packet["currentNodeId"]) if packet["currentNodeId"] in frag_route else -1
                    frag_next_hop = frag_route[frag_route_idx + 1] if 0 <= frag_route_idx < len(frag_route) - 1 else next_hop_id

                    frag_link = next(
                        (l for l in links
                         if (l["sourceId"] == packet["currentNodeId"] and l["targetId"] == frag_next_hop) or
                            (l["targetId"] == packet["currentNodeId"] and l["sourceId"] == frag_next_hop)),
                        next_link
                    )

                    newly_generated.append({
                        "packetId": str(uuid.uuid4()),
                        "originalId": packet["originalId"],
                        "sourceId": packet["sourceId"],
                        "targetId": packet["targetId"],
                        "currentNodeId": packet["currentNodeId"],
                        "nextHopId": frag_next_hop,
                        "linkId": frag_link["id"],
                        "progress": -(frag_index * 0.35),
                        "speedMultiplier": 1.0,
                        "route": frag_route,
                        "headerLength": packet["headerLength"],
                        "payloadLength": this_size,
                        "totalLength": this_size + packet["headerLength"],
                        "offset": current_offset_bytes // 8,
                        "mfFlag": combined_mf,
                        "dfFlag": 0,
                        "fragMath": {
                            "mtu": frag_link["mtu"],
                            "headerLen": packet["headerLength"],
                            "maxPayload": max_payload,
                            "offsetBytes": current_offset_bytes,
                        },
                        "status": "in-transit",
                    })
                    current_offset_bytes += this_size
                    frag_index += 1
            else:
                packet["nextHopId"] = next_hop_id
                packet["linkId"] = next_link["id"]
                packet["progress"] = 0.0
                packet["status"] = "in-transit"
                updated_packets.append(packet)

    # Reassembly buffers
    updated_buffers = list(reassembly_buffers)
    for pkt in new_buffered:
        buf = next(
            (b for b in updated_buffers
             if b["originalId"] == pkt["originalId"] and b["targetId"] == pkt["targetId"]),
            None
        )
        if not buf:
            buf = {
                "originalId": pkt["originalId"],
                "targetId": pkt["targetId"],
                "expectedTotal": None,
                "receivedFragments": [],
                "startTick": current_tick,
                "timeoutTicks": reassembly_timeout,
                "status": "buffering",
                "arrivalCounter": 0,
            }
            updated_buffers.append(buf)

        if buf["status"] == "buffering":
            buf["arrivalCounter"] += 1
            pkt["arrivalOrder"] = buf["arrivalCounter"]
            pkt["status"] = "delivered"
            buf["receivedFragments"].append(pkt)
            if pkt["mfFlag"] == 0:
                buf["expectedTotal"] = pkt["offset"] * 8 + pkt["payloadLength"]
            if buf["expectedTotal"] is not None:
                received = sum(f["payloadLength"] for f in buf["receivedFragments"])
                if received >= buf["expectedTotal"]:
                    buf["status"] = "complete"

    # Timeout check
    for buf in updated_buffers:
        if buf["status"] == "buffering":
            age = current_tick - buf["startTick"]
            if age > buf["timeoutTicks"]:
                buf["status"] = "timeout"
                for frag in buf["receivedFragments"]:
                    frag["status"] = "dropped"
                    frag["dropReason"] = "Reassembly Timeout"
                if buf["receivedFragments"]:
                    first_frag = buf["receivedFragments"][0]
                    icmp = create_icmp_packet(
                        buf["targetId"], first_frag["sourceId"], links,
                        buf["originalId"], "Type 11, Code 1", "Reassembly Time Exceeded"
                    )
                    if icmp:
                        newly_generated.append(icmp)

    # Expired drop animations (60 ticks ≈ 3 seconds)
    existing_animations = [a for a in new_drop_animations if current_tick - a["createdTick"] < 60]

    final_active = updated_packets + newly_generated
    is_running = bool(final_active) or any(b["status"] == "buffering" for b in updated_buffers)

    return {
        "tickCounter": current_tick,
        "activePackets": final_active,
        "droppedPackets": new_dropped,
        "deliveredPackets": [
            f for b in updated_buffers for f in b["receivedFragments"] if f["status"] == "delivered"
        ] + new_delivered_icmp,
        "reassemblyBuffers": updated_buffers,
        "dropAnimations": existing_animations,
        "isSimulationRunning": is_running,
    }
