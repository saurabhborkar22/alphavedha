"""Hetzner server scale helper — read status or resize for training runs.

Usage:
    python scripts/hetzner_scale.py status
    python scripts/hetzner_scale.py scale-up   --confirm   # requires explicit flag
    python scripts/hetzner_scale.py scale-down --confirm   # requires explicit flag

Environment:
    HETZNER_TOKEN   — API token from cloud.hetzner.com → Security → API Tokens
    HETZNER_SERVER  — server name (default: alphavedha-vps)

Scale strategy:
    Serving (always-on): cx23  — 2 vCPU, 4GB RAM, €3.99/mo
    Training (overnight): cx43 — 8 vCPU, 16GB RAM, €0.0192/hr ≈ €0.12/6hr run

    Resize causes ~5 min downtime (server reboot). Schedule during off-hours.
    After training, always scale back to cx23.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

SERVER_NAME = os.environ.get("HETZNER_SERVER", "alphavedha-vps")
SERVING_TYPE = "cx23"
TRAINING_TYPE = "cx43"
API_BASE = "https://api.hetzner.cloud/v1"


def _get_token() -> str:
    token = os.environ.get("HETZNER_TOKEN", "")
    if not token:
        print("ERROR: HETZNER_TOKEN environment variable not set.", file=sys.stderr)
        print("  export HETZNER_TOKEN=<your-token>", file=sys.stderr)
        sys.exit(1)
    return token


def _api_get(path: str) -> dict[str, Any]:
    token = _get_token()
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _api_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    token = _get_token()
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def _find_server() -> dict[str, Any]:
    data = _api_get("/servers")
    for s in data["servers"]:
        if s["name"] == SERVER_NAME:
            return s  # type: ignore[no-any-return]
    print(f"ERROR: Server '{SERVER_NAME}' not found.", file=sys.stderr)
    sys.exit(1)


def cmd_status() -> None:
    """Print current server type, status, and cost."""
    server = _find_server()
    st = server["server_type"]
    monthly = st["prices"][0]["price_monthly"]["gross"]
    hourly = st["prices"][0]["price_hourly"]["gross"]
    print(f"Server:  {server['name']}  (id={server['id']})")
    print(f"Status:  {server['status']}")
    print(f"Type:    {st['name']}  —  {st['cores']} vCPU  {st['memory']}GB RAM")
    print(f"Cost:    €{float(hourly):.4f}/hr  ·  €{float(monthly):.2f}/mo")
    print(f"IP:      {server['public_net']['ipv4']['ip']}")

    if st["name"] == SERVING_TYPE:
        print(f"\n[serving]  Currently on {SERVING_TYPE} (normal ops)")
    elif st["name"] == TRAINING_TYPE:
        print(f"\n[training] Currently scaled up to {TRAINING_TYPE} — remember to scale back!")
    else:
        print(f"\n[other]    Running on non-standard type {st['name']!r}")


def cmd_scale_up(confirm: bool) -> None:
    """Resize server from cx23 → cx43 for training. Causes ~5 min downtime."""
    server = _find_server()
    current = server["server_type"]["name"]

    if current == TRAINING_TYPE:
        print(f"Already on {TRAINING_TYPE} — nothing to do.")
        return

    if current != SERVING_TYPE:
        print(f"WARNING: Current type is {current!r}, expected {SERVING_TYPE!r}.")

    print(f"Plan: resize {SERVER_NAME} from {current} → {TRAINING_TYPE}")
    print("      This will reboot the server (~5 min downtime).")
    print(f"      Cost delta: +€{0.0192 - 0.0064:.4f}/hr while scaled up")

    if not confirm:
        print("\nRe-run with --confirm to execute.")
        return

    # Power off
    print("\nPowering off server...")
    _api_post(f"/servers/{server['id']}/actions/poweroff", {})
    _wait_for_status(server["id"], "off")

    # Resize
    print(f"Resizing to {TRAINING_TYPE}...")
    _api_post(
        f"/servers/{server['id']}/actions/change_type",
        {"server_type": TRAINING_TYPE, "upgrade_disk": False},
    )
    time.sleep(10)

    # Power on
    print("Powering back on...")
    _api_post(f"/servers/{server['id']}/actions/poweron", {})
    _wait_for_status(server["id"], "running")

    print(f"\nDone. Server is now {TRAINING_TYPE} (8 vCPU, 16GB RAM).")
    print("Run training, then: python scripts/hetzner_scale.py scale-down --confirm")


def cmd_scale_down(confirm: bool) -> None:
    """Resize server from cx43 → cx23 after training. Causes ~5 min downtime."""
    server = _find_server()
    current = server["server_type"]["name"]

    if current == SERVING_TYPE:
        print(f"Already on {SERVING_TYPE} — nothing to do.")
        return

    print(f"Plan: resize {SERVER_NAME} from {current} → {SERVING_TYPE}")
    print("      This will reboot the server (~5 min downtime).")

    if not confirm:
        print("\nRe-run with --confirm to execute.")
        return

    print("\nPowering off server...")
    _api_post(f"/servers/{server['id']}/actions/poweroff", {})
    _wait_for_status(server["id"], "off")

    print(f"Resizing to {SERVING_TYPE}...")
    _api_post(
        f"/servers/{server['id']}/actions/change_type",
        {"server_type": SERVING_TYPE, "upgrade_disk": False},
    )
    time.sleep(10)

    print("Powering back on...")
    _api_post(f"/servers/{server['id']}/actions/poweron", {})
    _wait_for_status(server["id"], "running")

    print(f"\nDone. Server is back to {SERVING_TYPE} (2 vCPU, 4GB RAM, €3.99/mo).")


def _wait_for_status(server_id: int, target: str, timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = _api_get(f"/servers/{server_id}")
        status = data["server"]["status"]
        if status == target:
            return
        print(f"  ... waiting for status={target!r}, current={status!r}")
        time.sleep(10)
    print(f"ERROR: Timed out waiting for server to reach {target!r}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = sys.argv[1:]
    confirm = "--confirm" in args
    cmd = next((a for a in args if not a.startswith("--")), "status")

    if cmd == "status":
        cmd_status()
    elif cmd == "scale-up":
        cmd_scale_up(confirm)
    elif cmd == "scale-down":
        cmd_scale_down(confirm)
    else:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        print("Usage: hetzner_scale.py [status|scale-up|scale-down] [--confirm]")
        sys.exit(1)


if __name__ == "__main__":
    main()
