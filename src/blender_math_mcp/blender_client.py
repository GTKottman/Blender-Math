"""Client for the Blender Math Bridge add-on (newline-delimited JSON over TCP)."""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Protocol


class BlenderError(RuntimeError):
    """Blender could not be reached, or a command failed inside Blender."""


class Transport(Protocol):
    def send(self, command: str, params: dict | None = None) -> Any: ...


class BlenderConnection:
    """Opens one short-lived connection per command (robust to Blender restarts)."""

    def __init__(self, host: str | None = None, port: int | None = None, timeout: float = 300.0):
        self.host = host or os.environ.get("BLENDER_MATH_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("BLENDER_MATH_PORT", 9877))
        self.timeout = timeout

    def send(self, command: str, params: dict | None = None) -> Any:
        payload = (json.dumps({"type": command, "params": params or {}}) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=10) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(payload)
                buf = bytearray()
                while not buf.endswith(b"\n"):
                    chunk = sock.recv(1 << 16)
                    if not chunk:
                        break
                    buf.extend(chunk)
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            raise BlenderError(
                f"Cannot reach Blender at {self.host}:{self.port} ({exc}). "
                "Open Blender, enable the 'Blender Math Bridge' add-on and press "
                "'Start Math MCP server' in the 3D View sidebar (N panel > Math MCP)."
            ) from exc
        if not buf:
            raise BlenderError("Blender closed the connection without answering.")
        response = json.loads(buf.decode("utf-8"))
        if response.get("status") != "ok":
            raise BlenderError(response.get("message", "Unknown error in Blender"))
        return response.get("result")
