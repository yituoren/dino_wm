"""Wire protocol for the RoboTwin eval client/server split.

Length-prefixed JSON over a single TCP connection. Numpy arrays are
base64-encoded so the whole payload is valid JSON.

Wire frame: [4-byte big-endian length][UTF-8 JSON body]
Request:    {"cmd": "<name>", "args": {...}}
Response:   {"res": {...}}  |  {"error": "...", "traceback": "..."}

This file is intentionally duplicated in each policy repo and in
RoboTwin. Keep it dependency-free (stdlib + numpy only).
"""
from __future__ import annotations

import base64
import json
import socket
from typing import Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _object_hook(dct):
    if dct.get("__ndarray__"):
        raw = base64.b64decode(dct["data"])
        return np.frombuffer(raw, dtype=np.dtype(dct["dtype"])).reshape(dct["shape"])
    return dct


def _encode(obj: Any) -> bytes:
    body = json.dumps(obj, cls=_NumpyEncoder).encode("utf-8")
    return len(body).to_bytes(4, "big") + body


def _decode(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"), object_hook=_object_hook)


def _read_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        c = sock.recv(min(remaining, 65536))
        if not c:
            raise ConnectionError("peer closed while reading")
        chunks.append(c)
        remaining -= len(c)
    return b"".join(chunks)


def send(sock: socket.socket, obj: Any) -> None:
    sock.sendall(_encode(obj))


def recv(sock: socket.socket) -> Any:
    hdr = _read_exact(sock, 4)
    length = int.from_bytes(hdr, "big")
    return _decode(_read_exact(sock, length))


class RpcClient:
    """Thin blocking RPC client over a single TCP connection."""

    def __init__(self, host: str, port: int, timeout: float | None = None):
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def call(self, cmd: str, **kwargs) -> Any:
        send(self.sock, {"cmd": cmd, "args": kwargs})
        resp = recv(self.sock)
        if "error" in resp:
            raise RuntimeError(
                f"server error [{cmd}]: {resp['error']}\n{resp.get('traceback', '')}"
            )
        return resp["res"]

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
