"""Remote LIBERO executor client for two-process (rats ↔ cap-x-geniac) operation.

Presents the same duck-typed call surface the RATS Executor / trial runner /
runtime expect from a ``FrankaLiberoCodeEnv`` (``reset`` / ``step(code)`` /
``render`` / ``_get_observation`` / video getters / ``_apis``), but delegates
every call over length-prefixed msgpack-over-TCP to a
``capx.serving.libero_executor_server`` running in the **cap-x-geniac** venv
(the "body"). Code-as-policy is executed **server-side** via ``env.step(code)``,
so cap-x-geniac's env / API / perception edits are reflected live here.

This module intentionally has **zero** LIBERO / capx imports so it can run
inside the rats venv, in a different process (and, if tunnelled, on a different
host). Swap it in via the env config::

    env:
      _target_: rats.envs.simulators.libero_remote.FrankaLiberoRemoteEnv
      remote_env_url: 127.0.0.1:9200

Wire protocol mirrors ``scripts/mlspaces_server.py`` /
``rats/envs/simulators/molmospaces_remote.py``.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np

m.patch()

logger = logging.getLogger("rats.libero_remote")


# RPCs that do not advance the sim and are safe to retry once after a
# transparent reconnect. Stateful RPCs (`step`, `reset`, `init`, `close`) raise
# on first failure so a silent retry can't double-step / re-init mid-iteration.
_IDEMPOTENT_RPCS: frozenset[str] = frozenset({
    "ping",
    "get_observation",
    "_get_observation",
    "compute_reward",
    "render",
    "render_wrist",
    "get_video_frame_count",
    "get_video_frames",
    "get_video_frames_range",
    "get_wrist_video_frames",
    "get_wrist_video_frames_range",
    "get_api_metadata",
    "get_oracle_code",
})


# ------------------------------------------------------------------
# Wire protocol (mirrors mlspaces_server.py)
# ------------------------------------------------------------------

def _encode(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def _decode(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=False)


def _send(sock: socket.socket, obj: dict) -> None:
    payload = _encode(obj)
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)


def _recvall(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv(sock: socket.socket) -> dict:
    header = _recvall(sock, 4)
    if header is None:
        raise ConnectionError("Server disconnected")
    (msg_len,) = struct.unpack("!I", header)
    payload = _recvall(sock, msg_len)
    if payload is None:
        raise ConnectionError("Server disconnected mid-message")
    return _decode(payload)


# ------------------------------------------------------------------
# API proxy — duck-types what runtime.py / libero_utils.py read off env._apis
# ------------------------------------------------------------------

class _RemoteApiProxy:
    """Stand-in for a server-side ``ApiBase``. The real callables run inside
    cap-x-geniac's env (server-side ``exec``); the agent only needs the doc
    string and the function *names* to build its policy-writer prompt."""

    def __init__(self, doc: str, function_names: list[str]) -> None:
        self._doc = doc
        self._function_names = list(function_names)

    def combined_doc(self) -> str:
        return self._doc

    def functions(self) -> dict[str, Any]:
        # Only the keys are ever read (runtime.py:57 / libero_utils.py).
        return {name: None for name in self._function_names}


# ------------------------------------------------------------------
# Remote env
# ------------------------------------------------------------------

class FrankaLiberoRemoteEnv:
    """RPC client mirroring the ``FrankaLiberoCodeEnv`` call surface.

    Connects to ``capx.serving.libero_executor_server`` and serialises every
    method call as a msgpack request; the server owns the real LIBERO sim +
    APIs + perception and executes ``step(code)`` there.
    """

    def __init__(
        self,
        remote_env_url: str | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 9200,
        connect_timeout: float = 60.0,
        rpc_timeout: float | None = None,
        init_kwargs: dict[str, Any] | None = None,
        max_reconnect_attempts: int | None = None,
        reconnect_backoff_seconds: float | None = None,
        cfg: Any = None,
        **_ignored: Any,
    ) -> None:
        if remote_env_url:
            host_part, _, port_part = remote_env_url.rpartition(":")
            if host_part:
                host = host_part
            if port_part:
                port = int(port_part)
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._connect_timeout = connect_timeout
        self._rpc_timeout = self._resolve_rpc_timeout(rpc_timeout)
        self._init_kwargs: dict[str, Any] | None = dict(init_kwargs) if init_kwargs else None
        self._max_reconnect_attempts = self._resolve_int_env(
            max_reconnect_attempts, env="RATS_LIBERO_REMOTE_MAX_RECONNECT", default=5,
        )
        self._reconnect_backoff = self._resolve_float_env(
            reconnect_backoff_seconds, env="RATS_LIBERO_REMOTE_RECONNECT_BACKOFF", default=1.0,
        )

        # Duck-typed attributes the RATS loop reads directly.
        self.cfg = cfg
        self.oracle_code = None

        self._connect()
        if self._init_kwargs:
            self._call("init", **self._init_kwargs)
        # Fetch the server's real API surface once, for prompt building.
        self._apis: dict[str, _RemoteApiProxy] = self._fetch_apis()

    # -- env-var / timeout helpers (mirror molmospaces_remote) ------------

    @staticmethod
    def _resolve_rpc_timeout(explicit: float | None) -> float | None:
        if explicit is not None:
            return explicit
        raw = os.environ.get("RATS_LIBERO_REMOTE_RPC_TIMEOUT", "300")
        if raw.strip().lower() in {"", "none", "0", "false", "off"}:
            return None
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid RATS_LIBERO_REMOTE_RPC_TIMEOUT=%r; using 300s", raw)
            return 300.0

    @staticmethod
    def _resolve_int_env(explicit: int | None, *, env: str, default: int) -> int:
        if explicit is not None:
            return int(explicit)
        raw = os.environ.get(env, "")
        if not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %d", env, raw, default)
            return default

    @staticmethod
    def _resolve_float_env(explicit: float | None, *, env: str, default: float) -> float:
        if explicit is not None:
            return float(explicit)
        raw = os.environ.get(env, "")
        if not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%r; using %.2fs", env, raw, default)
            return default

    # -- connection management --------------------------------------------

    def _connect(self) -> None:
        logger.info("Connecting to libero executor server at %s:%d ...", self._host, self._port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout)
        sock.connect((self._host, self._port))
        sock.settimeout(self._rpc_timeout)
        self._sock = sock
        logger.info("Connected to libero executor server")

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_reconnect_attempts + 1):
            try:
                self._connect()
            except (OSError, socket.timeout) as exc:
                last_exc = exc
                delay = self._reconnect_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "libero-remote reconnect %d/%d failed (%s: %s); retrying in %.1fs",
                    attempt, self._max_reconnect_attempts, type(exc).__name__, exc, delay,
                )
                self._close_socket()
                if attempt < self._max_reconnect_attempts:
                    time.sleep(delay)
                continue
            logger.info("Auto-reconnected to libero executor server (attempt %d)", attempt)
            return
        raise ConnectionError(
            f"Failed to reconnect to libero executor server at {self._host}:{self._port} "
            f"after {self._max_reconnect_attempts} attempts"
        ) from last_exc

    def _call(self, method: str, **kwargs: Any) -> Any:
        self._ensure_connected()
        try:
            _send(self._sock, {"method": method, "kwargs": kwargs})  # type: ignore[arg-type]
            resp = _recv(self._sock)  # type: ignore[arg-type]
        except socket.timeout as exc:
            self._close_socket()
            if method in _IDEMPOTENT_RPCS:
                logger.warning("libero-remote RPC %s timed out; reconnect+retry once", method)
                self._ensure_connected()
                _send(self._sock, {"method": method, "kwargs": kwargs})  # type: ignore[arg-type]
                resp = _recv(self._sock)  # type: ignore[arg-type]
            else:
                raise TimeoutError(
                    f"Timed out waiting for libero server response to {method} "
                    f"after {self._rpc_timeout}s"
                ) from exc
        except (ConnectionError, OSError) as exc:
            self._close_socket()
            if method in _IDEMPOTENT_RPCS:
                logger.warning(
                    "libero-remote RPC %s lost connection (%s: %s); reconnect+retry once",
                    method, type(exc).__name__, exc,
                )
                self._ensure_connected()
                _send(self._sock, {"method": method, "kwargs": kwargs})  # type: ignore[arg-type]
                resp = _recv(self._sock)  # type: ignore[arg-type]
            else:
                raise
        if resp.get("status") == "error":
            raise RuntimeError(f"libero server error ({method}): {resp.get('message')}")
        return resp.get("result")

    def _fetch_apis(self) -> dict[str, _RemoteApiProxy]:
        meta = self._call("get_api_metadata") or {}
        apis: dict[str, _RemoteApiProxy] = {}
        for name, entry in meta.items():
            apis[name] = _RemoteApiProxy(
                entry.get("doc", ""), entry.get("functions", []),
            )
        return apis

    # -- FrankaLiberoCodeEnv-compatible surface ---------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Realign request/response with a fresh socket before a new episode.
        self._close_socket()
        result = self._call("reset", seed=seed, options=options)
        return result["obs"], result["info"]

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        result = self._call("step", code=action)
        return (
            result["obs"],
            float(result["reward"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            result["info"],
        )

    def _get_observation(self) -> dict[str, Any]:
        return self._call("_get_observation")

    def get_observation(self) -> dict[str, Any]:
        return self._call("get_observation")

    def compute_reward(self) -> float:
        return float(self._call("compute_reward"))

    def render(self, mode: str = "rgb_array"):
        frame = self._call("render", mode=mode)
        return None if frame is None else np.asarray(frame)

    def render_wrist(self) -> np.ndarray | None:
        frame = self._call("render_wrist")
        return None if frame is None else np.asarray(frame)

    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = False, wrist_camera: bool = False,
    ) -> None:
        self._call("enable_video_capture", enabled=enabled, clear=clear, wrist_camera=wrist_camera)

    def get_video_frame_count(self) -> int:
        return int(self._call("get_video_frame_count"))

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = self._call("get_video_frames", clear=clear) or []
        return [np.asarray(f) for f in frames]

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        frames = self._call("get_video_frames_range", start=int(start), end=int(end)) or []
        return [np.asarray(f) for f in frames]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = self._call("get_wrist_video_frames", clear=clear) or []
        return [np.asarray(f) for f in frames]

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        frames = self._call("get_wrist_video_frames_range", start=int(start), end=int(end)) or []
        return [np.asarray(f) for f in frames]

    def close(self) -> None:
        try:
            self._call("close")
        except Exception:
            logger.warning("Remote close failed", exc_info=True)
        finally:
            self._close_socket()

    # Consumers use `getattr(env, "low_level_env", env)`; returning self keeps
    # video/observation passthrough working while low-level-only methods
    # (state trace, describe_scene_inventory) stay absent → getattr-guarded skip.
    @property
    def low_level_env(self):
        return self
