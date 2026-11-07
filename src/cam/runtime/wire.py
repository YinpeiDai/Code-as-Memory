# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""Explicit array transport. Never deserialize pickle across the policy boundary."""

import base64
import json
import numpy as np

FIELDS = (
    "front_rgb",
    "wrist_rgb",
    "front_K",
    "wrist_K",
    "front_extrinsic",
    "wrist_extrinsic",
    "tcp",
    "joints",
    "fingers",
)
DEPTH_FIELDS = ("front_depth", "wrist_depth")


def observation_fields(include_depth=False):
    """Missing depth configuration preserves the historical RGB contract."""
    return FIELDS + (DEPTH_FIELDS if include_depth else ())


def pack(array):
    a = np.ascontiguousarray(array)
    return {
        "dtype": a.dtype.str,
        "shape": list(a.shape),
        "data": base64.b64encode(a.tobytes()).decode(),
    }


def encoded(frames):
    return [
        {
            key: pack(value) if key in FIELDS + DEPTH_FIELDS else value
            for key, value in frame.items()
        }
        for frame in frames
    ]


def unpack(obj):
    dtype = np.dtype(obj["dtype"])
    if dtype.kind not in "buif":
        raise ValueError("Unsupported array dtype")
    a = np.frombuffer(
        base64.b64decode(obj["data"], validate=True), dtype=dtype
    ).reshape(obj["shape"])
    a.setflags(write=False)
    return a


def send(stream, obj):
    data = memoryview(
        json.dumps(obj, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    )
    while data:
        n = stream.write(data)
        if not n:
            raise BrokenPipeError("RPC channel closed")
        data = data[n:]
    stream.flush()


def receive(stream, limit=256 * 1024**2):
    line = stream.readline(limit + 1)
    if not line or len(line) > limit:
        raise ValueError("Missing or oversized RPC message")
    return json.loads(line)
