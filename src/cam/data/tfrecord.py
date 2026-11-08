"""Minimal TFRecord + ``tf.train.Example`` reader; no TensorFlow dependency.

Record framing: uint64 length, uint32 masked CRC of the length, payload, uint32
masked CRC of the payload. CRCs are not recomputed here; whole-file integrity is
checked against the release's SHA-256 (Hugging Face LFS object id) instead.
"""

import struct

import numpy as np


def records(path):
    with open(path, "rb") as stream:
        while True:
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header: {path}")
            (length,) = struct.unpack("<Q", header[:8])
            payload = stream.read(length)
            if len(payload) != length or len(stream.read(4)) != 4:
                raise ValueError(f"Truncated TFRecord payload: {path}")
            yield payload


def _varint(buffer, position):
    result = shift = 0
    while True:
        byte = buffer[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7


def _fields(buffer):
    """Yield (field number, wire type, value) for a protobuf message."""
    position, end = 0, len(buffer)
    while position < end:
        key, position = _varint(buffer, position)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, position = _varint(buffer, position)
        elif wire == 1:
            value, position = buffer[position : position + 8], position + 8
        elif wire == 2:
            length, position = _varint(buffer, position)
            value, position = buffer[position : position + length], position + length
        elif wire == 5:
            value, position = buffer[position : position + 4], position + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def _feature(buffer):
    for number, _, value in _fields(buffer):
        if number == 1:  # BytesList
            return [bytes(v) for n, _, v in _fields(value) if n == 1]
        if number == 2:  # FloatList, packed or not
            parts = []
            for n, wire, v in _fields(value):
                if n == 1:
                    parts.append(
                        np.frombuffer(v, "<f4") if wire == 2 else np.frombuffer(v, "<f4")
                    )
            return np.concatenate(parts) if parts else np.zeros(0, np.float32)
        if number == 3:  # Int64List, packed or not
            values = []
            for n, wire, v in _fields(value):
                if n != 1:
                    continue
                if wire == 2:
                    position = 0
                    while position < len(v):
                        item, position = _varint(v, position)
                        values.append(item)
                else:
                    values.append(v)
            values = [x - (1 << 64) if x >= 1 << 63 else x for x in values]
            return np.asarray(values, dtype=np.int64)
    return []


def example(payload):
    """Decode a serialized ``tf.train.Example`` into {key: list[bytes] | ndarray}."""
    result = {}
    for number, _, features in _fields(memoryview(payload)):
        if number != 1:
            continue
        for n, _, entry in _fields(features):
            if n != 1:
                continue
            key = value = None
            for m, _, part in _fields(entry):
                if m == 1:
                    key = bytes(part).decode()
                elif m == 2:
                    value = _feature(part)
            if key is not None:
                result[key] = value if value is not None else []
    return result
