"""Synthetic long-reset transport check through the real isolated Python worker."""

# Adapted from the CAP streamed observation check; see NOTICE.
import json
import math
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
from cam.interface.base import synthetic_value, zero_action
from cam.harness.isolation import command, limits, terminate
from cam.runtime.wire import observation_fields, pack, send, receive

PROTOCOL = "streamed-v1"
FRAME_COUNT = 600
MINIMUM_BYTES = 256 * 1024**2


def legacy_interface(include_depth=False):
    """RoboMME's 256x256 sensor set, used when no benchmark interface is given."""
    from cam.interface.base import action, sensor

    sensors = {
        "front_rgb": sensor("uint8", (256, 256, 3), "front RGB"),
        "wrist_rgb": sensor("uint8", (256, 256, 3), "wrist RGB"),
        "front_K": sensor("float32", (3, 3), "front intrinsics"),
        "wrist_K": sensor("float32", (3, 3), "wrist intrinsics"),
        "front_extrinsic": sensor("float32", (3, 4), "front extrinsics"),
        "wrist_extrinsic": sensor("float32", (3, 4), "wrist extrinsics"),
        "tcp": sensor("float64", (6,), "TCP"),
        "joints": sensor("float32", (7,), "joints"),
        "fingers": sensor("float32", (2,), "fingers"),
    }
    if include_depth:
        sensors["front_depth"] = sensor("int16", (256, 256), "front depth")
        sensors["wrist_depth"] = sensor("int16", (256, 256), "wrist depth")
    assert set(sensors) == set(observation_fields(include_depth))
    return {
        "sensors": sensors,
        "action": action(
            ("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
            "absolute TCP",
            low=[None] * 6 + [-1.0],
            high=[None] * 6 + [1.0],
        ),
    }


def synthetic_values(interface):
    """Distinct constant per sensor: (name, array) in declared order."""
    return {
        name: synthetic_value(spec, 23 + 48 * index)
        for index, (name, spec) in enumerate(interface["sensors"].items())
    }


def stream_initial(writer, goal, frames, frame_count, limits=None, action_spec=None):
    """Send encoded reset frames without constructing a video-sized message.

    Shared by the real simulator broker and its synthetic transport check.
    Frames can be a generator, keeping encoding memory bounded to one frame.
    """
    if not 1 <= frame_count <= 100000:
        raise ValueError("Invalid initial observation count")
    send(
        writer,
        {
            "kind": "initial",
            "goal": goal,
            "frame_count": frame_count,
            "limits": limits or {},
            "action_spec": action_spec or {},
        },
    )
    sent = 0
    for index, frame in enumerate(frames):
        if index >= frame_count:
            raise ValueError("Too many initial observation frames")
        send(writer, {"kind": "frame", "index": index, "frame": frame})
        sent += 1
    if sent != frame_count:
        raise ValueError("Missing initial observation frames")


def transport_preflight(
    runtime, parent=None, timeout=45, include_depth=False, interface=None, python=None
):
    """Transfer >256 MiB through the real isolated worker using bounded messages."""
    runtime = Path(runtime).resolve()
    started = time.monotonic()
    interface = interface or legacy_interface(include_depth)
    values = synthetic_values(interface)
    packed = {key: pack(value) for key, value in values.items()}
    frame = {"phase": "demo", "step": 0, **packed}
    message_bytes = (
        len(
            json.dumps(
                {"kind": "frame", "index": FRAME_COUNT - 1, "frame": frame},
                separators=(",", ":"),
            ).encode()
        )
        + 1
    )
    # The old one-message reset would exceed the decoder's 256 MiB limit.
    frame_count = min(100000, max(FRAME_COUNT, math.ceil(MINIMUM_BYTES / message_bytes) + 1))
    aggregate_bytes = frame_count * message_bytes
    expected = {
        name: [list(value.shape), value.reshape(-1)[0].item()]
        for name, value in values.items()
    }
    spec = interface["action"]
    report = {
        "schema": "cam-transport-preflight/2",
        "passed": False,
        "protocol": PROTOCOL,
        "frames": frame_count,
        "sensors": {name: shape for name, (shape, _) in expected.items()},
        "encoded_frame_message_bytes": message_bytes,
        "aggregate_frame_message_bytes": aggregate_bytes,
        "simulator_resets": 0,
        "training_attempts": 0,
        "api_calls": 0,
        "synthetic_observations": True,
        "depth": include_depth,
    }
    process = timer = None
    expired = threading.Event()
    stage = "worker_initialization"
    with tempfile.TemporaryDirectory(
        prefix=".cam_transport_preflight_", dir=parent
    ) as name:
        workspace = Path(name)
        (workspace / "tools").mkdir()
        (workspace / "task.json").write_text(
            json.dumps({"task": "TransportPreflight", "runtime": "python", "memory": []})
        )
        (workspace / "policy.md").write_text(
            "Synthetic transport check, no task policy.\n"
        )
        (workspace / "tools/policy.py").write_text(f'''EXPECTED = {expected!r}
ZERO = {zero_action(spec)!r}


def perceive_decide(obs, goal, memory, last):
    """Verify complete reset observations cross the isolation boundary."""
    if len(obs.frames) != {frame_count}:
        raise ValueError('Reset frames were dropped')
    for index, frame in enumerate(obs.frames):
        final = index == {frame_count - 1}
        if frame.phase != ('exec' if final else 'demo') or frame.step != (0 if final else index):
            raise ValueError('Reset frames were changed or reordered')
        if sorted(frame.sensors) != sorted(EXPECTED):
            raise ValueError('Sensor set changed')
        for name in EXPECTED:
            value = getattr(frame, name)
            if list(value.shape) != EXPECTED[name][0]:
                raise ValueError('Sensor shape changed: ' + name)
            if value.reshape(-1)[0] != EXPECTED[name][1]:
                raise ValueError('Sensor values changed: ' + name)
    return {{"memory": {{}}, "act": {{"tool": "act_observe", "args": {{}}}}}}


def act_observe(robot):
    """Send one synthetic action, never a simulator action."""
    robot.step(*ZERO)
''')
        try:
            process = subprocess.Popen(
                command(
                    workspace,
                    runtime,
                    ["python3.11", "-m", "cam.runtime.policy_worker"],
                    readonly=True,
                    python=python,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=limits,
                bufsize=0,
            )

            def stop_stalled_worker():
                expired.set()
                terminate(process)

            timer = threading.Timer(timeout, stop_stalled_worker)
            timer.daemon = True
            timer.start()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=timeout):
                    raise TimeoutError(
                        "Worker did not send streamed-v1 readiness within the preflight time limit"
                    )
            response = receive(process.stdout, 2 * 1024**2)
            if response != {"kind": "ready"}:
                report["worker_response"] = response
                raise ValueError("Worker failed before reset transfer")
            stage = "initial_observation_transfer"

            def initial_frames():
                for index in range(frame_count):
                    frame["phase"] = "exec" if index == frame_count - 1 else "demo"
                    frame["step"] = 0 if frame["phase"] == "exec" else index
                    yield frame

            stream_initial(
                process.stdin,
                "synthetic transport test",
                initial_frames(),
                frame_count,
                action_spec=spec,
            )
            stage = "policy_response"
            response = receive(process.stdout, 2 * 1024**2)
            if response.get("kind") != "decision":
                report["worker_response"] = response
                raise ValueError(
                    "Worker did not accept the complete reset observations"
                )
            action = receive(process.stdout, 2 * 1024**2)
            if action.get("kind") != "step" or len(action.get("action", [])) != len(
                spec["names"]
            ):
                raise ValueError("Worker did not return a valid synthetic action")
            send(process.stdin, {"frames": [frame], "status": "success"})
            if receive(process.stdout, 2 * 1024**2) != {
                "kind": "done",
                "status": "success",
            }:
                raise ValueError("Worker failed the synthetic terminal handshake")
            if process.wait(timeout=10):
                raise ValueError("Worker exited with an error")
            report["passed"] = True
        except Exception as exc:
            error = (
                f"TimeoutError: Transport check exceeded {timeout}s"
                if expired.is_set()
                else f"{type(exc).__name__}: {exc}"
            )
            report.update(stage=stage, error=error)
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            if process is not None:
                terminate(process)
                report["worker_stderr"] = process.stderr.read().decode(
                    errors="replace"
                )[-4000:]
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
            report["seconds"] = time.monotonic() - started
    return report
