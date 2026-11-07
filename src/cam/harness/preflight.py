"""Synthetic 600-frame transport check through the real isolated Python worker."""

# Adapted from the CAP streamed observation check; see NOTICE.
import json
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
import numpy as np
from cam.harness.isolation import command, limits, terminate
from cam.runtime.wire import observation_fields, pack, send, receive

PROTOCOL = "streamed-v1"
FRAME_COUNT = 600


def synthetic_frame(include_depth=False):
    values = {
        "front_rgb": np.full((256, 256, 3), 23, dtype=np.uint8),
        "wrist_rgb": np.full((256, 256, 3), 71, dtype=np.uint8),
        "front_K": np.eye(3, dtype=np.float32),
        "wrist_K": np.eye(3, dtype=np.float32),
        "front_extrinsic": np.zeros((3, 4), dtype=np.float32),
        "wrist_extrinsic": np.zeros((3, 4), dtype=np.float32),
        "tcp": np.zeros(6, dtype=np.float64),
        "joints": np.zeros(7, dtype=np.float32),
        "fingers": np.zeros(2, dtype=np.float32),
    }
    if include_depth:
        values.update(
            front_depth=np.full((256, 256), 1234, dtype=np.int16),
            wrist_depth=np.full((256, 256), 567, dtype=np.int16),
        )
    assert set(values) == set(observation_fields(include_depth))
    return {
        "phase": "demo",
        "step": 0,
        **{key: pack(value) for key, value in values.items()},
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


def transport_preflight(runtime, parent=None, timeout=45, include_depth=False):
    """Transfer >256 MiB through the real isolated worker using bounded messages."""
    runtime = Path(runtime).resolve()
    started = time.monotonic()
    frame = synthetic_frame(include_depth)
    message = {"kind": "frame", "index": FRAME_COUNT - 1, "frame": frame}
    message_bytes = len(json.dumps(message, separators=(",", ":")).encode()) + 1
    # The old one-message reset would exceed the decoder's 256 MiB limit.
    aggregate_bytes = FRAME_COUNT * message_bytes
    assert aggregate_bytes > 256 * 1024**2
    report = {
        "schema": "cam-transport-preflight/1",
        "passed": False,
        "protocol": PROTOCOL,
        "frames": FRAME_COUNT,
        "image_shape": [256, 256, 3],
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
            json.dumps({"task": "BinFill", "runtime": "python", "memory": []})
        )
        (workspace / "policy.md").write_text(
            "Synthetic transport check, no task policy.\n"
        )
        depth_check = (
            """        if frame.front_depth.shape != (256,256) or frame.wrist_depth.shape != (256,256):
            raise ValueError('Aligned depth resolution changed')
        if frame.front_depth.dtype != np.dtype('int16') or frame.wrist_depth.dtype != np.dtype('int16'):
            raise ValueError('Depth units/storage changed')
        if frame.front_depth[0,0] != 1234 or frame.wrist_depth[0,0] != 567:
            raise ValueError('Depth values changed')
"""
            if include_depth
            else ""
        )
        (
            workspace / "tools/policy.py"
        ).write_text(f'''def perceive_decide(obs, goal, memory, last):
    """Verify complete reset observations cross the isolation boundary."""
    if len(obs.frames) != {FRAME_COUNT}:
        raise ValueError('Reset frames were dropped')
    for index, frame in enumerate(obs.frames):
        final = index == {FRAME_COUNT - 1}
        if frame.phase != ('exec' if final else 'demo') or frame.step != (0 if final else index):
            raise ValueError('Reset frames were changed or reordered')
        if frame.front_rgb.shape != (256,256,3) or frame.wrist_rgb.shape != (256,256,3):
            raise ValueError('RGB resolution was changed')
        if frame.front_rgb[0,0,0] != 23 or frame.wrist_rgb[0,0,0] != 71:
            raise ValueError('Camera observations were changed')
{depth_check}\
    return {{"memory": {{}}, "act": {{"tool": "act_observe", "args": {{}}}}}}

def act_observe(robot):
    """Send one synthetic action, never a simulator action."""
    robot.step(*robot.now().tcp, 1.0)
''')
        try:
            process = subprocess.Popen(
                command(
                    workspace,
                    runtime,
                    ["python3.11", "-m", "cam.runtime.policy_worker"],
                    readonly=True,
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
                if not selector.select(timeout=20):
                    raise TimeoutError(
                        "Worker did not send streamed-v1 readiness; legacy runtime may be installed"
                    )
            response = receive(process.stdout, 2 * 1024**2)
            if response != {"kind": "ready"}:
                report["worker_response"] = response
                raise ValueError("Worker failed before reset transfer")
            stage = "initial_observation_transfer"

            def initial_frames():
                for index in range(FRAME_COUNT):
                    frame["phase"] = "exec" if index == FRAME_COUNT - 1 else "demo"
                    frame["step"] = 0 if frame["phase"] == "exec" else index
                    yield frame

            stream_initial(
                process.stdin, "synthetic transport test", initial_frames(), FRAME_COUNT
            )
            stage = "policy_response"
            response = receive(process.stdout, 2 * 1024**2)
            if response.get("kind") != "decision":
                report["worker_response"] = response
                raise ValueError(
                    "Worker did not accept the complete reset observations"
                )
            action = receive(process.stdout, 2 * 1024**2)
            if action.get("kind") != "step" or len(action.get("action", [])) != 7:
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
