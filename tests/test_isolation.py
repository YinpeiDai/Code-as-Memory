import pytest
from cam.harness.isolation import run
from cam.harness.preflight import transport_preflight
from cam.harness.prepare import make_runtime


@pytest.mark.isolation
def test_only_assigned_files_visible(run_factory, tmp_path):
    root = run_factory()
    task = root / "tasks/BinFill"
    secret = tmp_path / "hidden.txt"
    secret.write_text("not observable")
    script = f"""import os, socket, json
import h5py, numpy, scipy, PIL, cv2
assert not os.path.exists({str(secret)!r})
assert not os.path.exists({str(root / "inputs/metadata/test")!r})
assert not os.path.exists({str(root / "inputs/benchmark")!r})
assert not os.path.exists({str(root / "inputs/harness")!r})
assert not os.path.exists({str(task / "state")!r})
assert not os.path.exists('/workspace/inputs')
assert not os.path.exists('/workspace/api_reference/benchmarks')
assert not os.path.exists('/home/daiyp/robomme')
assert not os.environ.get('AWS_PROFILE')
assert not os.environ.get('OPENAI_API_KEY')
assert os.path.exists('/demonstrations/record_dataset_BinFill.h5')
try:
    socket.create_connection(('1.1.1.1', 443), timeout=1)
    raise AssertionError('network was available')
except OSError:
    pass
print('isolated')
"""
    code, out, err = run(
        task / "workspace",
        root / "runtime",
        ["python3.11", "-c", script],
        demonstrations=task / "demonstrations",
    )
    assert code == 0, (out + err).decode()
    assert b"isolated" in out
    code, out, err = run(
        task / "program",
        root / "runtime",
        [
            "python3",
            "-c",
            "import os; assert not any(os.path.exists(p) for p in "
            "('/demonstrations','/feedback','/workspace/scratch','/workspace/api_reference')); print('policy isolated')",
        ],
        readonly=True,
    )
    assert code == 0, (out + err).decode()


@pytest.mark.isolation
def test_analysis_python_aliases_use_pinned_interpreter(run_factory):
    root = run_factory()
    for executable in ("python", "python3", "python3.11"):
        code, out, err = run(
            root / "tasks/BinFill/workspace",
            root / "runtime",
            [
                executable,
                "-c",
                "import sys,h5py,numpy; assert sys.version_info[:2] == (3,11); print('ready')",
            ],
        )
        assert code == 0, (out + err).decode()
        assert b"ready" in out


def test_mcp_lifetime_follows_process_not_spawning_thread():
    import os
    import select
    import signal
    import subprocess
    import sys

    child_code = (
        "from cam.harness.isolation import watch_parent_process; import time; "
        "watch_parent_process(); print('ready',flush=True); time.sleep(60)"
    )
    parent_code = f"""import subprocess,sys,threading
children=[]
def launch():
    child=subprocess.Popen([sys.executable,'-u','-c',{child_code!r}],stdout=subprocess.PIPE,text=True)
    assert child.stdout.readline().strip() == 'ready'
    children.append(child)
thread=threading.Thread(target=launch)
thread.start(); thread.join()
print(children[0].pid,flush=True)
sys.stdin.read(1)
"""
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", parent_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    child = descriptor = None
    try:
        child = int(parent.stdout.readline())
        descriptor = os.pidfd_open(child)
        assert not select.select([descriptor], [], [], 0.2)[0], (
            "launch thread exit killed MCP"
        )
        parent.stdin.close()
        parent.wait(timeout=5)
        assert select.select([descriptor], [], [], 5)[0], (
            "MCP outlived its client process"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if descriptor is not None:
            os.close(descriptor)


@pytest.mark.isolation
def test_long_video_rgbd_transport(tmp_path):
    runtime = make_runtime(tmp_path / "runtime")
    report = transport_preflight(runtime, include_depth=True, timeout=120)
    assert report["passed"], report
    assert report["frames"] == 600
    assert report["aggregate_frame_message_bytes"] > 500 * 1024**2
    assert report["simulator_resets"] == report["api_calls"] == 0
