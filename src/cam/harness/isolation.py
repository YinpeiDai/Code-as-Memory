# Adapted for Code-as-Memory from the Apache-2.0 CAP implementation; see NOTICE.
"""OS isolation for coding commands and generated policies; no simulator mount."""

from pathlib import Path
import os
import resource
import signal
import subprocess
import sys
import ctypes
import threading
import select

PYTHON = Path(sys.executable).resolve().parent.parent
PACKAGES = Path(sys.prefix) / "lib/python3.11/site-packages"


def policy_python():
    """This interpreter's sandbox mounts; recorded at preparation for all later workers.

    Benchmark simulators may run in their own environments, but coding commands and
    generated policies always use the CaM Python 3.11 environment that prepared the run.
    """
    return {"base": str(PYTHON), "packages": str(PACKAGES), "prefix": sys.prefix}
DEPENDENCIES = (
    "numpy",
    "numpy.libs",
    "scipy",
    "scipy.libs",
    "PIL",
    "pillow.libs",
    "cv2",
    "opencv_python.libs",
    "opencv_python_headless.libs",
    "h5py",
    "h5py.libs",
    "imageio",
    "imageio_ffmpeg",
)


def command(
    workspace,
    runtime,
    argv,
    feedback=None,
    readonly=False,
    demonstrations=None,
    python=None,
):
    """Private PID/network/mount namespaces, zero inherited credentials or sockets."""
    base = Path(python["base"]) if python else PYTHON
    packages = Path(python["packages"]) if python else PACKAGES
    args = [
        "/usr/bin/bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--cap-drop",
        "ALL",
    ]
    for path in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--ro-bind",
        str(base),
        "/python",
        "--dir",
        "/packages",
        "--dir",
        "/tools",
    ]
    for name in ("python", "python3", "python3.11"):
        args += ["--symlink", "/python/bin/python3.11", "/tools/" + name]
    for name in DEPENDENCIES:
        if (packages / name).exists():
            args += ["--ro-bind", str(packages / name), "/packages/" + name]
    args += [
        "--ro-bind",
        str(runtime),
        "/runtime",
        "--ro-bind" if readonly else "--bind",
        str(workspace),
        "/workspace",
    ]
    if feedback:
        args += ["--ro-bind", str(feedback), "/feedback"]
    if demonstrations:
        args += ["--ro-bind", str(demonstrations), "/demonstrations"]
    for k, v in {
        "PATH": "/tools:/python/bin:/usr/bin:/bin",
        "PYTHONHOME": "/python",
        "PYTHONPATH": "/runtime:/packages",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }.items():
        args += ["--setenv", k, v]
    return args + ["--chdir", "/workspace", "--", *argv]


def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def run(
    workspace,
    runtime,
    argv,
    feedback=None,
    timeout=120,
    readonly=False,
    demonstrations=None,
):
    proc = subprocess.Popen(
        command(workspace, runtime, argv, feedback, readonly, demonstrations),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=limits,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        return 124, out, err + b"\nCommand time limit reached."


def terminate(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def parent_death():
    """Private workers must stop when the process that owns their budget dies."""
    ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0)
    if os.getppid() == 1:
        os._exit(125)


def watch_parent_process():
    """Keep an MCP server tied to its client process, not a short-lived launch thread.

    Linux PR_SET_PDEATHSIG follows the spawning thread. Coding clients can retire
    that thread while remaining alive, so the MCP service watches a process pidfd.
    Call after exec, never from a subprocess preexec_fn.
    """
    parent = os.getppid()
    if parent == 1:
        os._exit(125)
    try:
        descriptor = os.pidfd_open(parent)
    except ProcessLookupError:
        os._exit(125)

    def watch():
        try:
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
        os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=watch, name="cam-client-lifetime", daemon=True).start()
