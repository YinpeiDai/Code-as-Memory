from pathlib import Path
import runpy


def test_trace_audit_separates_bootstrap_and_sandbox_subprocesses(tmp_path):
    summarize = runpy.run_path(str(Path(__file__).parents[1] / "scripts/audit_run.py"))[
        "trace_summary"
    ]
    path = tmp_path / "call_000001.strace"
    path.write_text("""100 execve("/usr/bin/bwrap", ["bwrap"], 0x0) = 0
100 openat(AT_FDCWD, "/home/daiyp/CaM/inputs/host-mount", O_RDONLY) = 3
100 clone(child_stack=NULL, flags=SIGCHLD) = 101
101 execve("/python/bin/bash", ["bash", "--noprofile", "--norc", "-c", "command"], 0x0) = 0
101 openat(AT_FDCWD, "/workspace/tools/policy.py", O_RDONLY) = 3
101 clone(child_stack=NULL, flags=SIGCHLD) = 102
102 execve("/tools/python", ["python", "check.py"], 0x0) = 0
102 openat(AT_FDCWD, "/inputs/private.json", O_RDONLY) = -1 ENOENT (No such file or directory)
102 openat(AT_FDCWD, "/data/private.h5", O_RDONLY) = 4
102 socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 5
102 socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP) = 6
102 connect(7, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT (No such file or directory)
""")
    result = summarize(path)
    assert result["sandbox_shell_identified"]
    assert result["sandbox_process_count"] == 2
    assert len(result["internet_socket_events"]) == 2
    assert len(result["successful_absolute_private_path_opens"]) == 1
    assert "/data/private.h5" in result["successful_absolute_private_path_opens"][0]
    assert "/home" not in result["successful_absolute_open_prefixes"]
