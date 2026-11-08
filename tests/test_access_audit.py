"""Audit parsing must cover split syscall records without counting host setup."""

import importlib.util
from pathlib import Path


def test_split_syscalls_keep_sandbox_descendants_and_exclude_bootstrap(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "audit_run", Path(__file__).parents[1] / "scripts/audit_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = tmp_path / "call.strace"
    trace.write_text('''100 openat(AT_FDCWD, "/data/private/benchmark.py", O_RDONLY) = 3
100 clone(child_stack=NULL, flags=SIGCHLD <unfinished ...>
100 <... clone resumed>) = 101
101 execve("/python/bin/bash", ["bash", "--noprofile", "--norc", "-c", "python script.py"], 0x123 <unfinished ...>
101 <... execve resumed>) = 0
101 chdir("/workspace/scratch") = 0
101 clone(child_stack=NULL, flags=SIGCHLD <unfinished ...>
102 openat(AT_FDCWD, "script.py", O_RDONLY) = 3
101 <... clone resumed>) = 102
102 openat(AT_FDCWD, "/data/private/secret.py", O_RDONLY <unfinished ...>
102 <... openat resumed>) = 4
102 socket(AF_INET, SOCK_STREAM, IPPROTO_IP <unfinished ...>
102 <... socket resumed>) = 5
102 openat(AT_FDCWD, "/home/daiyp/hidden", O_RDONLY) = -1 ENOENT (No such file or directory)
''')
    result = module.trace_summary(trace)
    assert result["sandbox_shell_identified"]
    assert result["sandbox_process_count"] == 2
    assert len(result["internet_socket_events"]) == 1
    assert len(result["successful_absolute_private_path_opens"]) == 1
    assert "secret.py" in result["successful_absolute_private_path_opens"][0]
    assert result["successful_relative_open_names"] == {"script.py": 1}
    assert len(result["successful_directory_changes"]) == 1
