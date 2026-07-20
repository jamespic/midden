from __future__ import annotations

import argparse
import ctypes
import datetime
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import ExitStack, contextmanager
from http.client import HTTPSConnection
from subprocess import PIPE, Popen
from typing import Literal
from urllib.parse import urlparse

try:
    from sys import remote_exec  # ty: ignore[unresolved-import]
except ImportError:
    remote_exec = None

try:
    from os import setns  # ty: ignore[unresolved-import]
except ImportError:
    setns = None

try:
    import psutil

    _psutil_available = True
except ImportError:
    _psutil_available = False

GIL_ENABLED = True
try:
    from sys import _is_gil_enabled  # ty: ignore[unresolved-import]

    GIL_ENABLED = _is_gil_enabled()
except ImportError:
    pass

DEFAULT_DUMP_FILE = "/tmp/dump.jsonl"
DEFAULT_DUMP_ARCHIVE_FILE = "/tmp/dump.tar.gz"

_FILES_NEEDED_FOR_INJECTION = ["dump_heap.py", "inject.py"]


def _build_tarball_of_dumping_code():
    """Bundle the dump scripts so they can be copied into another namespace."""
    file_obj = io.BytesIO()
    with tarfile.open(fileobj=file_obj, mode="w:gz") as tar:
        for filename in _FILES_NEEDED_FOR_INJECTION:
            file_path = pathlib.Path(__file__).parent / filename
            tar.add(file_path, arcname=filename)
    return file_obj.getvalue()


# We do this at import time so that if we fork into another namespace, it's still available
_TARBALL = _build_tarball_of_dumping_code()


class PlatformNotSupportedError(Exception):
    """Raised when the current platform does not support the injection method being attempted."""


def dump_heap_from_pid(
    pid: PidType,
    output_file=DEFAULT_DUMP_FILE,
    can_use_namespace_injection=True,
    can_use_alternate_python_interpreter=True,
):
    """Dump the heap of a running Python process given its PID."""
    if pid == "all":
        dump_all_python_processes(output_file)
    else:
        _dump_heap_from_pid_possibly_in_namespace(
            pid,
            output_file,
            can_use_namespace_injection,
            can_use_alternate_python_interpreter,
        )


def dump_all_python_processes(output_file=DEFAULT_DUMP_FILE):
    """Dump the heap of all running Python processes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        for pid in _list_all_python_processes():
            pid_specific_output_file = output_dir / f"dump_{pid}.jsonl"
            print(
                f"Dumping heap of process {pid} to {pid_specific_output_file}",
                file=sys.stderr,
            )
            try:
                dump_heap_from_pid(pid, str(pid_specific_output_file))
            except Exception as e:
                print(f"Failed to dump heap of process {pid}: {e}", file=sys.stderr)
        # Archive the dumps
        archive_path = pathlib.Path(output_file)
        with tarfile.open(archive_path, "w:gz") as tar:
            for dump_file in output_dir.iterdir():
                tar.add(dump_file, arcname=dump_file.name)


def _list_all_python_processes() -> list[int]:
    """List all running Python processes."""
    if _psutil_available:
        return _list_all_python_processes_psutil()
    elif sys.platform == "linux":
        return _list_all_python_processes_procfs()
    else:
        raise PlatformNotSupportedError(
            "Listing Python processes is not supported on this platform."
        )


def _list_all_python_processes_psutil() -> list[int]:
    """List all running Python processes using psutil."""
    python_pids = []
    for proc in psutil.process_iter(["pid", "exe", "memory_maps"]):
        try:
            exe = proc.exe()
            if exe and pathlib.Path(exe).name.startswith("python"):
                python_pids.append(proc.info["pid"])
                break
            for m in proc.memory_maps(grouped=False):
                if re.match(r".*lib(python\d\.\d.*)\.so", m.path):
                    python_pids.append(proc.info["pid"])
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return python_pids


def _list_all_python_processes_procfs() -> list[int]:
    """List all running Python processes using /proc filesystem (Linux only)."""
    python_pids = []
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                exe = os.readlink(f"/proc/{pid}/exe")
                if pathlib.Path(exe).name.startswith("python"):
                    python_pids.append(int(pid))
                with open(f"/proc/{pid}/maps") as f:
                    for line in f:
                        if re.match(r".*lib(python\d\.\d.*)\.so", line):
                            python_pids.append(int(pid))
                            break
            except (FileNotFoundError, PermissionError):
                continue
    return python_pids


_DUMP_SCRIPT = (pathlib.Path(__file__).parent / "dump_heap.py").read_text()


def _build_dump_heap_code(output_file):
    """Rewrite the injected script so it writes to the requested output path."""
    return _DUMP_SCRIPT + f"\ndump_heap({output_file!r})\n"


def _dump_heap_from_pid_possibly_using_an_alternate_python_interpreter(
    pid,
    output_file=DEFAULT_DUMP_FILE,
    can_use_alternate_python_interpreter=True,
):
    """Pick a compatible Python interpreter before injecting, if needed."""
    if can_use_alternate_python_interpreter and (
        alternate_python := _should_use_alternate_python_interpreter(pid)
    ):
        print(
            f"Using alternate Python interpreter {alternate_python} for injection",
            file=sys.stderr,
        )
        _dump_heap_from_pid_using_alternate_python_interpreter(
            pid, alternate_python, output_file
        )
    else:
        _dump_heap_from_pid(pid, output_file)


def _dump_heap_from_pid_using_alternate_python_interpreter(
    pid, alternate_python, output_file=DEFAULT_DUMP_FILE
):
    """Run the injector with a different Python executable inside a temp copy of the code."""
    # The target mount namespace may not be able to see our original source tree.
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball_file_obj = io.BytesIO(_TARBALL)
        with tarfile.open(fileobj=tarball_file_obj, mode="r:gz") as tar:
            tar.extractall(path=tmpdir)

        cmd = [
            alternate_python,
            tmpdir + "/inject.py",
            str(pid),
            "--output-file",
            output_file,
            "--no-alternate-python-interpreter",  # Don't recurse into trying to find another alternate Python interpreter
            "--no-namespace-injection",  # Don't use namespace injection - this function should only be called after we've sorted out namespaces
        ]
        print(f"Running injection command: {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True)


def _should_use_alternate_python_interpreter(pid) -> str | None:
    """Return a better-matched Python executable for injection, if one is needed."""
    try:
        exe, maps = _get_exe_and_maps(pid)
    except Exception as e:
        print(
            f"Could not determine target process executable and maps, so proceeding with this interpreter: {e!r}",
            file=sys.stderr,
        )
        return None
    if not _can_this_python_inject(exe):
        if os.path.basename(exe).startswith("python"):
            print(
                f"Using {exe} as alternate Python interpreter based on process exe",
                file=sys.stderr,
            )
            return exe
        else:
            # Check maps for a python library to identify Python version
            for m in maps:
                if match := re.match(
                    r".*lib(python\d\.\d.*)\.so", m
                ):  # Look for a python library in the maps
                    python_exe_wanted = match.group(1)
                    if python_exe_wanted == _this_effective_executable_name():
                        print(
                            f"Mapped library {m} suggests target process is running same Python version, so no alternate interpreter needed",
                            file=sys.stderr,
                        )
                        return None
                    # Check if there's an executable with the same name on the path
                    if python_exe := shutil.which(python_exe_wanted):
                        print(
                            f"Using {python_exe} as alternate Python interpreter based on mapped library {m}",
                            file=sys.stderr,
                        )
                        return python_exe
            print(
                "Couldn't find an alternate Python interpreter to use for injection,"
                " but target process is running a different Python version, so injection may fail",
                file=sys.stderr,
            )


def _get_exe_and_maps(pid):
    """Inspect the target process executable and mapped libraries."""
    if _psutil_available:
        psutil_process = psutil.Process(pid)
        exe = psutil_process.exe()
        try:
            maps = [map.path for map in psutil_process.memory_maps(grouped=False)]
        except Exception:
            maps = []
        return exe, maps
    elif sys.platform == "linux":
        # Linux-only procfs fallback
        exe = os.readlink(f"/proc/{pid}/exe")
        maps = []
        try:
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 6:
                        path = parts[-1]
                        maps.append(path)
        except Exception:  # noqa: S110 # Maps are a nice-to-have, but not critical, so ignore errors
            pass
        return exe, maps
    else:
        raise PlatformNotSupportedError(
            "Can't determine process executable on this platform without psutil"
        )


def _can_this_python_inject(exe):
    """Return whether the current interpreter likely matches the target runtime."""
    return (
        exe == sys.executable
        or _get_effective_executable_name(exe) == _this_effective_executable_name()
    )


def _get_effective_executable_name(exe):
    """Return the versioned Python executable name for a given path, if it looks like Python."""
    basename = os.path.basename(exe)
    match = re.match(r"python\d\.\d.+t?", basename)
    if match:
        return match.group(0)
    if os.path.islink(exe):
        # If it's a symlink, check if the target looks like Python
        target = os.readlink(exe)
        return _get_effective_executable_name(target)
    return None


def _this_effective_executable_name():
    """Return the versioned Python executable name for the current runtime."""

    name = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if GIL_ENABLED:
        return name
    else:
        # For GIL-less Python, the executable is suffixed with -gil0
        return name + "t"


def _dump_heap_from_pid_possibly_in_namespace(
    pid,
    output_file=DEFAULT_DUMP_FILE,
    can_use_namespace_injection=True,
    can_use_alternate_python_interpreter=True,
):
    """Decide whether the dump has to run from inside the target mount namespace."""
    if can_use_namespace_injection and _should_use_namespace(pid):
        print(
            "Target process is in a different mount namespace, using namespace injection method",
            file=sys.stderr,
        )
        _dump_heap_from_pid_in_namespace(
            pid, output_file, can_use_alternate_python_interpreter
        )
    else:
        _dump_heap_from_pid_possibly_using_an_alternate_python_interpreter(
            pid, output_file, can_use_alternate_python_interpreter
        )


def _dump_heap_from_pid_in_namespace(
    pid, output_file=DEFAULT_DUMP_FILE, can_use_alternate_python_interpreter=True
):
    """Copy the dump file out of the target mount namespace after injection."""
    dump_loc_in_namespace = f"/tmp/dump_{pid}.jsonl"
    out = os.open(output_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)

    with _in_namespace(pid) as inner_pid:
        if inner_pid is not None:
            _dump_heap_from_pid_possibly_using_an_alternate_python_interpreter(
                inner_pid, dump_loc_in_namespace, can_use_alternate_python_interpreter
            )
            while not os.path.exists(dump_loc_in_namespace):
                time.sleep(0.1)
            with open(dump_loc_in_namespace) as f, os.fdopen(out, "w") as out_f:
                for line in f:
                    out_f.write(line)
            os.remove(dump_loc_in_namespace)


@contextmanager
def _in_namespace(pid):
    """Fork into the target mount and PID namespaces for the duration of the block."""
    if setns is None:
        raise NotImplementedError(
            "Namespace injection method is not available on this platform"
        )
    inner_pid = _identify_pid_within_namespace(pid)
    if (forked_pid := os.fork()) == 0:
        setns(os.open(f"/proc/{pid}/ns/pid", 0), 0)
        setns(os.open(f"/proc/{pid}/ns/mnt", 0), 0)
        # Must fork again after setns to actually be in the new PID namespace
        if (inner_forked_pid := os.fork()) == 0:
            yield inner_pid
            os._exit(0)
        else:
            yield None
            pid, status = os.waitpid(inner_forked_pid, 0)

        os._exit(status)
    else:
        yield None
        pid, status = os.waitpid(forked_pid, 0)
        if status != 0:
            print(f"Child process failed with status {status}", file=sys.stderr)


def _identify_pid_within_namespace(pid):
    """Translate a host PID into the PID seen inside the target namespace."""
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("NSpid:"):
                return int(line.split()[-1])


def _dump_heap_from_pid(
    pid, output_file=DEFAULT_DUMP_FILE, inactivity_timeout=datetime.timedelta(seconds=5)
):
    """Build the payload script and inject it into the target process.

    Use remote_exec when available, else fall back to gdb."""
    _check_and_warn_about_potential_privileges_issues()
    code = _build_dump_heap_code(output_file)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as script_file:
        script_file.write(code)
        script_file.flush()

        if remote_exec is not None:
            _inject_using_remote_exec(
                pid, script_file.name, output_file, inactivity_timeout
            )

        elif _gdb_available():
            print(
                "remote_exec not available, falling back to gdb method", file=sys.stderr
            )
            _inject_using_gdb(pid, script_file.name)
        else:
            raise PlatformNotSupportedError(
                "No available method to inject code into the target process. On Linux, you may be able to install gdb and try again."
            )


def _check_and_warn_about_potential_privileges_issues():
    """Check if the current user has sufficient privileges to inject into the target process."""
    if sys.platform == "linux":
        try:
            yama_ptrace_scope_path = "/proc/sys/kernel/yama/ptrace_scope"
            if os.path.exists(yama_ptrace_scope_path):
                with open(yama_ptrace_scope_path) as f:
                    yama_ptrace_scope = f.read().strip()
                if yama_ptrace_scope != "0":
                    print(
                        f"Warning: Yama ptrace_scope is set to {yama_ptrace_scope}. If you experience injection problems, try setting this to 0.",
                        file=sys.stderr,
                    )
        except Exception:  # noqa: S110 # Ignore any errors while checking ptrace_scope
            pass

    try:
        if os.geteuid() != 0:
            print(
                "Warning: You are not running as root. If you experience injection problems, try running as root.",
                file=sys.stderr,
            )
    except Exception:  # noqa: S110 # This will probably fail on Windows, so ignore any errors while checking privileges
        pass

    if sys.platform == "win32":
        try:
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print(
                    "Warning: You are not running as Administrator. If you experience injection problems, try running as Administrator.",
                    file=sys.stderr,
                )
        except Exception:  # noqa: S110 # Ignore any errors while checking for admin privileges
            pass


def _inject_using_remote_exec(
    pid: int,
    script_file_name: str,
    output_file: str,
    inactivity_timeout: datetime.timedelta,
):
    """Inject the script into the target process using remote_exec."""
    assert remote_exec is not None, "remote_exec is not available on this platform"
    print("Using remote_exec to inject code", file=sys.stderr)
    remote_exec(pid, script_file_name)
    timeout_time = time.monotonic() + inactivity_timeout.total_seconds()
    partial_file = output_file + ".partial"
    while not os.path.exists(output_file):
        if (
            not os.path.exists(partial_file)
            and not os.path.exists(output_file)
            and time.monotonic() > timeout_time
        ):
            raise PermissionError(
                "Timed out waiting for dump file to be created, injection may have failed"
            )
        time.sleep(0.1)


def _gdb_available() -> bool:
    """Check if gdb is available on the system."""
    try:
        subprocess.run(["gdb", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _inject_using_gdb(pid: int, script_file_name: str):
    """Inject the script into the target process using gdb."""
    gdb_cmds = [
        "(int) PyGILState_Ensure()",
        '(int) PyRun_SimpleString("'
        rf'exec(open(\"{script_file_name}\").read())")',
        "(void) PyGILState_Release($1)",
    ]
    p = Popen(
        [
            "gdb",
            "-p",
            str(pid),
            "--batch",
            *(f"--eval-command=call {cmd}" for cmd in gdb_cmds),
        ],
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    out, err = p.communicate()
    print(out, file=sys.stderr)
    print(err, file=sys.stderr)


def _should_use_namespace(pid):
    """Return whether the target process lives in a different mount namespace."""
    try:
        import os

        own_ns = os.readlink("/proc/self/ns/mnt")
        target_ns = os.readlink(f"/proc/{pid}/ns/mnt")
        return own_ns != target_ns
    except Exception:
        # This can fail for a number of reasons, but they're all cases where we can't join the processes namespace
        return False


PidType = int | Literal["all"]


def pid_type(value: str) -> PidType:
    """Custom argparse type for PID argument."""
    if value == "all":
        return "all"
    try:
        pid = int(value)
        if pid <= 0:
            raise ValueError
        return pid
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid PID: {value}")


def _upload(url, file_path):
    """Upload a file to a URL.

    The main use case is uploading to an S3 presigned URL, but this function can be used for any HTTP PUT upload.
    """
    parsed_url = urlparse(url)
    host = parsed_url.netloc
    path = parsed_url.path + "?" + parsed_url.query

    conn = HTTPSConnection(host)
    with ExitStack() as stack:
        stack.callback(conn.close)
        file = stack.enter_context(open(file_path, "rb"))
        content_length = file.seek(
            0, os.SEEK_END
        )  # Move to the end of the file to get its size
        file.seek(0)  # Move back to the beginning of the file
        conn.request(
            "PUT",
            path,
            body=file,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(content_length),
            },
        )
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"Failed to upload file. Status code: {response.status}, Response: {response.read().decode()}"
            )
        response.read()  # Read the response to ensure the connection is closed properly


def main():
    """CLI entry point for dumping a live Python process by PID."""
    parser = argparse.ArgumentParser(
        description="Dump the heap of a running Python process."
    )
    parser.add_argument(
        "pid",
        type=pid_type,
        help="PID of the target Python process, or 'all' to dump all Python processes.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default=None,
        help=f"Path to output file (default: {DEFAULT_DUMP_FILE})",
    )
    parser.add_argument(
        "--no-namespace-injection",
        action="store_false",
        dest="can_use_namespace_injection",
        help="Don't attempt to use namespace injection method.",
    )
    parser.add_argument(
        "--no-alternate-python-interpreter",
        action="store_false",
        dest="can_use_alternate_python_interpreter",
        help="Don't attempt to use an alternate Python interpreter for injection, even if the target process is running a different Python version.",
    )
    parser.add_argument(
        "--upload-url",
        "-u",
        default=None,
        help="If provided, upload the dump file to this URL after dumping. The URL should be a presigned S3 PUT URL or any other HTTP PUT endpoint.",
    )
    args = parser.parse_args()

    if args.output_file is None:
        if args.pid == "all":
            args.output_file = DEFAULT_DUMP_ARCHIVE_FILE
        else:
            args.output_file = DEFAULT_DUMP_FILE

    dump_heap_from_pid(
        args.pid,
        args.output_file,
        args.can_use_namespace_injection,
        args.can_use_alternate_python_interpreter,
    )
    if args.upload_url:
        _upload(args.upload_url, args.output_file)


if __name__ == "__main__":
    main()
