from contextlib import contextmanager
import io
import shutil
import re
import tarfile
import time
import tempfile
from subprocess import Popen, PIPE
import os
import pathlib
import sys
import subprocess

try:
    from sys import remote_exec
except ImportError:
    remote_exec = None  # type: ignore

try:
    import psutil

    _psutil_available = True
except ImportError:
    _psutil_available = False

DEFAULT_DUMP_FILE = "/tmp/dump.jsonl"

_FILES_NEEDED_FOR_INJECTION = ["dump_heap.py", "inject.py"]


def _build_tarball_of_dumping_code():
    file_obj = io.BytesIO()
    with tarfile.open(fileobj=file_obj, mode="w:gz") as tar:
        for filename in _FILES_NEEDED_FOR_INJECTION:
            file_path = pathlib.Path(__file__).parent / filename
            tar.add(file_path, arcname=filename)
    return file_obj.getvalue()


# We do this at import time so that if we fork into another namespace, it's still available
_TARBALL = _build_tarball_of_dumping_code()


def dump_heap_from_pid(
    pid,
    output_file=DEFAULT_DUMP_FILE,
    can_use_namespace_injection=True,
    can_use_alternate_python_interpreter=True,
):
    """Dump the heap of a running Python process given its PID."""
    _dump_heap_from_pid_possibly_in_namespace(
        pid,
        output_file,
        can_use_namespace_injection,
        can_use_alternate_python_interpreter,
    )


_DUMP_SCRIPT = (pathlib.Path(__file__).parent / "dump_heap.py").read_text()


def _build_dump_heap_code(output_file):
    return _DUMP_SCRIPT.replace(DEFAULT_DUMP_FILE, output_file).replace(
        "# _dump_heap()", "_dump_heap()"
    )


def _dump_heap_from_pid_possibly_using_an_alternate_python_interpreter(
    pid,
    output_file=DEFAULT_DUMP_FILE,
    can_use_alternate_python_interpreter=True,
):
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
    # We untar the dumping code into a temporary directory, since we might be namespaces and not have access to our original filesystem
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
    try:
        exe, maps = _get_exe_and_maps(pid)
    except Exception as e:
        print(
            f"Could not determine target process executable and maps, so proceeding with this interpreter: {e!r}",
            file=sys.stderr,
        )
        return None
    if not _can_this_python_inject(exe):
        if exe.startswith("python"):
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
                    if python_exe_wanted == _effective_executable_name():
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
        except Exception:
            pass
        return exe, maps
    else:
        raise Exception(
            "Can't determine process executable on this platform without psutil"
        )


def _can_this_python_inject(exe):
    if exe == sys.executable:
        return True
    basename = os.path.basename(exe)
    if basename == _effective_executable_name():
        return True


def _effective_executable_name():
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def _dump_heap_from_pid_possibly_in_namespace(
    pid,
    output_file=DEFAULT_DUMP_FILE,
    can_use_namespace_injection=True,
    can_use_alternate_python_interpreter=True,
):
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
    inner_pid = _identify_pid_within_namespace(pid)
    if (forked_pid := os.fork()) == 0:
        os.setns(os.pidfd_open(pid), os.CLONE_NEWNS | os.CLONE_NEWPID)
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
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("NSpid:"):
                return int(line.split()[-1])


def _dump_heap_from_pid(pid, output_file=DEFAULT_DUMP_FILE):
    code = _build_dump_heap_code(output_file)
    _inject_into_process(pid, code)


def _inject_into_process(pid, code):
    script_file = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    script_file.write(code)
    script_file.close()
    if remote_exec is not None:
        print("Using remote_exec to inject code", file=sys.stderr)
        remote_exec(pid, script_file.name)
    else:
        print("remote_exec not available, falling back to gdb method", file=sys.stderr)
        gdb_cmds = [
            "(char *) PyGILState_Ensure()",
            '(void) PyRun_SimpleString("'
            rf'exec(open(\"{script_file.name}\").read())")',
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
    try:
        import os

        own_ns = os.readlink("/proc/self/ns/mnt")
        target_ns = os.readlink(f"/proc/{pid}/ns/mnt")
        return own_ns != target_ns
    except Exception:
        # This can fail for a number of reasons, but they're all cases where we can't join the processes namespace
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dump the heap of a running Python process."
    )
    parser.add_argument("pid", type=int, help="PID of the target Python process")
    parser.add_argument(
        "--output-file",
        "-o",
        default=DEFAULT_DUMP_FILE,
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
    args = parser.parse_args()
    dump_heap_from_pid(
        args.pid,
        args.output_file,
        args.can_use_namespace_injection,
        args.can_use_alternate_python_interpreter,
    )


if __name__ == "__main__":
    main()
