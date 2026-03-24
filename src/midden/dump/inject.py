import time
import tempfile
from subprocess import Popen, PIPE
import os
import pathlib

try:
    from sys import remote_exec
except ImportError:
    remote_exec = None  # type: ignore

DEFAULT_DUMP_FILE = "/tmp/dump.jsonl"


def dump_heap_from_pid(pid, output_file=DEFAULT_DUMP_FILE):
    """Dump the heap of a running Python process given its PID."""
    if _should_use_namespace(pid):
        print(
            "Target process is in a different mount namespace, using namespace injection method"
        )
        _dump_heap_from_pid_in_namespace(pid, output_file)
    else:
        code = _build_dump_heap_code(output_file)
        _inject_into_process(pid, code)


def _build_dump_heap_code(output_file):
    res = pathlib.Path(__file__).parent / "dump_heap.py"
    code = res.read_text()
    return code.replace(DEFAULT_DUMP_FILE, output_file).replace(
        "# _dump_heap()", "_dump_heap()"
    )


def _dump_heap_from_pid_in_namespace(pid, output_file=DEFAULT_DUMP_FILE):
    dump_loc_in_namespace = f"/tmp/dump_{pid}.jsonl"
    inner_pid = _identify_pid_within_namespace(pid)
    code = _build_dump_heap_code(dump_loc_in_namespace)

    if (forked_pid := os.fork()) == 0:
        out = os.open(output_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.setns(os.pidfd_open(pid), os.CLONE_NEWNS | os.CLONE_NEWPID)
        if (
            inner_forked_pid := os.fork()
        ) == 0:  # Must fork again after setns to actually be in the new PID namespace
            _inject_into_process(inner_pid, code)
            while not os.path.exists(dump_loc_in_namespace):
                time.sleep(0.1)
            with open(dump_loc_in_namespace) as f, os.fdopen(out, "w") as out_f:
                for line in f:
                    out_f.write(line)
            os.remove(dump_loc_in_namespace)
            os._exit(0)
        else:
            pid, status = os.waitpid(inner_forked_pid, 0)
        os._exit(status)
    else:
        pid, status = os.waitpid(forked_pid, 0)
        if status != 0:
            print(f"Child process failed with status {status}")


def _identify_pid_within_namespace(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("NSpid:"):
                return int(line.split()[-1])


def _inject_into_process(pid, code):
    script_file = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
    script_file.write(code)
    script_file.close()
    if remote_exec is not None:
        print("Using remote_exec to inject code")
        remote_exec(pid, script_file.name)
    else:
        print("remote_exec not available, falling back to gdb method")
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
        print(out)
        print(err)


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
    args = parser.parse_args()
    dump_heap_from_pid(args.pid, args.output_file)

if __name__ == "__main__":
    main()
