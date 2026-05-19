import time
import json
from collections.abc import Generator
import subprocess
import pytest
import sys

from pathlib import Path

import docker

from midden.dump.inject import dump_heap_from_pid

if sys.platform == "linux":
    INJECTABLE_PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14", "3.14+gil"]
else:
    # On non-Linux platforms, we need sys.remote_exec, which only supports Python 3.14+
    INJECTABLE_PYTHON_VERSIONS = ["3.14"]


@pytest.fixture(params=INJECTABLE_PYTHON_VERSIONS)
def python_venv(tmp_path, request: pytest.FixtureRequest) -> Generator[Path]:
    """Create a temporary virtual environment for testing."""
    venv_dir = tmp_path / "venv"
    subprocess.run(["uv", "venv", "-p", f"{request.param}", venv_dir], check=True)
    return venv_dir / "bin" / "python"


@pytest.fixture
def injectable_process(python_venv: Path) -> Generator[int]:
    """Start the dummy injectable program in a subprocess."""
    print("Starting injectable process...", file=sys.stderr)
    proc = subprocess.Popen(
        [str(python_venv), Path(__file__).parent / "dummy_injectable_program.py"],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    read = proc.stdout.readline()
    assert read == b"Started!!!\n"
    try:
        yield proc.pid
    finally:
        proc.terminate()
        proc.wait()


def test_inject(injectable_process: int, tmp_path: Path):
    """Test that we can inject into the process and get a heap dump."""
    output_file = str(tmp_path / "dump.jsonl")
    dump_heap_from_pid(injectable_process, output_file)
    assert_dump_is_as_expected(output_file)


INJECTABLE_PROGRAM = (Path(__file__).parent / "dummy_injectable_program.py").read_text()


@pytest.fixture
def injectable_namespaced_process(request: pytest.FixtureRequest) -> Generator[int]:
    print(
        "Connecting to Docker and starting namespaced injectable process...",
        file=sys.stderr,
    )
    docker_client = docker.from_env()
    print("Running container with injectable program...", file=sys.stderr)
    container = docker_client.containers.run(
        image="python:3.14",
        command=["python", "-c", INJECTABLE_PROGRAM],
        detach=True,
        remove=True,
    )
    deadline = time.monotonic() + 5  # 5-second timeout
    while time.monotonic() < deadline:
        container.reload()
        if container.status == "running" and "Started!!!" in container.logs().decode():
            break
        time.sleep(0.5)
    else:
        print(container.logs(), file=sys.stderr)
        raise TimeoutError("Container did not start within 5 seconds")
    try:
        print("Started container", file=sys.stderr)
        yield container.attrs["State"]["Pid"]
    finally:
        print("Stopping container...", file=sys.stderr)
        container.stop()


@pytest.mark.linux
@pytest.mark.min_python((3, 12))
def test_namespaced_inject(injectable_namespaced_process: int, tmp_path: Path):
    """Test that we can inject into a namespaced process and get a heap dump."""
    output_file = str(tmp_path / "dump.jsonl")
    args = [
        "sudo",
        sys.executable,
        "-m",
        "midden.dump.inject",
        str(injectable_namespaced_process),
        "-o",
        output_file,
    ]
    print(f"Running injection subprocess with args: {args}", file=sys.stderr)
    subprocess.run(
        args,
        check=True,
    )
    assert_dump_is_as_expected(output_file)


def assert_dump_is_as_expected(dump_file: str):
    """Assert that the dump file contains the expected data."""
    objects = {}
    with open(dump_file) as f:
        for line in f:
            item = json.loads(line)
            objects[item["id"]] = item
    # Look for the test data item in the dump
    for obj in objects.values():
        if obj["type"] == "builtins.str" and obj.get("value") == "Test Data Item":
            test_data_item_id = obj["id"]
            break
    else:
        raise AssertionError("Test data item not found in dump")
    # Look for the test data dict and check it references the test data item
    for obj in objects.values():
        if (
            obj["type"] == "builtins.list"
            and test_data_item_id in obj["references"]
            and len(obj["references"]) == 5
        ):
            test_list = obj
            break
    else:
        raise AssertionError("List referencing test data item not found in dump")

    test_dict = objects[
        test_list["references"][1]
    ]  # The dict should be the second reference from the list
    assert test_dict["type"] == "builtins.dict", "Expected a dict"
    assert objects[test_dict["references"][0]]["value"] == "Test Data Key", (
        "Expected dict key not found"
    )
    assert objects[test_dict["references"][1]]["value"] == "Test Data Value", (
        "Expected dict value not found"
    )

    test_tuple = objects[test_list["references"][2]]
    assert test_tuple["type"] == "builtins.tuple", "Expected a tuple"
    assert objects[test_tuple["references"][0]]["value"] == "1", (
        "Expected tuple item not found"
    )
    assert objects[test_tuple["references"][0]]["type"] == "builtins.int", (
        "Expected tuple item not found"
    )

    test_set = objects[test_list["references"][3]]
    assert test_set["type"] == "builtins.set", "Expected a set"
    assert objects[test_set["references"][0]]["value"] == "2.0", (
        "Expected set item not found"
    )
    assert objects[test_set["references"][0]]["type"] == "builtins.float", (
        "Expected set item not found"
    )

    test_frozenset = objects[test_list["references"][4]]
    assert test_frozenset["type"] == "builtins.frozenset", "Expected a frozenset"
    assert objects[test_frozenset["references"][0]]["value"] == "b'3'", (
        "Expected frozenset item not found"
    )
    assert objects[test_frozenset["references"][0]]["type"] == "builtins.bytes", (
        "Expected frozenset item not found"
    )
