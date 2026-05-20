#!/usr/bin/env python

import subprocess
from argparse import ArgumentParser
from tomllib import load  # ty: ignore[unresolved-import]
import sys

if __name__ == "__main__":
    argparser = ArgumentParser(
        description="Check that the versions in pyproject.toml and Cargo.toml match the git tag"
    )
    argparser.add_argument("tag", help="The git tag to check against", nargs="?")
    args = argparser.parse_args()

    if args.tag is None:
        try:
            tag = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error getting git tag: {e}")
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
            exit(1)
        if tag.startswith("release/"):
            tag = tag.removeprefix("release/")
        elif tag.startswith("test/"):
            tag = tag.removeprefix("test/")
        else:
            raise ValueError(f"Unexpected tag format: {tag}")
    else:
        tag = args.tag

    failed = False
    with open("midden/pyproject.toml", "rb") as f:
        pyproject_data = load(f)
        midden_version = pyproject_data["project"]["version"]
        if midden_version != tag:
            print(
                f"Version mismatch in midden/pyproject.toml: {midden_version} != {tag}"
            )
            failed = True

        ui_deps = pyproject_data["project"]["optional-dependencies"]["ui"]
        if f"midden-analysis=={tag}" not in ui_deps:
            print(
                f"Version mismatch in midden/pyproject.toml optional dependencies: midden-analysis=={tag} not found in {ui_deps}"
            )
            failed = True

    with open("midden-analysis/Cargo.toml", "rb") as f:
        cargo_data = load(f)
        cargo_version = cargo_data["package"]["version"]
        if cargo_version != tag:
            print(
                f"Version mismatch in midden-analysis/Cargo.toml: {cargo_version} != {tag}"
            )
            failed = True

    if failed:
        exit(1)
    else:
        print("Versions match git tag")
