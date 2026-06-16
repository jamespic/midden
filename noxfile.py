import sys

from nox import Session, options
from nox_uv import session

options.default_venv_backend = "uv"
options.sessions = [
    "lint_midden",
    "lint_midden_analysis",
    "lint_rust",
    "test_injection",
    "test_analysis",
    "test_analysis_rust",
]


@session(
    python=["3.10", "3.11", "3.12", "3.13", "3.14"]
    if sys.platform == "linux"
    else ["3.14"],
    uv_packages=["midden"],
    uv_groups=["dev"],
)
def test_injection(session: Session) -> None:
    session.cd("midden")
    session.run("pytest")


@session(
    python=["3.10", "3.11", "3.12", "3.13", "3.14"],
    uv_packages=["midden-analysis"],
    uv_groups=["dev"],
)
def test_analysis(session: Session) -> None:
    session.cd("midden-analysis")
    session.run("pytest")


@session
def install_rust(session: Session) -> None:
    session.run_install("rustup", "install", "stable", external=True)


@session(requires=["install_rust"])
def test_analysis_rust(session: Session) -> None:
    session.cd("midden-analysis")
    session.run("cargo", "test", external=True)


@session(requires=["install_rust"])
def lint_rust(session: Session) -> None:
    session.cd("midden-analysis")
    session.run("cargo", "fmt", "--check", external=True)
    session.run("cargo", "clippy", external=True, silent=True)


@session(requires=["install_rust"])
def fix_rust(session: Session) -> None:
    session.cd("midden-analysis")
    session.run("cargo", "fmt", external=True)
    session.run("cargo", "clippy", "--fix", external=True, silent=True)


@session(uv_packages=["midden"], uv_all_groups=True, uv_all_extras=True)
def lint_midden(session: Session) -> None:
    session.cd("midden")
    session.run("ruff", "check")
    session.run("ty", "check")


@session(uv_packages=["midden-analysis"], uv_all_groups=True, uv_all_extras=True)
def lint_midden_analysis(session: Session) -> None:
    session.cd("midden-analysis")
    session.run("ruff", "check")
    session.run("ty", "check")


@session(python=["3.14"])
def check_versions(session: Session) -> None:
    session.run("python", "check-versions.py")
