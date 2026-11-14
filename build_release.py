# Author: bd78
"""Build a versioned release of the target analyser and publish it on GitHub.

Usage:
  .venv\\Scripts\\python build_release.py 1.2.0
  .venv\\Scripts\\python build_release.py 1.2.0 --dry-run   (build and zip only, no GitHub release)

Steps:
  1. check the version format, and (unless --dry-run) that the GitHub CLI is
     installed and logged in and that the tag v<version> does not exist yet;
  2. compile the package and build the exe into target\\<version>\\TargetAnalyzer\\;
  3. zip that folder to target\\TargetAnalyzer-<version>.zip;
  4. create the GitHub release v<version> at the current commit and attach the zip.

Requires the GitHub CLI (gh): https://cli.github.com/ , then `gh auth login`.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import build_exe as be

ROOT = be.ROOT
TARGET = ROOT / "target"
APP = be.APP_NAME
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def preflight(version: str, tag: str) -> str:
    """Check the environment before the slow build. Returns the commit SHA to tag."""
    if not VERSION_RE.match(version):
        be.fail(f"version '{version}' must look like 1.2.3")

    sha = run(["git", "rev-parse", "HEAD"])
    if sha.returncode != 0:
        be.fail("not a git repository, or git is not on PATH")
    commit = sha.stdout.strip()

    dirty = run(["git", "status", "--porcelain"]).stdout.strip()
    if dirty:
        print("WARNING: uncommitted changes. The release tag points at the last commit, "
              "so the zip may not match the code you see.")

    if shutil.which("gh") is None:
        be.fail("GitHub CLI (gh) not found. Install it from https://cli.github.com/")
    if run(["gh", "auth", "status"]).returncode != 0:
        be.fail("GitHub CLI is not logged in. Run: gh auth login")
    if run(["gh", "release", "view", tag]).returncode == 0:
        be.fail(f"release {tag} already exists on GitHub; choose another version")
    return commit


def build_exe_into(version_dir: Path) -> Path:
    be.check_compile()
    be.check_pyinstaller()
    if version_dir.exists():
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="build_release_") as tmp:
        be.build(Path(tmp), version_dir)
    exe = version_dir / APP / f"{APP}.exe"
    if not exe.is_file():
        be.fail(f"expected executable not found: {exe}")
    return version_dir / APP


def make_zip(app_dir: Path, zip_path: Path) -> None:
    print(f"Zipping {app_dir.name}/ -> {zip_path.name} ...")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                # Keep the TargetAnalyzer/ folder as the top level inside the zip.
                zf.write(path, path.relative_to(app_dir.parent))


def create_release(tag: str, commit: str, zip_path: Path, version: str) -> None:
    print(f"Creating GitHub release {tag} ...")
    notes = (f"Windows build of Target Impact Analyser {version}.\n\n"
             f"Download {zip_path.name}, extract it and run {APP}\\{APP}.exe. "
             "Keep the whole folder together.")
    cmd = [
        "gh", "release", "create", tag, str(zip_path),
        "--target", commit,
        "--title", f"{APP} {version}",
        "--notes", notes,
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        be.fail(f"gh release create failed with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="release version, e.g. 1.2.0")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and zip only; do not touch GitHub")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    tag = f"v{version}"

    if args.dry_run:
        if not VERSION_RE.match(version):
            be.fail(f"version '{version}' must look like 1.2.3")
        commit = ""
    else:
        commit = preflight(version, tag)

    version_dir = TARGET / version
    app_dir = build_exe_into(version_dir)
    zip_path = TARGET / f"{APP}-{version}.zip"
    make_zip(app_dir, zip_path)
    print(f"Built: {zip_path} ({zip_path.stat().st_size / 1_000_000:.1f} MB)")

    if args.dry_run:
        print("Dry run: GitHub release skipped.")
        return
    create_release(tag, commit, zip_path, version)
    print(f"Released {tag}.")


if __name__ == "__main__":
    main()
