# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Generate the locked minimal dependency set for ``threedgrut_playground``.

Run this **on the target CUDA env** (the one where the playground actually runs), with
the minimal deps already installed:

    python scripts/gen_playground_lock.py            # write _deps_lock.py + print toml
    python scripts/gen_playground_lock.py --check    # dry-run: report, change nothing

It reads ``threedgrut_playground._deps_lock.PACKAGES``, resolves each distribution's
installed version via ``importlib.metadata``, writes the resulting ``LOCKED`` mapping
back into ``threedgrut_playground/_deps_lock.py``, and prints the matching
``playground-lock`` block to paste into the repo-root ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "threedgrut_playground" / "_deps_lock.py"

# usd-core keeps its platform marker in the extra; it is not enforced by the runtime
# version check (it is optional / platform-gated), so it is not part of PACKAGES.
_PLATFORM_GATED = ['"usd-core>=26.5; sys_platform == \'linux\' and platform_machine == \'x86_64\'"']


def _resolve_versions(packages: list[str]) -> tuple[dict[str, str], list[str]]:
    locked: dict[str, str] = {}
    missing: list[str] = []
    for pkg in packages:
        try:
            locked[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            missing.append(pkg)
    return locked, missing


def _render_lock_dict(locked: dict[str, str]) -> str:
    body = "".join(f'    "{pkg}": "{ver}",\n' for pkg, ver in locked.items())
    return "LOCKED: dict[str, str] = {\n" + body + "}\n" if locked else "LOCKED: dict[str, str] = {}\n"


def _rewrite_lock_file(locked: dict[str, str]) -> None:
    text = LOCK_FILE.read_text()
    marker = "LOCKED: dict[str, str] ="
    idx = text.index(marker)
    new_text = text[:idx] + _render_lock_dict(locked)
    LOCK_FILE.write_text(new_text)


def _render_pyproject_block(locked: dict[str, str]) -> str:
    pins = [f'    "{pkg}=={ver}",' for pkg, ver in locked.items()] + [f"    {g}," for g in _PLATFORM_GATED]
    return "playground-lock = [\n" + "\n".join(pins) + "\n]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="Dry run: print what would change, write nothing")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from threedgrut_playground._deps_lock import PACKAGES

    locked, missing = _resolve_versions(PACKAGES)
    if missing:
        print(f"[warn] not installed in this env (skipped): {', '.join(missing)}", file=sys.stderr)
    if not locked:
        print("[error] none of the locked packages are installed; run this on the playground env.", file=sys.stderr)
        return 1

    print("# resolved versions:")
    for pkg, ver in locked.items():
        print(f"#   {pkg}=={ver}")
    print("\n# paste into pyproject.toml [project.optional-dependencies]:\n")
    print(_render_pyproject_block(locked))

    if args.check:
        print("\n[check] dry-run; _deps_lock.py not modified.", file=sys.stderr)
        return 0

    _rewrite_lock_file(locked)
    print(f"\n[ok] wrote LOCKED ({len(locked)} pkgs) -> {LOCK_FILE.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
