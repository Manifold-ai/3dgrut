# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Locked minimal runtime dependencies for ``threedgrut_playground`` (headless).

This is the single source of truth enforced by the import-time version check in
``threedgrut_playground/__init__.py`` and mirrored by the ``playground-lock`` extra
in the repo-root ``pyproject.toml``.

``PACKAGES`` lists the minimal headless dependency set (the ``Engine3DGRUT``
shadow-catcher path; GUI-only ``polyscope``/``viser`` and training-only packages are
intentionally excluded). ``LOCKED`` maps distribution name -> exact version and is
**generated on the target CUDA env** by ``scripts/gen_playground_lock.py`` (this repo
has no local CUDA env to read versions from). While ``LOCKED`` is empty the import-time
check is a no-op, so importing the package stays clean until the lock is populated.
"""

from __future__ import annotations

# Distribution names (as known to importlib.metadata / pip), in install order-agnostic
# form. Keep in sync with the ``playground-lock`` extra in pyproject.toml.
PACKAGES: list[str] = [
    "torch",
    "numpy",
    "kaolin",
    "hydra-core",
    "imageio",
    "scipy",
    "opencv-python",
    "pillow",
    "tqdm",
    "pygltflib",
    "msgpack",
    "omegaconf",
    "plyfile",
    "ninja",
]

# distribution name -> exact version. Filled by scripts/gen_playground_lock.py.
LOCKED: dict[str, str] = {}
