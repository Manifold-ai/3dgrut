# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for ``threedgrut_playground``.

Downstream projects use only this namespace (never ``threedgrut`` directly):

    import threedgrut_playground as pg
    engine = pg.Engine3DGRUT(gs_object=ckpt, mesh_assets_folder=..., default_config=...)

The four exported names are imported **lazily** (on first attribute access) so that
``import threedgrut_playground`` itself stays light — it does not pull in kaolin, torch
CUDA, or the OptiX/CUDA tracers, which only load when you actually touch ``Engine3DGRUT``
et al. At import time we run a lightweight check (``importlib.metadata`` only) that the
installed dependency versions match the locked set in :mod:`threedgrut_playground._deps_lock`.
"""

__all__ = ["Engine3DGRUT", "Light", "LightType", "OptixPrimitiveTypes"]


def _check_locked_deps() -> None:
    """Warn (or, under THREEDGRUT_PLAYGROUND_STRICT_DEPS=1, raise) when installed
    versions diverge from the locked set. No-op while the lock is empty/ungenerated."""
    import importlib.metadata as md
    import os
    import warnings

    from ._deps_lock import LOCKED

    if not LOCKED:
        return
    mismatches = []
    for pkg, want in LOCKED.items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = None
        if got != want:
            mismatches.append((pkg, want, got))
    if mismatches:
        detail = ", ".join(f"{p} expected {w} but found {g or 'MISSING'}" for p, w, g in mismatches)
        msg = (
            "threedgrut_playground locked dependencies do not match the installed "
            f"environment: {detail}. Install with `pip install \"threedgrut[playground-lock]\"` "
            "for the validated versions, or set THREEDGRUT_PLAYGROUND_STRICT_DEPS=1 to treat "
            "this as an error."
        )
        if os.environ.get("THREEDGRUT_PLAYGROUND_STRICT_DEPS") == "1":
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)


_check_locked_deps()


def __getattr__(name: str):
    # PEP 562 module-level lazy attribute access: defer the heavy engine import until
    # one of the public names is actually used.
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
