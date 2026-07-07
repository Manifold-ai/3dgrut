# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Headless functional test for the shadow-catcher feature (Phase 1+2+3).

Run on a CUDA/OptiX machine with a real 3DGS checkpoint + a ground catcher mesh
(externally aligned to the GS ground). Renders a sequence of scenes through the
headless Engine3DGRUT API and saves a PNG per scene under --out.

  gs_only      : nothing added -> use it to AIM THE CAMERA at your scene first
  A baseline   : occluder only, no catcher, no lights   (hybrid path, numLights=0)
  B P1 透明     : + catcher, no lights                   -> ~= A (catcher invisible)
  C P2 no-op    : occluder + light, no catcher           -> ~= A (light unused w/o catcher)
  D P3 硬阴影    : occluder + catcher + 1 directional      -> ground GS darkens
  E P3 软阴影    : D with angular_radius>0                 -> softer shadow edge
  F 关 GS       : D with gaussian tracing disabled         -> must not crash

Each scene PRINTS what is actually in the scene (primitive names + light count)
right before rendering, and the BVH is force-rebuilt so the occluder/catcher are
guaranteed to be in the acceleration structure. The PNGs are the ground truth;
the numeric asserts are sanity heuristics (disable with --skip_assert).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch


def vec3(s: str) -> tuple:
    return tuple(float(x) for x in s.split(","))


ALL_SCENE_TAGS = (
    "gs_only",
    "A_baseline",
    "B_catcher_nolight",
    "C_light_nocatcher",
    "D_hard_shadow",
    "E_soft_shadow",
    "F_disable_gs",
    "G_multi_light",
    "H_point_light",
    "I_point_plus_dir",
    "J_area_light",
)

SCENE_ALIASES = {
    "gs_only": "gs_only",
    "a": "A_baseline",
    "a_baseline": "A_baseline",
    "b": "B_catcher_nolight",
    "b_catcher_nolight": "B_catcher_nolight",
    "c": "C_light_nocatcher",
    "c_light_nocatcher": "C_light_nocatcher",
    "d": "D_hard_shadow",
    "d_hard_shadow": "D_hard_shadow",
    "e": "E_soft_shadow",
    "e_soft_shadow": "E_soft_shadow",
    "f": "F_disable_gs",
    "f_disable_gs": "F_disable_gs",
    "g": "G_multi_light",
    "g_multi_light": "G_multi_light",
    "h": "H_point_light",
    "h_point_light": "H_point_light",
    "i": "I_point_plus_dir",
    "i_point_plus_dir": "I_point_plus_dir",
    "j": "J_area_light",
    "j_area_light": "J_area_light",
}

ASSERTION_SCENE_TAGS = (
    "A_baseline",
    "B_catcher_nolight",
    "C_light_nocatcher",
    "D_hard_shadow",
    "F_disable_gs",
    "J_area_light",
)


def _scene_tokens(value) -> list[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"Scene names must be strings, got {item!r}")
            raw_items.extend(item.split(","))
    else:
        raise ValueError(f"--scenes must be 'all', a comma-separated string, or a JSON list; got {value!r}")
    return [item.strip() for item in raw_items if item.strip()]


def resolve_scene_selection(value="all") -> list[str]:
    tokens = _scene_tokens(value)
    if not tokens:
        raise ValueError("No scenes selected")
    if any(token.lower() == "all" for token in tokens):
        return list(ALL_SCENE_TAGS)

    selected = []
    for token in tokens:
        key = token.lower()
        if key == "shadow_diff":
            expanded = ("B_catcher_nolight", "D_hard_shadow")
        elif key in SCENE_ALIASES:
            expanded = (SCENE_ALIASES[key],)
        else:
            valid = ", ".join(["all", "shadow_diff", *SCENE_ALIASES.keys()])
            raise ValueError(f"Unknown scene {token!r}. Valid scenes/aliases: {valid}")
        for tag in expanded:
            if tag not in selected:
                selected.append(tag)
    return selected


def validate_assertion_scene_dependencies(selected_scenes: list[str], skip_assert: bool) -> None:
    if skip_assert:
        return
    missing = [tag for tag in ASSERTION_SCENE_TAGS if tag not in selected_scenes]
    if missing:
        raise ValueError(
            "Partial scene selection is missing scenes required by numeric asserts: "
            f"{', '.join(missing)}. Add those scenes or pass --skip_assert."
        )


def _cuda_synchronize_for_timing() -> None:
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()


def parse_args() -> argparse.Namespace:
    import json
    # Pre-parse --preset so a JSON preset can supply defaults (explicit CLI flags
    # still win). --config is already taken (3dgrt model yaml), so we use --preset.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--preset", default="")
    _preset_path = _pre.parse_known_args()[0].preset
    _presets = {}
    if _preset_path:
        with open(_preset_path) as _f:
            _presets = json.load(_f)

    p = argparse.ArgumentParser(description="Shadow-catcher P1-P3 functional test")
    p.add_argument("--preset", default="", help="Load a JSON preset as defaults (explicit CLI flags still override)")
    p.add_argument("--save_preset", default="", help="Write the resolved config to this JSON path, then continue rendering")
    # required only when the preset does not already supply them
    p.add_argument("--gs_object", required=("gs_object" not in _presets), help="3DGS checkpoint (.pt/.ingp/.ply)")
    p.add_argument("--catcher", required=("catcher" not in _presets), help="External ground catcher mesh (.obj/.glb/.gltf)")
    p.add_argument("--mesh_assets", default="threedgrut_playground/assets", help="Folder with occluder assets")
    p.add_argument("--occluder", default="Teapot", help="Occluder asset name (stem capitalized), e.g. 'Teapot'")
    p.add_argument("--occluder_type", default="PBR", choices=["PBR", "DIFFUSE"], help="Occluder primitive type")
    p.add_argument("--occluder_pos", default="0,-1,0", type=vec3, help="Occluder world position (panoramic scene: near origin, slightly down)")
    p.add_argument("--occluder_frac", type=float, default=0.08, help="Scale occluder to this fraction of scene extent (visibility)")
    p.add_argument("--occluder_rot", default="0,0,0", type=vec3, help="Rotate occluder by XYZ euler degrees (B-plan: '180,0,0' flips it upright for up=0,-1,0)")
    p.add_argument("--catcher_scale", type=float, default=0.98, help="Scale catcher about scene center (shrink just inside the GS surface)")
    p.add_argument("--scene_rot", default="0,0,0", type=vec3, help="Rotate the room (GS + catcher) by XYZ euler degrees about the scene center")
    p.add_argument("--config", default="apps/colmap_3dgrt.yaml", help="Default config for .ingp/.ply or non-3dgrt .pt")
    p.add_argument("--out", default="./shadow_test_out", help="Output dir for PNGs")
    p.add_argument("--res", type=int, default=512, help="Render resolution (square)")
    p.add_argument("--scenes", default="all",
                   help="Comma-separated scenes to render: all, gs_only, A-J/full names, or shadow_diff")
    # Camera: look-at the occluder from a modest distance
    p.add_argument("--eye_dir", default="0,0.4,-1", type=vec3, help="Direction from the occluder to the eye (world)")
    p.add_argument("--cam_dist", type=float, default=0.25, help="Eye distance = cam_dist * scene_extent")
    p.add_argument("--up", default="0,1,0", type=vec3, help="Camera up axis")
    p.add_argument("--fov", type=float, default=60.0, help="Vertical FOV in degrees")
    p.add_argument("--eye", default="", help="Override absolute eye 'x,y,z'")
    p.add_argument("--cam_at", default="", help="Override look-at target 'x,y,z' (else = occluder position)")
    p.add_argument("--cam_azimuth", type=float, default=0.0, help="Rotate the camera horizontally around the world Y axis by N degrees (azimuth)")
    # Quality (off by default for deterministic regression; turn on for clean images)
    p.add_argument("--spp", type=int, default=0, help="Anti-aliasing samples (0=off/deterministic; 32-64 = clean)")
    p.add_argument("--denoise", action="store_true", help="Enable OptiX denoiser (cleaner; breaks pixel-exact regression)")
    # Light (direction = FROM shading point TO light, world space)
    p.add_argument("--light_dir", default="0,-1,0", type=vec3)
    p.add_argument("--intensity", type=float, default=3.0)
    p.add_argument("--soft_angle", type=float, default=0.05)
    p.add_argument("--shadow_min", type=float, default=0.0, help="Shadow floor in [0,1]: 0=shadows reach black, 0.2=darkest keeps 20%")
    p.add_argument("--shadow_spp", type=int, default=128, help="Soft-shadow occlusion samples per light (used when --soft_angle>0)")
    p.add_argument("--tol", type=float, default=2e-2, help="Tolerance for the A==B / A==C regression checks")
    p.add_argument("--skip_assert", action="store_true", help="Only render + save, skip numeric asserts")
    p.add_argument("--light_dirs", default="",
                   help="Semicolon-separated extra directional lights for multi-light, "
                        "e.g. '0.6,1,0.4;-0.6,1,0.4;0.6,1,-0.4'. Empty = use the single --light_dir")
    p.add_argument("--point_pos", default="",
                   help="POINT-light world position 'x,y,z'. Empty = auto: above the occluder along --light_dir")
    p.add_argument("--point_dist", type=float, default=0.5,
                   help="Auto point-light height = point_dist * scene_extent above the occluder (when --point_pos empty)")
    p.add_argument("--area_size", type=float, default=0.1,
                   help="AREA light rectangle half-edge length = area_size * scene_extent (bigger = softer)")
    if _presets:
        p.set_defaults(**_presets)  # preset values become defaults; CLI flags still override
    args = p.parse_args()
    try:
        resolve_scene_selection(args.scenes)
    except ValueError as e:
        p.error(str(e))
    if args.save_preset:
        _cfg = {k: v for k, v in vars(args).items() if k not in ("preset", "save_preset")}
        with open(args.save_preset, "w") as _f:
            json.dump(_cfg, _f, indent=2)
        print(f"[preset] saved resolved config -> {args.save_preset}")
    return args


def main() -> int:
    args = parse_args()
    selected_scenes = resolve_scene_selection(args.scenes)
    try:
        validate_assertion_scene_dependencies(selected_scenes, skip_assert=args.skip_assert)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)

    import kaolin
    from threedgrut_playground.engine import Engine3DGRUT, Light, LightType, OptixPrimitiveTypes

    occluder_type = getattr(OptixPrimitiveTypes, args.occluder_type)

    engine = Engine3DGRUT(gs_object=args.gs_object, mesh_assets_folder=args.mesh_assets, default_config=args.config)
    engine.use_spp = args.spp > 0
    if args.spp > 0:                    # multi-sample for clean images (non-deterministic)
        engine.spp.mode = "low_discrepancy_seq"
        engine.spp.spp = args.spp
        engine.antialiasing_mode = "Quasi-Random (Sobol)"
    engine.use_optix_denoiser = args.denoise
    engine.shadow_min = args.shadow_min
    engine.shadow_spp = args.shadow_spp
    engine.camera_type = "Pinhole"
    engine.camera_fov = 60.0
    if (args.spp > 0 or args.denoise) and not args.skip_assert:
        print("[warn ] --spp/--denoise add sampling noise; pixel-exact P1/P2 regression may FAIL. Use with --skip_assert for image inspection.")

    # ---- diagnostics + optional room rotation ------------------------------
    def bbox(v):
        v = v.detach()
        lo, hi = v.min(dim=0).values, v.max(dim=0).values
        return lo, hi, (lo + hi) / 2

    def fmt(t):
        return [round(x, 3) for x in t.tolist()]

    def euler_to_R(deg):  # XYZ euler degrees -> [3,3], column convention (x' = R @ x)
        rx, ry, rz = (math.radians(a) for a in deg)
        cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
        Rx = torch.tensor([[1.0, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = torch.tensor([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
        return Rz @ Ry @ Rx

    mog = engine.scene_mog
    _, _, scene_center = bbox(mog.positions)
    R = euler_to_R(args.scene_rot).to(device=mog.positions.device, dtype=mog.positions.dtype)

    def rotate_gs():  # rotate gaussians about scene_center: positions + covariance quat + specular SH
        from threedgrut.export.partition import so3_to_quaternion_wxyz
        from threedgrut.export.sh_rotation import rotate_specular
        from threedgrut.utils.misc import quaternion_to_so3

        mog.positions.data = (mog.positions.data - scene_center) @ R.T + scene_center
        mog.rotation.data = so3_to_quaternion_wxyz(R @ quaternion_to_so3(mog.get_rotation())).to(mog.rotation.dtype)
        spec = mog.features_specular
        deg = int(round((spec.shape[1] / 3 + 1) ** 0.5)) - 1
        if deg > 0:
            mog.features_specular.data = rotate_specular(spec, R, deg)
        mog.build_acc(rebuild=True)   # scene_mog renderer BVH (gs_only path)
        engine.rebuild_bvh(mog)       # playground tracer BVH (HYBRID path) -- must also rebuild

    def transform_catcher(name):  # rotate + scale catcher about scene_center (stays aligned with GS)
        obj = engine.primitives.objects[name]
        obj.vertices = ((obj.vertices - scene_center) @ R.T) * args.catcher_scale + scene_center
        obj.vertex_normals = obj.vertex_normals @ R.T

    if any(args.scene_rot):
        rotate_gs()
        print(f"[rot  ] rotated room by euler {list(args.scene_rot)} deg about center {fmt(scene_center)}")

    gs_lo, gs_hi, gs_c = bbox(mog.positions)
    print(f"[world] GS      bbox min={fmt(gs_lo)} max={fmt(gs_hi)} center={fmt(gs_c)}")
    _cn = engine.load_shadow_catcher(args.catcher)  # peek (rotated + scaled like the scenes)
    transform_catcher(_cn)
    c_lo, c_hi, catcher_center = bbox(engine.primitives.objects[_cn].vertices)
    print(f"[world] catcher bbox min={fmt(c_lo)} max={fmt(c_hi)} center={fmt(catcher_center)} (x{args.catcher_scale})")
    engine.primitives.remove_primitive(_cn)
    _extent_vec = gs_hi - gs_lo
    teapot_target = torch.tensor(args.occluder_pos, device=catcher_center.device, dtype=catcher_center.dtype)
    print(f"[place] occluder -> world pos {fmt(teapot_target)}  (scale {args.occluder_frac:.3f} * extent)")
    print("[aim  ] camera look-at occluder; tune --occluder_pos / --eye_dir / --cam_dist / --scene_rot / --catcher_scale")

    # ---- scene helpers ------------------------------------------------------
    def reset_scene():
        for name in list(engine.primitives.objects.keys()):
            engine.primitives.remove_primitive(name)
        engine.clear_lights()
        engine.disable_gaussian_tracing = False

    def add_occluder():
        before = set(engine.primitives.objects.keys())
        engine.primitives.add_primitive(geometry_type=args.occluder, primitive_type=occluder_type, device="cuda")
        name = (set(engine.primitives.objects.keys()) - before).pop()
        tf = engine.primitives.objects[name].transform
        # add_primitive unit-normalizes the occluder at the ORIGIN. Scale it to a
        # visible fraction of the scene, then move it to the requested world pos.
        tf.scale(args.occluder_frac * float(_extent_vec.max()))
        if any(args.occluder_rot):
            tf.rotate(torch.tensor([math.radians(a) for a in args.occluder_rot], device=teapot_target.device, dtype=teapot_target.dtype))
        tf.translate(teapot_target)
        return name

    def load_catcher():
        name = engine.load_shadow_catcher(args.catcher)
        transform_catcher(name)  # same rotate + 0.98 scale as the GS, about scene_center
        return name

    # Default multi-light rig (used by G_multi_light when --light_dirs is empty):
    # three directional lights from distinct directions -> overlapping shadows.
    MULTI_LIGHT_DIRS = [(0.6, 1.0, 0.4), (-0.6, 1.0, 0.4), (0.6, 1.0, -0.4)]

    def _light_dir_list():
        # --light_dirs 'd1;d2;...' (semicolon-separated) overrides the single --light_dir.
        if args.light_dirs:
            return [tuple(float(x) for x in d.split(",")) for d in args.light_dirs.split(";") if d.strip()]
        return [tuple(args.light_dir)]

    def add_light(angular_radius=0.0):
        for d in _light_dir_list():
            engine.add_light(Light(direction=d, color=(1.0, 1.0, 1.0), intensity=args.intensity, angular_radius=angular_radius))

    def _point_light_pos():
        # Explicit --point_pos, else auto: above the occluder along --light_dir
        # (light_dir points shading-point -> light, so the light sits "up-light").
        if args.point_pos:
            return tuple(float(x) for x in args.point_pos.split(","))
        L = torch.tensor(tuple(args.light_dir), dtype=torch.float32)
        L = L / L.norm().clamp_min(1e-8)
        occ = torch.tensor(tuple(args.occluder_pos), dtype=torch.float32)
        return tuple((occ + L * (args.point_dist * _extent)).tolist())

    def add_point_light(angular_radius=0.0, pos=None):
        engine.add_light(Light(light_type=int(LightType.POINT),
                               position=tuple(pos if pos is not None else _point_light_pos()),
                               color=(1.0, 1.0, 1.0), intensity=args.intensity,
                               angular_radius=angular_radius))

    def add_area_light(pos=None, half=None):
        # Rectangle centered at `pos`, two half-edges perpendicular to --light_dir.
        Ld = torch.tensor(tuple(args.light_dir), dtype=torch.float32)
        Ld = Ld / Ld.norm().clamp_min(1e-8)
        a = torch.tensor([0.0, 1.0, 0.0]) if abs(float(Ld[0])) > 0.1 else torch.tensor([1.0, 0.0, 0.0])
        u = torch.cross(a, Ld, dim=0); u = u / u.norm().clamp_min(1e-8)
        v = torch.cross(Ld, u, dim=0)
        h = half if half is not None else (args.area_size * _extent)
        p = pos if pos is not None else _point_light_pos()
        engine.add_light(Light(light_type=int(LightType.AREA),
                               position=tuple(p),
                               color=(1.0, 1.0, 1.0), intensity=args.intensity,
                               tangent_u=tuple((u * h).tolist()),
                               tangent_v=tuple((v * h).tolist())))

    # ---- explicit scene builders (no lambda-tuple tricks) -------------------
    def build_gs_only():
        pass  # nothing added; pure GS -> aim the camera with this one

    def build_A_baseline():
        add_occluder()

    def build_B_catcher_nolight():
        add_occluder()
        load_catcher()

    def build_C_light_nocatcher():
        add_occluder()
        add_light()

    def build_D_hard_shadow():
        add_occluder()
        load_catcher()
        add_light(angular_radius=0.0)

    def build_E_soft_shadow():
        add_occluder()
        load_catcher()
        add_light(angular_radius=args.soft_angle)

    def build_F_disable_gs():
        add_occluder()
        load_catcher()
        add_light()
        engine.disable_gaussian_tracing = True

    def build_G_multi_light():
        # Multiple directional lights -> overlapping contact shadows on the GS
        # ground. The clearest visual proof directional lights take effect:
        # each light casts its own soft shadow in its own direction.
        add_occluder()
        load_catcher()
        dirs = _light_dir_list() if args.light_dirs else MULTI_LIGHT_DIRS
        for d in dirs:
            engine.add_light(Light(direction=d, color=(1.0, 1.0, 1.0),
                                   intensity=args.intensity, angular_radius=args.soft_angle))

    def build_H_point_light():
        # Single POINT light: its world position (not a global direction) decides
        # the shadow, and the occlusion ray's tmax is clamped to the light distance
        # so geometry behind the light casts no shadow (unlike directional's
        # parallel, infinite-distance rays). Move --point_pos and the shadow moves.
        add_occluder()
        load_catcher()
        pos = _point_light_pos()
        print(f"[H    ] point light @ {tuple(round(float(x), 2) for x in pos)}")
        add_point_light(angular_radius=args.soft_angle, pos=pos)

    def build_I_point_plus_dir():
        # Point light + parallel (directional) light together -> two overlapping
        # contact shadows: one radiating from the point's position, one along --light_dir.
        add_occluder()
        load_catcher()
        pos = _point_light_pos()
        print(f"[I    ] point light @ {tuple(round(float(x), 2) for x in pos)} + directional {tuple(args.light_dir)}")
        add_point_light(angular_radius=args.soft_angle, pos=pos)
        add_light(angular_radius=args.soft_angle)

    def build_J_area_light():
        # Single AREA (rectangle) light -> soft, rectangular contact shadow.
        # Its softness comes from sampling the rect area (not angular_radius), so
        # a bigger --area_size yields a softer, wider penumbra. G-B proof.
        add_occluder()
        load_catcher()
        pos = _point_light_pos()
        print(f"[J    ] area light @ {tuple(round(float(x), 2) for x in pos)} half={args.area_size * _extent:.3f}")
        add_area_light(pos=pos)

    # ---- camera (look-at the catcher center) / render / save ---------------
    _extent = float((gs_hi - gs_lo).max())
    _at = torch.tensor(vec3(args.cam_at), dtype=torch.float32) if args.cam_at else teapot_target.detach().cpu().float()
    _up = torch.tensor(args.up, dtype=torch.float32)
    if args.eye:
        _eye = torch.tensor(vec3(args.eye), dtype=torch.float32)
    else:
        az = math.radians(args.cam_azimuth)
        dx, dy, dz = args.eye_dir
        rdx = dx * math.cos(az) + dz * math.sin(az)   # rotate (dx,dz) about world Y
        rdz = -dx * math.sin(az) + dz * math.cos(az)
        _d = torch.tensor([rdx, dy, rdz], dtype=torch.float32)
        _eye = _at + _d / _d.norm() * (_extent * args.cam_dist)
    print(f"[cam  ] look-at at={fmt(_at)} eye={fmt(_eye)} up={list(args.up)} extent={_extent:.2f}")

    def make_camera():
        return kaolin.render.camera.Camera.from_args(
            eye=_eye, at=_at, up=_up,
            fov=math.radians(args.fov), width=args.res, height=args.res, device="cuda",
        )

    @torch.no_grad()
    def render():
        # Force-rebuild the mesh BVH so the just-added occluder/catcher are in it.
        engine.primitives.rebuild_bvh_if_needed(force=True, rebuild=True)
        fb = engine.render(make_camera())
        return fb["rgb"][0].float().clamp(0.0, 1.0).cpu()  # (H,W,3)

    def save(img, tag):
        path = os.path.join(args.out, f"{tag}.png")
        try:
            import torchvision

            torchvision.utils.save_image(img.permute(2, 0, 1), path)
        except Exception:  # noqa: BLE001
            path = path.replace(".png", ".pt")
            torch.save(img, path)
        return path

    scene_builders = (
        ("gs_only", build_gs_only),
        ("A_baseline", build_A_baseline),
        ("B_catcher_nolight", build_B_catcher_nolight),
        ("C_light_nocatcher", build_C_light_nocatcher),
        ("D_hard_shadow", build_D_hard_shadow),
        ("E_soft_shadow", build_E_soft_shadow),
        ("F_disable_gs", build_F_disable_gs),
        ("G_multi_light", build_G_multi_light),
        ("H_point_light", build_H_point_light),
        ("I_point_plus_dir", build_I_point_plus_dir),
        ("J_area_light", build_J_area_light),
    )
    builder_by_tag = dict(scene_builders)
    results = {}
    timings = {}
    statuses = {}

    def run(tag, builder):
        _cuda_synchronize_for_timing()
        started = time.perf_counter()
        try:
            reset_scene()
            builder()
            # >>> confirm what is ACTUALLY in the scene before rendering <<<
            prims = list(engine.primitives.objects.keys())
            print(f"[{tag}] primitives={prims}  lights={len(engine.lights)}  disable_gs={engine.disable_gaussian_tracing}")
            img = render()
        except Exception as e:  # noqa: BLE001
            _cuda_synchronize_for_timing()
            elapsed = time.perf_counter() - started
            timings[tag] = elapsed
            statuses[tag] = "CRASHED"
            print(f"[{tag}] !!! CRASHED: {e}  time={elapsed:.3f}s")
            results[tag] = None
            return
        results[tag] = img
        p = save(img, tag)
        _cuda_synchronize_for_timing()
        elapsed = time.perf_counter() - started
        timings[tag] = elapsed
        statuses[tag] = "OK"
        print(f"[{tag}] mean={img.mean():.5f}  saved {p}  time={elapsed:.3f}s")

    def print_timing_summary():
        print("\n=== timing summary (end-to-end per selected scene) ===")
        for tag in selected_scenes:
            elapsed = timings.get(tag)
            status = statuses.get(tag, "SKIPPED")
            if elapsed is None:
                print(f"[timing] {tag}: {status}")
            else:
                print(f"[timing] {tag}: {elapsed:.3f}s  status={status}")
        total = sum(timings.get(tag, 0.0) for tag in selected_scenes)
        print(f"[timing total] {total:.3f}s for {len(selected_scenes)} scene(s)")

    print("=== rendering scenes (check the [tag] lines: primitives/lights must be non-empty as expected) ===")
    print(f"[scenes] selected={','.join(selected_scenes)}")
    for tag in selected_scenes:
        run(tag, builder_by_tag[tag])

    # Shadow diagnostic: B (catcher, no light) - D (catcher + light) = darkening.
    # Normalized so even a faint shadow is visible; the printed max is the true strength.
    _b, _d = results.get("B_catcher_nolight"), results.get("D_hard_shadow")
    if _b is not None and _d is not None:
        _dark = (_b - _d).clamp(min=0.0).mean(dim=-1, keepdim=True).expand(-1, -1, 3)
        _m = _dark.max()
        save(_dark / _m if _m > 1e-8 else _dark, "shadow_diff")
        print(f"\n[shadow] max darkening B-D = {_m.item():.5f}  (shadow_diff.png normalized; bright = where the teapot shadows the GS)")

    if args.skip_assert:
        print("\n--skip_assert set: inspect the PNGs in", args.out)
        print_timing_summary()
        return 0

    # ---- sanity checks ------------------------------------------------------
    print("\n=== checks (heuristic; the PNGs are the real ground truth) ===")
    a, b, c, d = (results.get(k) for k in ("A_baseline", "B_catcher_nolight", "C_light_nocatcher", "D_hard_shadow"))
    if any(x is None for x in (a, b, c, d)):
        print("some scene failed to render; cannot run asserts. Inspect the logs/PNGs.")
        print_timing_summary()
        return 1

    def maxdiff(x, y):
        return (x - y).abs().max().item()

    p1, p2 = maxdiff(a, b), maxdiff(a, c)
    p3_max = (b - d).clamp(min=0.0).max().item()
    p3_mean = (b.mean() - d.mean()).item()
    ok = True
    print(f"[P1 transparency] max|A-B|={p1:.5f} (tol {args.tol}) -> {'PASS' if p1 <= args.tol else 'FAIL'}")
    print(f"[P2 light no-op ] max|A-C|={p2:.5f} (tol {args.tol}) -> {'PASS' if p2 <= args.tol else 'FAIL'}")
    print(f"[P3 hard shadow ] max(B-D)={p3_max:.5f}, mean darken={p3_mean:+.5f} -> {'PASS' if p3_max > args.tol else 'FAIL'}")
    print(f"[F disable-GS   ] {'PASS (no crash)' if results.get('F_disable_gs') is not None else 'FAIL (crashed)'}")
    ok &= p1 <= args.tol and p2 <= args.tol and p3_max > args.tol and results.get("F_disable_gs") is not None

    # ---- G-B area light check: the AREA light must cast a contact shadow ------
    j = results.get("J_area_light")
    if j is not None:
        jb_max = (b - j).clamp(min=0.0).max().item()
        print(f"[G-B area light] max(B-J)={jb_max:.5f} (tol {args.tol}) -> {'PASS' if jb_max > args.tol else 'FAIL'}")
        ok &= jb_max > args.tol
    else:
        print("[G-B area light] J_area_light failed to render -> FAIL")
        ok = False

    print("\ninspect:", args.out, "(esp. D_hard_shadow.png vs B_catcher_nolight.png)")
    print("OVERALL:", "PASS" if ok else "FAIL")
    print_timing_summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
