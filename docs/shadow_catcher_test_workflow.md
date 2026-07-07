# Shadow-catcher 测试脚本运行文档

本文档说明如何运行 `test_shadow_catcher_p1p3.py`，如何使用 JSON preset 配置相机、光源和测试参数，以及如何解读输出结果。

## 脚本定位

`test_shadow_catcher_p1p3.py` 是一个 headless 功能测试脚本，用来验证 shadow-catcher Phase 1/2/3：

- P1: shadow catcher 自身透明，不应改变 baseline 图像。
- P2: 没有 catcher 时，加入 light 不应改变 baseline 图像。
- P3: 有 catcher、有 occluder、有 light 时，GS ground 应出现阴影。

脚本会依次渲染多个场景，保存 PNG 到 `--out`，并在未设置 `--skip_assert` 时运行数值 sanity check。

## 运行前检查

脚本需要 CUDA/OptiX 环境，并会触发 `threedgrut_playground` 的 CUDA/OptiX JIT 扩展：

```bash
git submodule update --init --recursive
ls threedgrt_tracer/dependencies/optix-dev/include/optix.h
nvcc --version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`--denoise` 依赖 OptiX Denoiser。建议只在视觉验收时打开，并配合 `--skip_assert`。

## 最小运行命令

需要两个输入：

- `--gs_object`: 3DGS/3DGRT checkpoint，支持 `.pt/.ingp/.ply`。
- `--catcher`: 外部生成并已经对齐到 GS ground 的 catcher mesh，支持 `.obj/.glb/.gltf`。

示例：

```bash
python test_shadow_catcher_p1p3.py \
  --gs_object /path/to/ckpt_last.pt \
  --catcher /path/to/ground_catcher.obj \
  --occluder Sphere \
  --out ./shadow_test_out/debug
```

当前仓库里 `threedgrut_playground/assets` 只看到 `sphere.obj`，它会注册成 asset key `Sphere`。脚本默认 `--occluder Teapot`，如果运行环境里没有 `Teapot.obj/.glb/.gltf`，请改成已有资产，例如 `--occluder Sphere`，或把 teapot 资产放进 `--mesh_assets` 指向的目录。

## 推荐两阶段流程

先跑可重复的 correctness check：

```bash
python test_shadow_catcher_p1p3.py \
  --gs_object /path/to/ckpt_last.pt \
  --catcher /path/to/ground_catcher.obj \
  --occluder Sphere \
  --out ./shadow_test_out/regression
```

再跑视觉检查版，使用高 SPP 和 OptiX Denoiser：

```bash
python test_shadow_catcher_p1p3.py \
  --gs_object /path/to/ckpt_last.pt \
  --catcher /path/to/ground_catcher.obj \
  --occluder Sphere \
  --spp 32 \
  --denoise \
  --skip_assert \
  --out ./shadow_test_out/visual
```

原因：`--spp` 和 `--denoise` 会改变像素值，可能破坏 P1/P2 的 pixel-exact 回归检查。它们更适合看图验收。

## 只渲染选定场景

默认 `--scenes all` 会保持旧行为，按固定顺序渲染所有测试场景。调相机、调光源或只看某几种光源时，可以用 `--scenes` 只跑几张图：

```bash
python test_shadow_catcher_p1p3.py \
  --gs_object /path/to/ckpt_last.pt \
  --catcher /path/to/ground_catcher.obj \
  --occluder Sphere \
  --scenes D,E,J \
  --skip_assert \
  --out ./shadow_test_out/selected
```

只跑部分场景时通常需要 `--skip_assert`。如果没有设置 `--skip_assert`，脚本要求断言依赖场景 `A,B,C,D,F,J` 都被渲染，否则会提前报错并列出缺失场景。

`shadow_diff` 是派生输出请求，不是独立渲染场景。传入 `--scenes shadow_diff --skip_assert` 时，脚本会自动渲染 B 和 D，并在二者都成功后输出 `shadow_diff.png`。

每个被选中的场景都会打印端到端单帧耗时。计时范围从 `reset_scene()` 前开始，到图片保存完成后结束，包含场景搭建、BVH rebuild、render、CPU copy 和保存图片；这不是纯 GPU kernel 时间。

## JSON preset 机制

脚本支持 `--preset <json>`。它会先读取 preset JSON，把里面的字段作为 argparse 默认值，然后再解析命令行。命令行显式传入的参数会覆盖 preset。

已有 preset：

```bash
python test_shadow_catcher_p1p3.py \
  --preset presets/r5strict.json \
  --gs_object /path/to/ckpt_last.pt \
  --catcher /path/to/ground_catcher.obj \
  --occluder Sphere
```

`presets/r5strict.json` 当前内容：

```json
{
  "occluder_pos": [2.0, -4.0, 5.0],
  "occluder_frac": 0.25,
  "light_dir": [0.6, 1.0, 0.4],
  "scene_rot": [180.0, 0.0, 0.0],
  "cam_dist": 0.35,
  "cam_azimuth": 180.0,
  "spp": 512,
  "skip_assert": true,
  "out": "./shadow_test_out/r5strict"
}
```

注意：

- `--config` 不是这个 preset。`--config` 是加载 `.ingp/.ply` 或非 3dgrt `.pt` 时使用的 3DGRUT/3DGRT Hydra config，默认是 `apps/colmap_3dgrt.yaml`。
- CLI 的 vec3 参数写成 `"x,y,z"`，例如 `--light_dir "0.6,1,0.4"`。
- JSON preset 里可以直接写数组，例如 `"light_dir": [0.6, 1.0, 0.4]`。
- `--save_preset path.json` 会把解析后的配置写成 JSON，然后继续渲染，不是 dry-run。

## 常用 preset 模板

调相机和位置时，可以先跳过断言：

```json
{
  "gs_object": "/path/to/ckpt_last.pt",
  "catcher": "/path/to/ground_catcher.obj",
  "mesh_assets": "threedgrut_playground/assets",
  "occluder": "Sphere",
  "occluder_type": "PBR",
  "occluder_pos": [0.0, -1.0, 0.0],
  "occluder_frac": 0.08,
  "occluder_rot": [0.0, 0.0, 0.0],
  "catcher_scale": 0.98,
  "scene_rot": [0.0, 0.0, 0.0],
  "out": "./shadow_test_out/tune",
  "res": 512,
  "scenes": ["D", "E", "J"],
  "eye_dir": [0.0, 0.4, -1.0],
  "cam_dist": 0.25,
  "up": [0.0, 1.0, 0.0],
  "fov": 60.0,
  "cam_azimuth": 0.0,
  "spp": 0,
  "denoise": false,
  "light_dir": [0.0, -1.0, 0.0],
  "intensity": 3.0,
  "soft_angle": 0.05,
  "shadow_min": 0.0,
  "shadow_spp": 128,
  "tol": 0.02,
  "skip_assert": true
}
```

运行：

```bash
python test_shadow_catcher_p1p3.py --preset presets/my_shadow_debug.json
```

## 重要参数

输入和资产：

- `gs_object`: 必填，除非 preset 已提供。3DGS checkpoint。
- `catcher`: 必填，除非 preset 已提供。外部 shadow-catcher mesh。脚本按原始坐标加载，不会 recenter 或 autoscale。
- `mesh_assets`: occluder 资产目录，默认 `threedgrut_playground/assets`。
- `occluder`: occluder asset key。由资产文件名 stem 首字母大写得到，例如 `sphere.obj` -> `Sphere`。也支持 procedural `Quad`。
- `occluder_type`: `PBR` 或 `DIFFUSE`。

摆位：

- `occluder_pos`: occluder 世界坐标。
- `occluder_frac`: occluder 尺寸，占 scene extent 的比例。
- `occluder_rot`: occluder XYZ 欧拉角，单位 degree。
- `catcher_scale`: 以 scene center 为中心缩放 catcher，默认 `0.98`，用于让 catcher 略微缩到 GS surface 内。
- `scene_rot`: 同时旋转 GS 和 catcher，单位 degree。

相机：

- `eye_dir`: 从 occluder 指向 camera eye 的方向。
- `cam_dist`: eye 距离，等于 `cam_dist * scene_extent`。
- `cam_azimuth`: 绕世界 Y 轴水平旋转 camera。
- `eye`: 绝对 eye 坐标。设置后覆盖 `eye_dir/cam_dist/cam_azimuth`。
- `cam_at`: 绝对 look-at 坐标。默认看向 `occluder_pos`。
- `up`: camera up。
- `fov`: 垂直视场角，单位 degree。
- `res`: 方形输出分辨率。

质量：

- `spp`: 抗锯齿样本数。`0` 表示关闭，用于确定性回归；`32-64` 适合视觉检查。
- `denoise`: 打开 OptiX Denoiser。建议配合 `skip_assert`。
- `scenes`: 选择要渲染的场景。默认 `all`。CLI 使用逗号分隔字符串；preset 可写字符串或 JSON 数组。
- `skip_assert`: 只渲染并保存输出图，不做数值断言。
- `tol`: P1/P2/P3 sanity check 阈值，默认 `2e-2`。

光源和阴影：

- `light_dir`: 方向光方向，语义是从 shading point 指向 light 的世界空间向量。
- `light_dirs`: 多方向光列表，用分号分隔，例如 `"0.6,1,0.4;-0.6,1,0.4;0.6,1,-0.4"`。只影响使用 `_light_dir_list()` 的场景；`G_multi_light` 在为空时使用脚本内置三方向光。
- `intensity`: 光源强度。
- `soft_angle`: 方向光/点光软阴影角半径，单位 radians。`0` 为硬阴影。
- `shadow_min`: 阴影下限。`0` 表示可到全黑，`0.2` 表示最暗保留 20% 亮度。
- `shadow_spp`: 软阴影 occlusion samples per light。
- `point_pos`: 点光源世界坐标。为空时自动用 `occluder_pos + normalize(light_dir) * point_dist * scene_extent`。
- `point_dist`: 自动点光距离系数。
- `area_size`: 面积光矩形半边长系数，实际半边长为 `area_size * scene_extent`。

## 脚本会渲染哪些场景

默认 `--scenes all` 会按顺序输出这些文件；使用 `--scenes` 时只输出被选中的场景，以及满足 B/D 依赖时的 `shadow_diff.png`：

- `gs_only.png`: 纯 GS，不加 occluder/catcher/light。用来先瞄准相机。
- `A_baseline.png`: 只加 occluder，无 catcher，无 light。
- `B_catcher_nolight.png`: occluder + catcher，无 light。P1 期望接近 A。
- `C_light_nocatcher.png`: occluder + light，无 catcher。P2 期望接近 A。
- `D_hard_shadow.png`: occluder + catcher + 硬方向光。P3 期望比 B 变暗。
- `E_soft_shadow.png`: occluder + catcher + 软方向光。
- `F_disable_gs.png`: D 场景但关闭 gaussian tracing，用于确认不崩溃。
- `G_multi_light.png`: 多方向光，观察多重叠加阴影。
- `H_point_light.png`: 点光源。
- `I_point_plus_dir.png`: 点光源 + 方向光。
- `J_area_light.png`: 面积光。
- `shadow_diff.png`: `B_catcher_nolight - D_hard_shadow` 的归一化差异图，亮处表示 hard shadow 造成的 darkening。

`--scenes` 支持以下 alias：

| Alias | 场景名 |
| --- | --- |
| `gs_only` | `gs_only` |
| `A` | `A_baseline` |
| `B` | `B_catcher_nolight` |
| `C` | `C_light_nocatcher` |
| `D` | `D_hard_shadow` |
| `E` | `E_soft_shadow` |
| `F` | `F_disable_gs` |
| `G` | `G_multi_light` |
| `H` | `H_point_light` |
| `I` | `I_point_plus_dir` |
| `J` | `J_area_light` |
| `shadow_diff` | 自动渲染 `B_catcher_nolight` 和 `D_hard_shadow` 后生成差异图 |

## 断言逻辑

未设置 `--skip_assert` 时，脚本会检查：

- P1 transparency: `max|A - B| <= tol`
- P2 light no-op: `max|A - C| <= tol`
- P3 hard shadow: `max(B - D) > tol`
- F disable-GS: 场景未崩溃
- Area light: `max(B - J) > tol`

这些是 sanity heuristics，PNG 才是最终视觉判断依据。调相机、调光源、开 SPP 或 Denoiser 时，优先使用 `--skip_assert`。

如果通过 `--scenes` 只跑部分场景且没有设置 `--skip_assert`，脚本会在加载 CUDA/OptiX 前提前报错。要运行完整数值断言，至少需要选择 `A,B,C,D,F,J`。

## 调参建议

1. 先只看 `gs_only.png`，确认 camera 对准场景。
2. 调 `occluder_pos` 和 `occluder_frac`，让 occluder 明显处在 catcher/ground 上方。
3. 调 `light_dir` 或 `point_pos`，让阴影落在相机可见区域。
4. 如果 catcher 和 GS ground 不贴合，先调外部 catcher 生成流程；脚本只提供 `catcher_scale` 和 `scene_rot` 作为轻量修正。
5. 确认可见后，再增加 `spp`、打开 `denoise`，并使用 `--skip_assert` 做视觉输出。

## 常见问题

`KeyError: 'Teapot'` 或找不到 occluder：

- 当前资产目录没有 `Teapot`。改用 `--occluder Sphere`，或把 `teapot.obj/.glb/.gltf` 放到 `--mesh_assets`。

找不到 `optix.h`：

- 子模块未初始化。运行 `git submodule update --init --recursive`。

打开 `--denoise` 后断言失败：

- 这是预期风险。Denoiser 会改变最终 `rgb`，使用 `--skip_assert` 做视觉检查。

输出是 `.pt` 不是 `.png`：

- `torchvision.utils.save_image` 导入或保存失败，脚本会 fallback 到 `torch.save`。检查 `torchvision` 安装。

阴影太黑或太浅：

- 太黑可调高 `shadow_min`，例如 `0.2`。
- 太浅可提高 `intensity`、移动 light/occluder，或检查 occluder 是否真的挡在 catcher 到 light 的路径上。
