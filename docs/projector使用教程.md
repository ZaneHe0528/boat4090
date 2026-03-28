# Projector（相机地面投影）使用教程

本文说明 `scripts/realtime_pilot.py` 中 **开启 / 不开启 projector** 的用法、相关代码位置，以及需要改代码时的修改要点。

---

## 1. 两种模式分别是什么

| 模式 | 触发条件 | 含义 |
|------|----------|------|
| **河宽比例法**（无 projector） | **不传** `--camera-height` | 用 `--river-width` 与每行水面像素宽度估算「米」，不依赖真实相机内参 |
| **相机投影**（有 projector） | **传入** `--camera-height` | 使用 `IMX390Projector` 针孔模型，把像素投到水面平面，得到前向/横向米坐标 |

启动时终端会打印当前模式，例如：

- `坐标模式: 河宽比例法 | 河道宽度=2.5m`
- `坐标模式: 相机投影 | IMX390-GMSL2 | HFOV=...`

---

## 2. 命令行使用方法

### 2.1 不开启 projector（默认）

不传 `--camera-height` 即可。

```bash
python scripts/realtime_pilot.py \
  --images dataset_final/images/test \
  --model models/segformer_river/best_model.pth \
  --output vis_out
```

可调参数（与「米」相关）：

- `--river-width`：河道实际宽度（米），默认 `2.5`。与分割 mask 里每行水面宽度一起决定水平比例尺。
- `--shore-stop-dist`：岸边截断阈值（米），路径在距岸约该距离处截断。

### 2.2 开启 projector

**必须**提供 `--camera-height`（相机光心到水面的垂直高度，单位米）。  
俯仰与水平视场角有默认值，可按实际镜头调整。

```bash
python scripts/realtime_pilot.py \
  --images dataset_final/images/test \
  --model models/segformer_river/best_model.pth \
  --output vis_out \
  --camera-height 0.5 \
  --camera-pitch 10.0 \
  --camera-hfov 120.6
```

参数说明（定义在 `parse_args()` 的 `IMX390-GMSL2 相机参数` 组）：

| 参数 | 含义 | 默认 |
|------|------|------|
| `--camera-height` | 相机离水面高度（米），**不传则不启用 projector** | 无（None） |
| `--camera-pitch` | 俯仰角（度），>0 表示光轴向下倾 | `10.0` |
| `--camera-hfov` | 水平视场角（度）。文档注释示例：H60≈63.9、H120≈120.6 | `120.0` |

相机实时模式同样生效：在原有 `--camera 0 --model ...` 基础上加上述 `--camera-height` 等即可。

---

## 3. 与哪些代码相关

### 3.1 入口：是否创建 projector

文件：`scripts/realtime_pilot.py`  
函数：`main()`

逻辑要点：

- `if args.camera_height is not None:` 时构造 `IMX390Projector(...)`，并赋给 `args._projector`
- 否则 `projector = None`

### 3.2 参数解析

文件：`scripts/realtime_pilot.py`  
函数：`parse_args()`

- `--camera-height`、`--camera-pitch`、`--camera-hfov` 在 `cam = p.add_argument_group('IMX390-GMSL2 ...')` 中注册。

### 3.3 投影器类本身

文件：`scripts/realtime_pilot.py`  
类：`IMX390Projector`

- 构造函数参数：`camera_height`、`camera_pitch_deg`、`hfov_deg`
- 关键方法：`update_resolution(img_w, img_h)`、`pixel_to_ground(u, v) -> (x_m, y_m)`  
  其中 `x_m` 为前向、`y_m` 为横向（右正左负），原点在「相机正下方水面点」的简化船体坐标系。

### 3.4 单帧流水线里 projector 的用法

文件：`scripts/realtime_pilot.py`  
函数：`process_frame(...)`

`projector` 会传给：

1. **`_truncate_path_at_shore`**：岸距（米）。有 projector 时，用路径点与**同一行最近边界像素**两点投影到地面后算欧氏距离；投影失败则退回河宽比例。
2. **`_compute_path_mileage`**：里程 `x_m`、横向 `y_m`、CSV 里每点 `heading_deg` 所用的 `_gx/_gy`。有 projector 时全程用 `pixel_to_ground`；无 projector 时用河宽比例 + 内部固定的 `_heading_proj`（见下文注意点）。
3. **HUD 中的 `heading_deg`**：有 projector 时用预瞄目标点的地面坐标 `atan2`；无 projector 时用像素路径切线近似。

图片/相机循环里：`run_on_images` / `run_on_camera` 把 `args._projector` 作为 `projector=` 传入 `process_frame`。

---

## 4. 「米」在无 projector / 有 projector 下怎么算

### 无 projector

- **横向偏移、累计里程**：按行用 `scale = river_width_m / 该行水面像素宽度`，像素位移 × scale（或相邻点平均 scale）得到米。
- **岸距**：同一行上路径点到最近 `BOUNDARY` 像素的**水平像素差** × 该行 `scale`。
- **局限**：假设「整行河宽对应 `river_width_m`」，未建模真实透视与相机高度，适合快速跑通或标定粗的情况。

### 有 projector

- **里程、目标点世界坐标**：相邻路径点地面坐标差分累加 / 查表。
- **岸距**：路径点与同行最近边界点分别 `pixel_to_ground`，再算平面距离。
- **前提**：`camera_height`、`camera_pitch`、`hfov` 与真实安装、镜头一致；图像分辨率变化由 `update_resolution` 按比例缩放焦距。

---

## 5. 需要改代码时改哪里、怎么改

### 5.1 只换安装高度 / 镜头（仍是 IMX390 + 针孔近似）

**优先改命令行**，无需改代码：

```bash
--camera-height <实测米> --camera-pitch <度> --camera-hfov <度>
```

### 5.2 修改 IMX390 类内「写死」的传感器基准

文件：`scripts/realtime_pilot.py` → `IMX390Projector` 类常量，例如：

- `NATIVE_W`、`NATIVE_H`（基准分辨率）
- `native_fx` 由 `hfov_deg` 推导

若你的视频不是从该基准分辨率缩放来的，应保证 `update_resolution` 与真实 `img_w/img_h` 一致（当前已在每帧调用）。

### 5.3 非 IMX390 相机或需完整内参矩阵

当前实现是 **单组 HFOV + 高度 + 俯仰** 的简化针孔模型，**没有** `cx, cy, fx, fy` 独立标定文件。

若要接真实标定：

1. 新增或替换投影类（例如读 `camera_matrix` + `distCoeffs` + 水面平面）。
2. 在 `main()` 里用新类替代 `IMX390Projector` 的构造。
3. 保持与 `process_frame` 相同的接口：`update_resolution(w,h)`、`pixel_to_ground(u,v) -> Optional[Tuple[float,float]]`（前向、横向米）。

### 5.4 调整河宽比例（无 projector）

改 CLI `--river-width`，或改 `process_frame(..., river_width_m=...)` 的调用处默认值（不推荐硬编码，优先 CLI）。

### 5.5 无 projector 时里程表 heading 仍用固定投影器

在 `_compute_path_mileage` 的**模式 2（河宽比例）**里，存在：

```python
_heading_proj = IMX390Projector(camera_height=0.5, camera_pitch_deg=10.0, hfov_deg=120.0)
```

它仅用于 `_fill_heading` 所需的 `_gx/_gy`，**与**你是否传入 `--camera-height` **无关**。若你从未开 projector，但希望 CSV 里每点 `heading_deg` 更可信，可考虑：

- 与真实相机一致的参数替换上述常量，或  
- 改为纯像素几何算 heading（需改 `_compute_path_mileage` 分支逻辑）。

---

## 6. 快速对照表

| 需求 | 操作 |
|------|------|
| 快速离线跑图、不关心绝对米精度 | 不传 `--camera-height`，调 `--river-width` |
| 船上/仿真要较可信的米与岸距 | 传 `--camera-height`，并标定 `pitch`/`hfov` |
| 改截断距离 | `--shore-stop-dist` |
| 换相机型号 | 扩展或替换 `IMX390Projector`，并在 `main()` 中实例化 |

---

## 7. 相关文件路径小结

| 内容 | 路径 |
|------|------|
| CLI、main、两种模式打印 | `scripts/realtime_pilot.py` |
| 投影类 | 同上，`IMX390Projector` |
| 岸距截断 | 同上，`_truncate_path_at_shore` |
| 里程与 CSV heading | 同上，`_compute_path_mileage`、`_fill_heading` |
| 路径规划（与 projector 无关） | `scripts/plan_path.py` 等 |

---

*文档对应仓库内 `realtime_pilot.py` 逻辑；若代码变更，请以源码为准并同步更新本节。*
