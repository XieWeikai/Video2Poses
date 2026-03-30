# Video2Poses

Video2Poses 做两件事：

1. 把 MapAnything 包装成一个可以长期运行的 HTTP service。
2. 提供一个视频批处理脚本，把一个目录里的视频转成每个视频一个 `*-camera.json` 输出。

这份 README 只讲“怎么跑通”：

1. 如何 build Docker 镜像
2. 有镜像后如何启动服务
3. 如何用 `uv run` 启动视频导出脚本

## 0. 前置条件

- 操作目录：仓库根目录
- 已安装：
  - `docker`
  - `ffmpeg`
  - `ffprobe`
  - `uv`
- 模型目录可用，例如：`path/to/mapanything`
- 如果服务跑在 Docker 里，所有送到 `/infer` 的图片路径必须在容器里可见

建议先同步 Python 环境：

```bash
cd path/to/Video2Poses
uv sync
```

---

## 1. Build 镜像

### 1.1 命令

```bash
cd path/to/Video2Poses
docker build \
  -f docker/mapanything-service.Dockerfile \
  -t video2poses-mapanything-service:latest \
  .
```

### 1.2 参数说明

- `docker build`
  构建镜像。
- `-f docker/mapanything-service.Dockerfile`
  指定 Dockerfile 路径。
- `-t video2poses-mapanything-service:latest`
  指定镜像名和 tag。后面启动服务脚本默认就会使用这个名字。
- `.`
  build context，必须是仓库根目录，这样 `docker/`、`service/`、`third_party/`、`configs/` 都能被打进镜像。

### 1.3 什么时候需要重新 build

- `docker/mapanything-service.Dockerfile`
  变了
- `docker/entrypoint.sh`
  变了
- `service/`
  里的服务代码变了
- 镜像不存在

如果镜像已经 build 好了，不需要重复 build。

---

## 2. 启动服务

### 2.1 推荐方式

仓库里已经有一个启动脚本：

- `scripts/run_mapanything_service_docker.sh`

它会：

- 使用已有镜像启动容器
- 挂载模型目录
- 挂载配置文件
- 挂载 torch cache
- 允许你额外挂载宿主机路径给容器使用

### 2.2 4 GPU 启动示例

仓库里已经提供了 4 卡配置：

- `configs/service/mapanything_infer_service_4gpu.yaml`

启动命令：

```bash
cd path/to/Video2Poses

mkdir -p ./tmp/docker_visible_frames

bash scripts/run_mapanything_service_docker.sh \
  --container-name video2poses-mapanything-service-4gpu \
  --model-dir path/to/map-anything \
  --config-path ./configs/service/mapanything_infer_service_4gpu.yaml \
  --host-port 18080 \
  --mount "$(pwd)/tmp/docker_visible_frames:$(pwd)/tmp/docker_visible_frames:ro" \
  --gpus all
```

这里的 `--mount` 非常重要：

- 视频导出脚本会先切帧到临时目录
- service 只接受图片路径，不接受视频路径
- 所以“临时帧所在目录”必须对容器可见
- 最简单的方式就是把宿主机路径挂载到容器里的同一个绝对路径

### 2.3 启动脚本参数说明

`scripts/run_mapanything_service_docker.sh` 支持下面这些参数：

- `--image-tag`
  要启动的镜像名。默认：`video2poses-mapanything-service:latest`
- `--container-name`
  容器名。默认：`video2poses-mapanything-service`
- `--host-port`
  宿主机暴露端口。默认：`18080`
- `--host-addr`
  仅用于最后打印“从宿主机如何访问”的 URL。默认：`127.0.0.1`
  这里不要写 `0.0.0.0` 当作客户端访问地址；`0.0.0.0` 是监听地址，不是稳定可访问的目标地址。
- `--gpus`
  Docker `--gpus` 参数。可以写 `all`，也可以写具体设备表达式。
- `--model-dir`
  宿主机上的模型目录，必填。
- `--config-path`
  宿主机上的 service YAML 配置文件路径。
- `--torch-cache-dir`
  宿主机 torch cache 目录。默认：`${HOME}/.cache/torch`
- `--mount`
  额外挂载路径，格式：
  `host_path:container_path[:mode]`
  这个参数可以重复写多次。

### 2.4 服务配置文件说明

service 配置是一个 YAML 文件，顶层有两个部分：

- `service`
- `model`

#### `service` 段

- `host`
  服务监听地址。容器内通常写 `0.0.0.0`
- `port`
  服务监听端口。容器内默认 `18080`
- `log_level`
  uvicorn 日志级别
- `request_timeout_sec`
  单次 `/infer` 请求的超时时间

#### `model` 段

- `model_dir`
  模型目录，容器会将模型目录挂在 `/models/map-anything`，故不用修改就填`/models/map-anything`即可。
- `devices`
  启动哪些 GPU worker，例如：
  - `cuda0`
  - `cuda1`
  - `cuda2`
  - `cuda3`
- `infer`
  直接透传给 `model.infer(...)` 的推理参数

#### `model.infer` 段

- `memory_efficient_inference`
  是否启用更省显存的推理路径
- `minibatch_size`
  模型内部 mini-batch 大小。`null` 表示使用默认行为
- `use_amp`
  是否启用自动混合精度
- `amp_dtype`
  AMP 精度类型，例如 `bf16`
- `apply_mask`
  是否应用输出 mask
- `mask_edges`
  是否在边缘区域应用 mask
- `apply_confidence_mask`
  是否按置信度过滤
- `confidence_percentile`
  置信度过滤阈值百分位
- `use_multiview_confidence`
  是否启用多视角 confidence 逻辑
- `multiview_conf_depth_abs_thresh`
  多视角深度绝对阈值
- `multiview_conf_depth_rel_thresh`
  多视角深度相对阈值

### 2.5 仓库里的配置文件

- `configs/service/mapanything_infer_service.yaml`
  单卡模板
- `configs/service/mapanything_infer_service_4gpu.yaml`
  4 卡模板

4 卡配置当前内容是：

```yaml
service:
  host: 0.0.0.0
  port: 18080
  log_level: info
  request_timeout_sec: 1800

model:
  model_dir: path/in/container/mapanything
  devices:
    - cuda0
    - cuda1
    - cuda2
    - cuda3
  infer:
    memory_efficient_inference: true
    minibatch_size: null
    use_amp: true
    amp_dtype: bf16
    apply_mask: true
    mask_edges: true
    apply_confidence_mask: false
    confidence_percentile: 10
    use_multiview_confidence: false
    multiview_conf_depth_abs_thresh: 0.02
    multiview_conf_depth_rel_thresh: 0.02
```

### 2.6 如何确认服务启动成功

健康检查：

```bash
cd path/to/Video2Poses
uv run python - <<'PY'
from video2poses.image_pose_cli import ImagePoseClient
client = ImagePoseClient("http://127.0.0.1:18080", request_timeout_sec=10)
health = client.health()
print({
    "ok": health.ok,
    "devices": health.devices,
    "pending": health.pending,
})
PY
```

注意两点：

- `scripts/run_mapanything_service_docker.sh` 里容器内部仍然监听 `0.0.0.0:18080`，但宿主机访问时应该用 `http://127.0.0.1:18080` 或宿主机真实 IP，不要用 `http://0.0.0.0:18080`
- 服务会在 FastAPI startup 阶段等待所有 GPU worker 把模型加载完；在日志出现 `Application startup complete.` 之前，`/health` 可能还访问不到

如果是 4 卡服务，通常会看到：

```python
{
  "ok": True,
  "devices": ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
  "pending": (0, 0, 0, 0),
}
```

### 2.7 如何关闭服务

如果容器是通过 `scripts/run_mapanything_service_docker.sh` 启动的，它带了 `--rm`，停止后会自动删除：

```bash
docker stop video2poses-mapanything-service-4gpu
```

---

## 3. 从视频目录导出输出目录

### 3.1 启动脚本

视频导出脚本是：

- `scripts/run_video_pose_cli.py`

它会：

1. 扫描输入目录里的视频文件
2. 用 `ffmpeg` 切帧到临时目录
3. 调用 `video2poses.image_pose_cli.ImagePoseClient`
4. 按分块推理
5. 把多块结果对齐到同一个世界坐标系
6. 写出每个视频对应的 `*-camera.json`

### 3.2 输出目录规则

如果输入目录是：

```text
XXX
```

默认输出目录是：

```text
XXX-camera-info
```

如果输入视频是：

```text
xxx.mp4
```

输出文件是：

```text
xxx-camera.json
```

### 3.3 推荐命令

```bash
mkdir -p ./tmp/docker_visible_frames

uv run scripts/run_video_pose_cli.py \
  --input-dir path/to/video_dir \
  --output-dir path/to/video_dir-camera-info \
  --server-url http://127.0.0.1:18080 \
  --sample-fps 1.0 \
  --initial-max-inference-frames 16 \
  --min-inference-frames 2 \
  --initial-max-concurrent 2 \
  --max-video-workers 2 \
  --max-retries 5 \
  --request-timeout-sec 1800 \
  --temp-root-dir ./tmp/docker_visible_frames \
  --log-level INFO
```

### 3.4 一个非常重要的注意事项

如果 service 跑在 Docker 里，`--temp-root-dir` 必须满足：

- 它是宿主机上的真实目录
- 它已经通过 `--mount` 挂给了容器
- 挂载后容器里看到的是同一个绝对路径

例如：

- 宿主机临时帧目录：
  `./tmp/docker_visible_frames`
- Docker 启动时挂载：
  `$(pwd)/tmp/docker_visible_frames:$(pwd)/tmp/docker_visible_frames:ro`
- CLI 运行时：
  `--temp-root-dir ./tmp/docker_visible_frames`

这样脚本切出来的帧路径，对 service 来说也是可见的。

### 3.5 视频导出脚本参数说明

#### 输入输出

- `--input-dir`
  输入视频目录，必填。
- `--output-dir`
  输出目录。省略时默认是 `<input_dir>-camera-info`
- `--server-url`
  服务地址。默认：`http://127.0.0.1:18080`
- `--sample-fps`
  切帧采样率，必填。

#### 分块和并发

- `--initial-max-inference-frames`
  初始单次请求帧数。运行中可能自动放大或缩小。
- `--min-inference-frames`
  自动降载后的最小帧数。
- `--initial-max-concurrent`
  初始全局请求并发数。
- `--max-video-workers`
  同时处理多少个视频。
- `--max-inference-frames-cap`
  自适应分块的上限。
- `--max-concurrency-cap`
  自适应并发的上限。

#### 重试和超时

- `--max-retries`
  发生可重试错误时的最大重试次数。
- `--request-timeout-sec`
  单次 `/infer` 请求超时。
- `--base-backoff-sec`
  首次重试退避时间。
- `--max-backoff-sec`
  最大退避时间。

#### 延迟感知自适应策略

- `--success-window`
  连续多少次“健康成功请求”之后才尝试扩容。
- `--growth-max-latency-sec`
  成功请求如果超过这个总延迟，就不允许扩容。
- `--growth-max-latency-per-frame-sec`
  成功请求如果超过这个单帧延迟，就不允许扩容。
- `--degrade-latency-sec`
  成功请求如果超过这个总延迟，也会主动降载。
- `--degrade-latency-per-frame-sec`
  成功请求如果超过这个单帧延迟，也会主动降载。

#### 临时目录和调试

- `--temp-root-dir`
  切帧临时目录根路径。Docker 场景下推荐显式设置。
- `--keep-temp`
  不清理临时帧目录，便于调试。
- `--log-level`
  日志级别，例如：`INFO`、`WARNING`、`DEBUG`

### 3.6 结果文件里有什么

每个 `*-camera.json` 至少包含：

- `schema_version`
- `video_id`
- `source_video`
- `source_video_fps`
- `sample_fps`
- `num_sampled_frames`
- `coordinate_system`
- `frames`

每个 `frames[i]` 里包含：

- `sample_index`
- `source_frame_index`
- `timestamp_sec`
- `image_size`
- `intrinsics.fx`
- `intrinsics.fy`
- `intrinsics.cx`
- `intrinsics.cy`
- `pose.camera_pose`
- `pose.cam_quats`
- `pose.cam_trans`

### 3.7 一个最小可运行例子

如果你只想快速验证整条链路，可以先准备一个短视频目录，然后跑：

```bash
cd path/to/Video2Poses
uv sync

uv run scripts/run_video_pose_cli.py \
  --input-dir ./tmp/docker_uv_run_input \
  --output-dir ./tmp/docker_uv_run_output \
  --server-url http://127.0.0.1:18080 \
  --sample-fps 1.0 \
  --initial-max-inference-frames 8 \
  --min-inference-frames 2 \
  --initial-max-concurrent 1 \
  --max-video-workers 1 \
  --max-retries 3 \
  --request-timeout-sec 1800 \
  --temp-root-dir ./tmp/docker_visible_frames \
  --log-level INFO
```

---

## 4. 可视化 `*-camera.json`

### 4.1 启动脚本

可视化脚本是：

- `scripts/run_pose_visualizer.py`

它会做三件事：

- 读取一个 `*-camera.json`
- 用同样的 `sample_fps` 从原始视频重新采样右半边视频帧
- 在左半边逐帧渲染相机视锥，然后合成为一个 MP4

输出视频的布局是：

- 左边：相机位姿可视化
- 右边：原始视频的对应采样帧

### 4.2 推荐命令

```bash
cd path/to/Video2Poses
uv sync

uv run scripts/run_pose_visualizer.py \
  --camera-json path/to/xxx-camera.json \
  --view-mode 3d \
  --visualization-convention OpenCV \
  --history-window-sec 10 \
  --output-video path/to/xxx-3d-visualization.mp4 \
  --log-level INFO
```

如果你想看顶视图 2D 版本：

```bash
uv run scripts/run_pose_visualizer.py \
  --camera-json path/to/xxx-camera.json \
  --view-mode 2d \
  --history-window-sec 10 \
  --output-video path/to/xxx-2d-visualization.mp4
```

### 4.3 当前脚本完整用法

```bash
uv run scripts/run_pose_visualizer.py \
  --camera-json path/to/xxx-camera.json \
  [--output-video path/to/output.mp4] \
  [--view-mode 2d|3d] \
  [--visualization-convention OpenCV|OpenGL] \
  [--output-fps 10.0] \
  [--frustum-scale 1.0] \
  [--history-window-sec 10.0] \
  [--max-history-frustums 80] \
  [--temp-root-dir ./tmp/some_tmp_root] \
  [--keep-temp] \
  [--log-level DEBUG|INFO|WARNING|ERROR]
```

### 4.4 参数说明

- `--camera-json`
  输入的 `*-camera.json` 路径，必填。
  脚本会从这个 JSON 里读取：
  `source_video`、`sample_fps`、`coordinate_system`、每一帧的 `pose.camera_pose` 和 `intrinsics`
- `--output-video`
  输出 MP4 路径。
  省略时默认写成 `<camera-json-stem>-<view-mode>-visualization.mp4`
- `--view-mode`
  左侧可视化模式。
  `2d` 是右/前顶视图，只看 `X` 和前进方向。
  `3d` 是三维视锥视图，会显示右、前、上三个方向。
- `--visualization-convention`
  左侧视锥采用哪种坐标约定绘制。支持：
  `OpenCV` 或 `OpenGL`
  省略时默认使用 `camera.json` 里写入的坐标约定。
- `--output-fps`
  输出视频 FPS。
  省略时默认使用 `camera.json` 里的 `sample_fps`，这样左图和右图的时间步长和原始采样保持一致。
- `--frustum-scale`
  只调节“视锥画出来有多大”，不改位姿平移本身的米制单位。
  如果你觉得视锥太小不好看，可以把它调大，例如 `1.5` 或 `2.0`
- `--history-window-sec`
  只显示最近这段时间窗口内的轨迹和历史视锥。默认：`10`
  这个参数是用来避免轨迹一直累计到全局，导致后半段视锥缩成很小一团。
  如果你想更聚焦当前运动，可以改成 `5`；如果你想看更长上下文，可以改成 `15` 或 `20`
- `--max-history-frustums`
  在当前时间窗口内，最多同时绘制多少个历史视锥。
  时间窗口很长时，这个参数可以避免左图里同时堆太多蓝色历史视锥。
- `--temp-root-dir`
  临时目录根路径。
  脚本会把两类中间文件先写到这里，再合成为最终 MP4：
  1. 右半边重采样出来的原视频帧
  2. 左半边渲染出来的可视化帧
  省略时会使用系统临时目录。
- `--keep-temp`
  不删除中间帧，便于你检查：
  1. 左右两边有没有对错帧
  2. 某一帧视锥画得对不对
  3. 动态窗口缩放是否符合预期
- `--log-level`
  日志级别，例如：`INFO`、`DEBUG`
  `DEBUG` 适合调试，`INFO` 适合日常跑任务。

### 4.5 运行流程说明

脚本实际做的事情是：

1. 读取 `*-camera.json`
2. 从 `source_video` 按 `sample_fps` 重新采样右半边视频帧
3. 读取每一帧的 `pose.camera_pose` 和 `intrinsics`
4. 在左半边渲染 2D 或 3D 视锥
5. 只保留最近 `history-window-sec` 秒的轨迹和历史视锥参与显示范围计算
6. 用 `ffmpeg` 把左右两路帧横向拼接成一个 MP4

### 4.6 对齐和坐标约定说明

- 右半边不是随便读取原视频，而是按 `camera.json` 里的 `sample_fps` 重新采样，保证和 `sample_index` 一一对应
- 左半边使用 `pose.camera_pose` 逐帧绘制真实视锥，不使用伪造轨迹
- 左边不会再一直累计全局轨迹，默认只显示最近 `10s` 的轨迹和历史视锥，避免后半段缩得过小
- 坐标单位仍然是米，图上的坐标轴标签会明确写出 `(m)`
- `OpenCV` 模式下使用 `+X right, +Y down, +Z forward`
- `OpenGL` 模式下会把整套位姿变换到 `+X right, +Y up, +Z backward` 后再画

### 4.7 一个最小 smoke test

```bash
cd path/to/Video2Poses
uv sync

uv run scripts/run_pose_visualizer.py \
  --camera-json ./tmp/pose_visualizer_smoke/demo-short-camera.json \
  --view-mode 3d \
  --history-window-sec 10 \
  --output-video ./tmp/pose_visualizer_smoke/demo-3d.mp4 \
  --log-level INFO
```

---

## 5. 相关文档

- service API 和部署说明：`service/service.md`
- image client 使用说明：`video2poses/image_pose_cli.md`
- 当前代码设计说明：`video2poses/design.md`
