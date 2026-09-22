# HarnessNav 可视化测试

在仓库根目录、`harnessnav` 环境里跑。默认 `--stage val`、`--seed 5`。图和 `meta.json` 写在各脚本打印的输出目录。

```bash
cd /home/xsuper/hc_workplace/HarnessNav
conda activate harnessnav
```

长 episode 录像仍应写 `/DATA_HDD/hc/`，不要改默认大目录。下列脚本默认只在 `test/` 下写几张 PNG。

## 建图 / 前沿

三值栅格 BEV（FREE 灰、OCC 白、UNKNOWN 暗；含低头扫地板）：

```bash
python test/occupancy/grid_test.py --seed 5
```

输出：`test/occupancy/output/`（`grid_bev.png`、`meta.json`）

出生点环视两圈（全景 + 彩色/黑白 BEV）：

```bash
python test/occupancy/run_occupancy.py --seed 5 --pano-even
```

输出：`test/occupancy/out/`

前沿画在 BEV 与第一视角（会转到画面里真有前沿的偶序号扇区）：

```bash
python test/frontier/frontier_test.py --seed 5
```

输出：`test/frontier/output/`（`bev.png`、`ego.png`、`meta.json`）

## Skills（输出已分开）

Look：平视 / 低头 / 抬头并排：

```bash
python test/skills/look_test.py --seed 5
```

输出：`test/skills/look/output/`（`look_level.png`、`look_down.png`、`look_up.png`、`look_strip.png`）

Depth：环视后用 `door` / `bed` / `window` 找实例；mask IoU≥0.5 或小框被大框盖住 ≥50% 视为重复，同类最多 3 个：

```bash
python test/skills/depth_test.py --seed 5
python test/skills/depth_test.py --seed 5 --text bed
```

输出：`test/skills/depth/output/`（`depth_anno.png`、`meta.json`）

Recall：两处 ScanNode 后取回节点 0 的图：

```bash
python test/skills/recall_test.py --seed 5
```

输出：`test/skills/recall/output/`（`recall_bev.png`、`recall_side_by_side.png`、当前/召回 FOV、各次 Scan 的 BEV）

不要看旧的 `test/skills/output/`，那是拆目录之前的产物。

Verify：低头扫过的 BEV + 多视点核实（默认 `door`）：

```bash
python test/skills/verify_test.py --seed 5
python test/skills/verify_test.py --seed 5 --text bed
```

输出：`test/skills/verify/output/`（`verify_bev.png`、`verify_view*.png`、`meta.json`）

## Memory

节点图（2～3 个 ScanNode，BEV 含地板点云）：

```bash
python test/memory/nodes_test.py --seed 5 --scans 3
```

输出：`test/memory/nodes/output/`（`nodes_bev.png`、`history.json`、各 `scan*_n*_bev.png`）

不要看旧的 `test/memory/output/`。

TraceBack 回到节点 0：

```bash
python test/memory/traceback_test.py --seed 5
```

输出：`test/memory/traceback/output/`（`traceback_bev.png`、`meta.json`）

## Mover

默认 semantic，查询 `door`/`bed`/`window`，NMS 后最多 3 个：

```bash
python test/mover/overlay_test.py --seed 5
python test/mover/overlay_test.py --seed 5 --mode frontier
python test/mover/overlay_test.py --seed 5 --mode semantic --text window
```

输出：`test/mover/output/`（`ego_annotated.png`、`meta.json`）

## 端到端 / 回归

脚本 Planner/Mover 跑若干 ScanNode：

```bash
python test/harness/run_episode.py --seed 5 --max-scans 3
python test/harness/run_episode.py --seed 5 --max-scans 2 --no-glee
```

输出：`test/harness/output/`

pix2move oracle（抽出 `nav/goto.py` 后必跑）：

```bash
python test/pix2move/run_pix2move.py --oracle --episodes 5 --seed 5 --no-save
```

分割样例图（不进仿真器）：

```bash
python test/seg_test/run_seg.py --text "door"
python test/seg_test/run_seg.py --text "chair" --backend glee
```

输出：`test/seg_test/out/`

play2nav 网页层自检（约 2 分钟，不起浏览器）：

```bash
python play2nav/test_webserver.py
```

## Qwen-VL（qwen3 环境）

权重是 `/DATA_HDD/hc/llm/qwen/Qwen3.6-27B`（带视觉）。相对你原来的 serve 命令：`--limit-mm-per-prompt` 提到 8 张（6 方向 + BEV），并加上官方 `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`。

```bash
conda activate qwen3
cd /DATA_HDD/hc/llm/qwen
CUDA_VISIBLE_DEVICES=2,3 vllm serve /DATA_HDD/hc/llm/qwen/Qwen3.6-27B --port 8711 \
  --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --max-num-seqs 16 --dtype bfloat16 --limit-mm-per-prompt '{"image": 8}' \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --served-model-name Qwen-VL
```

服务起来后（评测用 harnessnav，因为要起仿真器）：

```bash
cd /home/xsuper/hc_workplace/HarnessNav
conda activate harnessnav
python run_HarnessNav.py --episodes 10 --seed 5 \
  --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 20
python run_HarnessNav.py --no-vlm --episodes 10 --seed 5 --max-scans 20
```

上下文压缩与 Stop 闸门（无仿真）：

```bash
python test/harness/test_ctx_compress.py
python test/harness/test_plan_query.py
```

离线图理解（qwen3）：

```bash
conda activate qwen3
python test/vlm/VLM_test.py --wait-s 30
```

产物：`/DATA_HDD/hc/harness_outputs/p1/<run_id>/`（每集 `debug.txt`、`metrics.json`（含 `time_cost`、`tool_counts`）、`topdown.mp4`、scan 全景/BEV、mover 标注图、`planner_chat.json`；整次运行另有 `summary.json` 与 `brief_summary.txt`，后者含 sr / dtg / spl 均值、总耗时与每集平均用时）。验收 10 集 mean success ≥ 0.60。
VLM 单测图：`test/vlm/output/report.json`。

## 看图时注意

- ScanNode / memory 的彩色 BEV 应能看到 agent 脚下附近的地面（地毯等），不只是远处墙脚。高度切片相对机身，不是相对相机。
- 扇区数字在扇区中心（`0` 为正前方），分割线在 ±30° 边界，不要和数字叠在一起。
- GLEE 同类实例：mask IoU≥0.5 或较小 mask 被覆盖 ≥50% 则去重，最多留 3 个。
- `look_up.png` 应比 `look_level.png` 明显抬头，不能是同一张平视图。
