# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

Habitat HM3D 上的 ObjectNav 导航。目标链路（M1→M4）：

1. **M1 感知** —— 文本 → 实例 mask（GLEE 或 GroundingDINO+SAM），取 mask 中心像素
2. **M2 定位** —— 2D 像素 + 深度 → 反投影到 3D 场景坐标
3. **M3 规划** —— 尚未完成，等待用户进行规划
4. **M4 执行** —— 离散动作 `Forward 0.25 m` / `Left` `Right` 30°

**这不是 git 仓库**，没有版本控制，改文件前先自己备份。

## 代码风格约束
1. 注释统一用中文，函数注释参考如下
def function_name(param1, param2):
    """Function summary line. （简单描述该函数或方法的功能）

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    Raises: （提供了可能会碰到的报错，非必要可不写）
        ValueError: If `param1` is equal to `param2`.
    """
2. 不要在注释中描述问题、修改过程等无效内容。不要在函数文件前写大段的废话。注释只叙述功能或解释关键步骤。
3. 所有的 import 均写在文件开头，不要隐藏在函数内。

## 环境与启动

```bash
cd /home/xsuper/hc_workplace/HarnessNav
conda activate harnessnav          # Python 3.9, torch 2.5.1+cu121, habitat-sim/lab
```

`requirements.txt` 只列了在 `statenav` 克隆基础上**补装**的包；`detectron2` 与 `groundingdino` 都是源码编译的，不要试图 pip 重装：

```bash
CUDA_HOME=/usr/local/cuda-12.2 FORCE_CUDA=1 MAX_JOBS=4 pip install . --no-build-isolation --no-deps
```

## 常用命令

```bash
# M1 分割测试（图片放 test/seg_test/img/），两个后端都跑 + 出对比图
python test/seg_test/run_seg.py --text "chair"
python test/seg_test/run_seg.py --text "chair" --backend glee

# M0 oracle 基线：跑标注目标出指标（改动执行链路后必跑，见下"验收基线"）
python test/pix2move/run_pix2move.py --oracle --episodes 20 --seed 5 --no-save

# M0 交互式：文本目标 → 分割 → 反投影 → 吸附 → 自动跟随
python test/pix2move/run_pix2move.py
python test/pix2move/run_pix2move.py --seed 0 --glee-threshold 0.15

# play2nav：人用键盘玩，浏览器打开终端打印的网址（默认监听 0.0.0.0:8080）
# 轻触一下走一步，按住满 1 秒转为持续移动（--hold-delay 0 退回「一按下就持续动」）
python play2nav/app.py
python play2nav/app.py --host 127.0.0.1          # 只给本机 / SSH 转发访问

# play2nav 的网页层端到端自检（真起服务、真打 HTTP、真跑 env，约 2 分钟；不需要浏览器）
python play2nav/test_webserver.py
```

## 架构

### 两条并行的链路（互不替换）

| 目录 | 驱动来源 | 用途 |
|---|---|---|
| [test/pix2move/](test/pix2move/) | 文本 → 分割 → 反投影 → navmesh 贪心跟随 | **感知**研究，跑 baseline 拿 SPL |
| [play2nav/](play2nav/) | **人**按键盘 | 人工试玩、录回放 |

`play2nav/` 刻意**不 import torch、不碰分割**，只用 RGB 观测，启动快、内存小。两者都复用 [nav/](nav/) 和 [config/objectnav_hm3d_step_2000.yaml](config/objectnav_hm3d_step_2000.yaml)。

### nav/ 的定位

[nav/](nav/) 里的 `geometry.py` / `transform.py` / `habitat_config.py` 是从 **StateNav 复制**过来的，复制的原因就是**不让这里的改动污染 StateNav 那边**。改动时保持这个方向（只复制、不回写）。

规划本身**不在这里**——是 habitat 自己的 navmesh 代码：

| 环节 | API |
|---|---|
| 吸附到可行走面 | `sim.pathfinder.snap_point()` |
| 测地路径 / 距离 | `ShortestPath` + `pathfinder.find_path()`；`sim.geodesic_distance()` |
| 离散动作 | `sim.make_greedy_follower(goal_radius=…, stop_key=0).next_action_along(goal)` |
| 指标 | `env.get_metrics()` |

早先手写的占用栅格 + A* + pure-pursuit（`nav/local_plan.py`、`nav/path_planning.py`）**已删除**（备份 `/tmp/pix2move_bak/`）。别再往那个方向重建：HM3D 发布预烘焙的 `<scene>.basis.navmesh` 覆盖整个楼层，"幻影墙 / 饥饿栅格 / 近场盲区 / 退化吸附"那几类 bug 全是"从单帧深度自己造栅格"引入的，在新架构里**在设计上不存在**。

### M0 交互链路的数据流

[test/pix2move/run_pix2move.py](test/pix2move/run_pix2move.py)：RGB → 分割 → `mask_center_pixel` → `unproject_pixel`（用**相机**位姿）→ `snap_goal_to_navmesh` → `pursue` → 重观测。

`snap_goal_to_navmesh`（用户点名要求的**跨墙吸附守卫**）做三个检查，阈值都是实测定的：

- **`offset`** —— 吸附位移，> 2.5 m 拒绝
- **连通性** —— `find_path` 失败 ⇒ 跨岛，**不再原地拒绝**，改走下面的回退
- **绕行** —— `detour_reject`：直线 < 0.5 m 用绝对绕路量（≤ 3.0 m），否则用 `测地/直线` 比（≤ 4.0）

`euclid < 0.5 m` 与"比值"两个判据覆盖**不相交**的距离区间——用绝对绕路量当通用判据会误杀 p75 以上的合法目标（室内绕行几十米是常态，p99 = 36.8 m）。

**跨岛回退**（`farthest_reachable_along`）：HM3D 的 navmesh 是**分片**的，每个场景 1–3 个互不连通的岛，实测 **36.1%** 的标注 view point 与 agent 不同岛，而**所有** `find_path` 失败**全部**是跨岛、同岛零失败。所以"目标不可达"经常是**真的**不可达——habitat 自己的 `distance_to_goal` 对这类目标也返回 `inf`。做法是沿 `agent → goal` 连线从目标端往回扫（`t = 1 → 0`），第一个既在 navmesh 上、又能 `find_path` 到 agent 的点就是能走到的最远处。越过墙的采样点落在另一个岛上，`find_path` 直接失败 ⇒ **天然不穿墙**。采样循环里**同时**带绕行判据一路往回收——`find_path` 成功但绕 24 m 的点（实测 60 个里有 15 个）照走等于原地绕圈。

**绕行守卫实际是当楼层探测器在用**：被拒的点 `|Δy|` 清一色 = 2.80 m（正好一层楼），同层 3 m 内零误杀。所以拒绝理由要打印`高差 X m —— 目标多半在楼上/楼下`，比甩一个绕行比有用得多。

### play2nav

[app.py](play2nav/app.py) 的 `Navigator` 类同时服务**网页模式**与 `--headless --scripted` 两条路径。界面是**网页**（[web.py](play2nav/web.py) + [page.html](play2nav/page.html)），因为本机没有图形界面（`DISPLAY` 未设、无 Xvfb）——早先那版 Tk 窗口因此从未被真正运行验证过，已删除。HTTP 那层可以被自动验证（[test_webserver.py](play2nav/test_webserver.py) 把整条交互链路打一遍），本地窗口做不到。

**线程模型是这块最要紧的约束**（详见 README 同名小节）：sim 循环在**主线程**且独占 `env`，Werkzeug 在 daemon 线程且**永不接触 `env`**；两者之间只有一把锁保护按键状态与意图标志，任何锁都不许跨 `env.reset()`。

按键状态机在 [keystate.py](play2nav/keystate.py)（已单测）、步进与落盘在 `app.py`（已 headless 验证）。

界面上的七条行为约束，改前端时别弄丢：**轻触一下 = 走一步，按住满 `HOLD_DELAY_S = 1.0` 秒才转持续移动**（心跳重发**不得**重置计时、也不得补发那一步，否则永远只走一步，有单测守着）；**那一「步」是离散事件，不是按住状态的一部分**——欠着就一定发得出去（上限 `TAP_PENDING_S = 2.0`），**不要求服务端取键那一刻键盘还按着**：sim 线程走一步要 200 ms 上下，30 ms 的极短轻触完全可能整段落在这一步的执行期内，要求"取键时还按着"就会把那一口吃掉（实测「按一下没反应、再按一下才动」）；**心跳必须显式标出来**（前端 `hb: true`），而**不带 `hb` 的按下，哪怕距上一次松手只有几毫秒，也必须是新的一次按下**（重新计时、重新拿到一步）——两者在时序上无法区分，把宽限期内的按下当心跳吞掉就是"连点两下只走一步"；**两个画面都是常开的 MJPEG 流**（`/stream` 与 `/topdown_stream`），不许退回「逐帧换 `<img src>`」的轮询——那会在浏览器每域名 6 条连接的限制下把按键 POST 排到图片请求后面，实测表现为按一下要等 1~2 秒；**每一帧必须作为两段 part 发**（`_mjpeg` 里那一帧 `yield` 两次）——浏览器要等到更后面一段 part 到达才把当前帧提交上屏，而这条流在 agent 站着不动时是静默的，只发一段就会让画面永远停在上一步的 obs 上，实测表现为「按一下键画面不动、再按一下才显示上一个动作的结果」（Chrome 145；把 boundary 紧跟帧后发、连下一段首部发、去掉 `Content-Length` 都没用，只有同一帧发两遍管用）；**「碰到障碍物」只看 `env.step` 前后位置是否相同**（`STUCK_EPS_M = 1e-4`，被挡住时位移是**真的** 0），不拿 habitat 的 `collisions.is_collision` 当闸门（那是**接触**标志，正常直行擦到家具也为真，实测每步走满 0.29 m 时照样置位）；**第一视角画面在昂贵的俯视图上色与 mp4 编码之前推出去**，且两条画面路径**都不许把更新吞掉**（俯视图限速只推迟不吞，第一视角每 tick 有兜底补推；`/state` 里的 `pub_rgb_seq==emit_seq`、`pub_topdown_seq==frame_seq` 就是"画面追平了没有"的判据）。

产物默认写 `/DATA_HDD/hc/play2nav_outputs/`。**不要改到默认路径**：`/` 分区只剩十几 GB，录像累积会撑爆。`--stage` 在本机**只有 `val` 可用**（train/val_mini 的场景摆放路径对不上，是数据问题不是配置问题），`play2nav/config.py::check_split` 会在构造 `habitat.Env` **之前**预检并说清楚。

## 坐标与约定（易错，改代码前必读）

这些坑每一条都**不报错**，只是静默画错 / 走错：

- **`maps.to_grid(a, b, …)` 的第一个参数是 z，不是 x**（形参却叫 `realworld_x`）。habitat 自己传的是 `to_grid(pos[2], pos[0], …)`，返回值直接索引 `map[a, b]` ⇒ 最终 **row↔z、col↔x**。写反会让所有标记**静默画到画外**。`grid_size` 要按 `map.shape` 逐轴算——`calculate_meters_per_pixel` 返回的是两轴 **min**，非正方形场景下不等于两轴的换算系数。
- **机身位姿 ≠ 相机位姿。** YAML 里传感器挂在 `position: [0, 0.88, 0]`、机身本身又在高度 0.88 ⇒ 相机在 **1.76 m**。`sim.set_agent_state` 要的是**机身**位姿，喂相机位姿会把 agent 抬高 0.88 m 且**不可逆**；而反投影**必须**用相机位姿。两者分开（`body_pose()` / `sensor_pose()`），别混用。
- **`env.reset()` 会让之前拿到的一切句柄失效**，而且有两种失效方式：`make_greedy_follower` 的返回值会抛 `InvalidAttachedObject`；**`env.sim.pathfinder` 更阴——不报错，照常返回结果，但那些结果来自上一个场景的 navmesh**。表现是 `snap_point` 出界、`is_navigable` 返 False、`find_path` 说不可达，于是本来有解的 episode 被判成无解。多 episode 必须每集重取（`Navigator.start_episode()` 统一收口，`verify_episode()` 守住院线）。
- **`habitat.Env(config)` 只构造、不自动 reset**，不 reset 就 `env.step` 直接 `AssertionError`。
- **深度用 `min_depth` 编码"这条射线没打到东西"**，不是 0 也不是 `max_depth`。HM3D 网格有洞，实测一帧 **30%** 的像素读数**恰好** `min_depth`；反投影这些点会在正前方生成一堵假墙。`preprocess_depth` 默认的 `lower_bound=0.1` 拦不住，必须按传感器实际的 `min_depth`/`max_depth` 设边界（见 `clean_depth()`）。排查手法：打印 `depth == min_depth` 的像素数**及其行分布**——散布全图就是哨兵值，集中在某块才是真近物。
- **深度传感器给的是 `(H, W, 1)` 不是 `(H, W)`**，`float(depth[y, x])` 会发 `DeprecationWarning`、将来直接报错。统一在 `clean_depth` 里压维。
- **`quaternion` 库的约定是 `[w,x,y,z]`**，habitat `AgentState.rotation` 是 `[x,y,z,w]`。传错静默得到错误旋转矩阵 ⇒ `nav/transform.py::as_quaternion`。
- **`mapper_local_to_world(local_pos, initial_position)` 第二个参数要传 `habitat_translation(origin_world)`**，不是 `origin_world`（它只对偏移量做 `(x,z,y)→(x,y,z)` 重排）。传错**不报错**，标记静默消失。凡是"坐标算出来但画不出来/对不上"，先做一次 `habitat_translation()` 往返比对。
- **argparse 的默认值会盖掉函数签名里的默认值**，两处必须一起改（守卫阈值曾因此 CLI 用上了更紧的旧值，正常目标被大面积误杀）。

## 行为约定（不是 bug，别"修"）

- **`env.step(HAB_STOP)` 会终止 episode**，而 `Success` 依赖 `task.is_stop_called`。所以同一个 `pursue` 在两种模式下需求**正好相反**：oracle 评测**必须**发 stop（否则 success/SPL 恒为 0），交互模式**必须**不发（否则到一次目标整个会话就没了）。用 `stop_at_goal` 显式分开。
- **`make_greedy_follower` 的 `stop_key` 必须显式传**：默认 `None` 时到达只返回 `None`、**不会真的执行 stop**。
- **`--success-distance` 默认 1.0 m**（HM3D ObjectNav 标准），仓库 YAML 里写的是 0.25 m。用 0.25 几乎不可能 stop 在圈内，success/SPL 会恒为 0。
- **有 open-vocabulary 分割在后头时，交互指令词必须当保留字提前拦**（`MANUAL_ACTIONS`）。`left`/`right` 送进 GLEE 不会被拒绝，而是匹配出一个低分实例然后一本正经走过去（阈值 0.15 时 `left` 匹到 0.186）。这类 bug 不报错、只是行为诡异。凡是"输入了 X 但机器人做的不是 X"，先查有没有被下游开放词表吃掉。
- **`p`（play2nav）与 `stop`（pix2move）只结束本集**，`habitat.Env._check_episode_is_active` 只返回 `not is_stop_called`，不按就**不会**自己结束。
- **「走到目标旁边」靠吸附到最近自由格实现，不要靠把终点沿路径回缩**——再回缩一格会让到达判据把 0.6 m 外当成已到达，表现为"走到附近就站着不动"。
- **出生点可能顶着家具**（实测连发 6 次 `move_forward` 全部 `collided`，总共挪 0.0834 m，后 5 步位移精确为 0.0000）。不是 bug，**遇到走不动就先转身**（转身实测永不碰撞）；网页据此提示「碰到障碍物，无法移动」。`allow_sliding` 在 habitat-lab 0.3.1 的 Hydra 配置里没暴露，**保持默认（不滑行）**。

## 文档

- **[progress.md](progress.md)** —— 完整的改动史、实测数据、踩过的坑（58 条 + 逐条排查手法）。**动 `test/pix2move/`、`nav/` 或 `play2nav/` 之前先读它**，绝大多数坑上面都记了，重踩一遍的成本很高。
- **[play2nav/README.md](play2nav/README.md)** —— 网址与键位、产物格式、`metrics.json` 字段、参数含义、**线程模型**、自检各项输出的含义、以及**「我没能验证的部分」（浏览器本身）的验收清单**。

改完行为要同步更新 progress.md（`play2nav/` 已在 2026-09-14 那节补进来）。

## 网络与镜像

**直连不通**，不要浪费时间重试：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # huggingface.co 被墙
source /home/xsuper/.hf_env                  # HF_TOKEN，chmod 600，仓库外
```

- GitHub + `dl.fbaipublicfiles.com` 均超时 → 加前缀 `https://ghproxy.net/https://github.com/<owner>/<repo>`
- PyPI / conda 已配好清华镜像（`~/.pip/pip.conf`、`~/.condarc`）

**大文件下载必须断点续传 + 重试 + 显式校验大小**——镜像会中途掐断，**且 wrapper 脚本仍以 0 退出**（GLEE 曾停在 291 MB/1.44 GB、SAM 停在 130 MB/2.56 GB，都"成功"了）：

```bash
curl -sSL -C - --retry 3 --retry-delay 5 --connect-timeout 30 \
     --speed-time 60 --speed-limit 10240 -o "$out" "$url"
# 外层 for i in 1..8 循环，每轮用 stat -c%s 对比预期大小
```

**`nvidia-smi` / EGL 报 driver-library version mismatch**（内核模块 vs 用户态版本不一致）会让 habitat-sim 起不来，报 `unable to find CUDA device 0 among N EGL devices`，而此时 `torch.cuda.is_available()` 仍是 `True`，极易误判成代码问题。**重启机器**即可。

`/home` 磁盘曾接近满（97–98%），大文件放 `/DATA_HDD`（1.7 TB 可用）。
