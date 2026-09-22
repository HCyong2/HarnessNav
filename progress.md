# HarnessNav 进展记录

最后更新：2026-09-14

## 项目目标

Habitat HM3D 上的导航任务，底层执行链路：

1. **M1 感知**：按语义取感兴趣像素（文本 → 实例 mask），取 mask 中心点
2. **M2 定位**：2D 像素 + 深度 → 反投影到 3D 场景坐标
3. **M3 规划**：3D 目标 → 2D 成本图 → A*
4. **M4 执行**：离散指令（Forward 0.25m / Left / Right 30°）逼近目标

已确定的选型：文本目标用**文本类别**（如"去椅子"）；目标**单帧可见**；路径规划用 **2D 成本图 A***。

---

## 已完成

### M1 感知后端（已完成并验证）

两套"文本 → mask"方案都已跑通，权重全部下载到 `HarnessNav/model/`：

| 方案 | 说明 | 权重 |
|---|---|---|
| **GLEE**（默认） | 单模型，文本直接出 box + mask | `GLEE_SwinL_Scaleup10m.pth` (1.4G) + `clip-vit-base-patch32/` |
| **GroundingDINO + SAM v1**（保留做消融） | DINO 出 box，SAM 出 mask | `groundingdino_swint_ogc.pth` + `sam_vit_h_4b8939.pth` + `bert-base-uncased/` |

代码：

- `perception/base.py` — `SegResult` / `SegBackend` / `mask_center_pixel`（统一接口，图像约定 **RGB uint8 (H,W,3)**、box **xyxy 像素**、mask **bool (H,W)**）
- `perception/glee_backend.py`、`perception/gdino_sam_backend.py`、`perception/render.py`
- `test/seg_test/run_seg.py` — 分割测试 CLI

**结论：默认用 GLEE**（效果更好）。GroundingDINO+SAM 接口保留，方便做推理成本/效果的消融对比。

对比测试观察（供参考）：`toyroom.png` 上两方案对同一把椅子一致（GLEE 中心 (297,382) vs GS (298,381)）；`bedroom.png` 上 GLEE 在阈值 0.2 时 0 实例、GS 出 1 个 (0.303)；`toyroom.png` 上 GLEE 多出一个墙/壁画上的误检 (0.213)。

### M0 端到端验证（已完成并验证）

> **2026-09-12 已重写**：A* 自建栅格那条链路已被删除，改用 habitat 原生 navmesh。下面这段保留的是**当时的**实现记录，现已不适用，看「2026-09-12 重写」一节。保留它是为了说明那些 bug 从何而来。

`test/pix2move/` —— 交互式：输入语义目标 → 分割 → 2D 像素 → 反投影 3D → **navmesh 吸附 + 贪心跟随** → 重观测，输入 `stop` 结束。（`--oracle` 则直接跑标注目标出指标。）

**工具函数已复制进 HarnessNav（不 import StateNav，避免后续改动污染 statenav 环境）：**

| HarnessNav 文件 | 复制自 StateNav | 内容 |
|---|---|---|
| `nav/geometry.py` | `mapping_utils/geometry.py` + `preprocess.py` | `get_pointcloud_from_depth(_mask)`、`translate_to_world`、`preprocess_depth` |
| `nav/transform.py` | `mapping_utils/transform.py` | `habitat_camera_intrinsic`、`habitat_translation/rotation`、`mapper_local_to_world`、新增 `as_quaternion` |
| `nav/habitat_config.py` | `config_utils.py` | `hm3d_config`（YAML 指向 HarnessNav 自己的副本） |
| `config/objectnav_hm3d_step_2000.yaml` | `config/` | 环境配置副本 |

~~`nav/path_planning.py`~~、~~`nav/local_plan.py`~~ —— **已删除**（2026-09-12），备份在 `/tmp/pix2move_bak/`。

**输出文件**（`test/pix2move/obs/`，`stem = 场景名_epid`，`N` = 累计动作数从 0 起）：

| 文件 | 内容 |
|---|---|
| `{stem}_{N}.png` | 原始 RGB 观测 |
| `{stem}_{N}_seg.png` | 分割叠加图（mask + box + 中心像素） |
| `{stem}_{N}_map.png` | **habitat 原生 topdown 图** + 历史轨迹 + 目标点 |

`_map.png` 用 `env.get_metrics()["top_down_map"]`（即 `statenav_agent.py` 里那个 measurement，由 `nav/habitat_config.py` 配置），`colorize_topdown_map` 上色 + `draw_agent` 画 agent 图标，再叠加：

| 标记 | 含义 |
|---|---|
| **洋红折线** | agent 走过的历史轨迹 |
| **青色折线** | navmesh 测地路径（`ShortestPath.points`） |
| **黄色小实心点** | 反投影的原始目标（落在物体上） |
| **红色空心圆环** | 吸附到 navmesh 后的实际导航目标（GOAL），最后画、半径最大，压在 agent 图标上也看得见 |

标记半径按图幅算（`min(H,W)//40`，约 25px），不是固定像素——habitat 的 topdown 图约一千像素见方，早先 6px 的点和 8px 的环在这个尺度下等于没画，而且 8px 的环完全被 32px 的 agent 图标盖住，这正是"图上看不到目标点"的原因。坐标系为 habitat 世界 `(x, z)`；注意 `maps.to_grid(pos[2], pos[0], ...)` 的第一个参数是 **z**（虽然形参名叫 `realworld_x`），即 **row 对应 z、col 对应 x**，写反会让所有标记画错位置。

### 2026-09-11 修复：「目标点总被阻碍 / 只能走一步 / 图上没有目标点 / 帧号不递增」

用户给了 `BAbdmeyTvMZ_19` 的完整终端日志，四个现象两个根因：

1. **找到的目标点总被阻碍 + 只能移动一步** —— 两个叠加的原因：

   **(a) 深度传感器把"没打到东西"编码成 `min_depth`。** habitat 对射空的射线返回**恰好** `min_depth = 0.5`；HM3D 网格有洞，所以这非常普遍——实测某一帧 **307200 个像素里有 92742 个（30%）读数正好 0.5**。这些不是测量值，但原来的 `preprocess_depth(lower_bound=0.1)` 放行了它们，反投影后变成 agent 正前方约 0.5m 处的**假墙**（局部高度 +0.18 ~ −0.19m，全部高于 `floor_h = −0.5`，于是被判成障碍）。
   → 新增 `clean_depth()`，把 `min_depth` 和 `max_depth` 两个哨兵值一并清零；清理后的深度图**统一用于**分割中心点选取、目标反投影和点云建图。`mask_center_pixel` 本来就以 `depth > 0` 为有效判据，配合得很好。

   **(b) 相机正下方那片地面在物理上永远看不到。** 相机高 0.88m、`fy/cy = 1.621`、俯角 31.7°，最低一条成像光线要落到 `0.88 × 1.621 ≈ 1.43m` 才着地，**比这更近的地面落在画面下沿之外**。把"没看到"当障碍，那么 agent 自己脚下就是障碍，A\* 连起点都不可通行。原来靠 `force_free_xy` 硬挖 3×3 兜底——**而那个补丁就是全图唯一稳定自由的区域**，于是 `snap_goal_to_free` "最近可达自由格"返回的永远是 agent 自己脚下（日志里 `cell=[7, 3] dist=0.23m` 正是如此），A\* 给出一条单点路径，控制器宣布"已到达"，`actions: []`。
   → 新增 `blind_spot_cells()`：把"半径 1.43m、相机方位角 ±hfov/2 以内"的格子当作**地面**。关键是**填充发生在 raised 覆盖之前**，所以该区域内真正看到的障碍照样是障碍；方位角限制则保证不会穿过身后的墙去规划。`force_free_xy` 随之缩小为"只挖 agent 所在那一格"，仅用于保证 A\* 一定有可行起点。

   附带的连锁修复：栅格范围原来只由点云决定，去掉哨兵点后**最近的可见地面在 agent 前方 1.43m**，栅格边界的 `pad_cells`（0.5m）盖不住 agent 自己——起点会掉到图外。现在栅格范围取"点云 ∪ blind spot 圆盘 ∪ force_free 点"的包围盒。

2. **地图上没有目标点** —— 两个独立原因，第二个是真 bug：

   **(a) 标记尺寸写死成像素了。** habitat 的 topdown 图约一千像素见方，6px 的黄点等于没画，8px 的蓝环则完全被 32px 的 agent 图标盖住（而 `dist=0.23m` 时目标恰好就在 agent 脚下）。
   → 目标改为**红色空心圆环**，半径 `min(H,W)//40`（约 25px）、线宽 3，**最后绘制**因而压得住 agent 图标；另加一条淡红连线从 agent 当前位置指向该环，即使两者重合也能区分"有目标"和"没目标"。

   **(b) `mapper_local_to_world` 的 `initial_position` 要的是 mapper 局部序 `(x, z, y)`，不是 habitat 世界序 `(x, y, z)`。** 这个函数只把**偏移量**从 `(x, z, y)` 排成 `(x, y, z)`，原点本身是按局部序直接相加的。驱动代码传了 `origin_world`（世界序），于是原点的 y/z 被换了一次——实测起点世界 z=−1.70 时返回 z=+2.41（即起点的**高度**）。算出来的点行号 1160 掉在一张 1049 行的图外，`to_px` 直接返回 `None`，**标记被静默跳过，一个都不画**（连图例都只剩 "agent path" 一项）。这与坑 9/14 是同一类坐标序错误，而且因为失败方式是不画而不是报错，非常难发现。
   → 驱动改传 `origin_local`；`mapper_local_to_world` 补上明确的入参顺序说明和实测数值；另加一条"目标世界 z 与 agent 世界 z 相差超过 30m 就告警"的兜底检查。回归测试里加了局部/世界序的往返比对。

3. **obs 图片帧号不递增** —— 主循环每轮开头无条件落盘一次，而"输入未匹配到实例""目标不可达"这类分支不消耗动作数却会 `continue`，于是同一个 `_N` 被反复重写（日志里 `_2.png` 连写三次）。
   → 记录 `saved_index`，只在 `move_count` 变化时落盘；同一帧重复尝试只更新终端提示。实测同一次运行里同一帧号打印 7 次而磁盘上只有 1 个文件。

另外把 `target sits on an obstacle -- snapping...` 的措辞改掉了——那是**预期行为**（反投影点必然落在物体表面），原来的措辞容易被当成错误；另外加了两条诊断：执行前后打印实际位移，以及"发出前进指令却没动"的警告（说明被 `allow_sliding: False` 卡住）。

**验证结果（2026-09-11 修复后重新验证，全部通过）：**
- 合成场景单测：栅格朝向（+x→+列、+z→+行）、障碍判定、吸附不出物体、A* 全程走自由格
- 控制器闭环回归测试（`/tmp/t_nonsim.py` 系列）：直走廊 4 个朝向、L 形走廊 4 个朝向均收敛且全程不进入障碍格。**L 形走廊从修复前的 0.52m 停滞变为收敛到 0.03m**
- topdown 像素映射与 habitat 自己的 `maps.to_grid` 逐点比对，41×41 采样 **0 误差**
- 真机：自动标定得 **turn_sign = −1**（`turn_left` 让朝向角变化 −30°）、**forward_sign = +1**（`move_forward` 位移实测 0.25m）。以 `wall` 为目标跑通全链路：中心像素 (58,218) 深度 1.33m → 局部 (−1.41,−0.76) → 9×11 栅格 → 吸附到 (−0.29,−0.19) → A* 2 格 → 执行 `[left, forward]` → 存新帧重规划

**2026-09-11 四项反馈修复后的回归（`/tmp/t_fix.py`，35 项断言全过）：** 盲区填充使近处地面可走、盲区填充不会击穿已观测到的障碍（0.8m 处的墙仍是障碍）、越出相机方位角的地面不会被当成可走、退化吸附被拒、正常吸附不受影响、`clean_depth` 两个哨兵值都清零、`mapper_local_to_world` 的入参坐标序往返一致。

**真机复测（`--seed 3`，场景 `h1zeeAwLh9Z` ep27，提示词 `door`）：**

```
depth      : 0.5-5.0 m; 0.5 和 5.0 are no-hit sentinels
blind spot : floor closer than 1.43 m is below the image ... -- assumed walkable
  goal (local)  : x=+1.30 z=-0.00 dist=1.30 m  cell=(8, 12)
  grid          : 19x28 cells free=40 (depth valid 100%, blind-spot fill r=1.43 m)
  goal (snapped): cell=[8, 11] x=+0.95 z=-0.11 dist=0.96 m  path_len=5 cells
  actions       : ['forward', 'forward', 'left', 'forward']
  executed 4 action(s); distance to goal 1.30 -> 0.57 m (travelled 0.73 m)
```

对照修复前同场景的 `9x17 cells free=10` + `snapped dist=0.23 m` + `actions: []`：自由格 10→40，吸附点从 agent 脚下 0.23m 变成前方 0.96m，动作从空列表变成 4 个，**到目标距离 1.30→0.57m 实测下降**。`_map.png` 上红圈、黄点、洋红轨迹、三项图例齐全。同一次运行的终端里 `--- frame ..._4.png ---` 打印了 3 次而磁盘上只有一个该文件。

**已知限制：** 单帧成本图只有 1–3m 见方，完全取决于当前帧看到多少地面。默认 episode（GLAQ4DNUx5U ep=25）出生点几乎贴着柜子，画面里只有柜体和左边一扇门，**连 `door` 都匹配不到**，建议直接用 `--seed 3` 或 `--seed 1`（实测这两个 seed 首帧有可分割的目标：seed 3 有 door 112727px，seed 1 有 wall 54603px）。出生点常正对墙时，提示符输入 `left`/`right`/`forward` 可手动转视角。

### 2026-09-11 修复：「分割出目标后走不动」

用户反馈成功分割出目标却无法向其移动。三个真实 bug，都在 `test/pix2move/run_pix2move.py` 与 `nav/local_plan.py`：

1. **`calibrate()` 把相机位姿写回了机身位姿** —— `set_agent_state` 收的是 **body** 位置，而 YAML 里传感器挂在 `position: [0, 0.88, 0]`、机身本身又在高度 0.88，所以相机在 1.76m。把 `sensor_pose` 的结果喂回 `set_agent_state`，等于把 agent 抬高 0.88m 并**再也放不下来**。后果是 `floor_h = -0.5` 的判据失效（相机升到 1.76m 后，1.26m 以下的东西全被判成"地面"），家具不再算障碍，agent 径直撞上目标物体，而 `allow_sliding: False` 让它卡死不动。
   → 新增 `body_pose()` 专管机身位姿，标定结束用 `body_pose` 恢复；`sensor_pose()` 只用于反投影。

2. **控制器把终点沿路径往回缩了一整格** —— 原意是"停在物体旁边不要贴上去"，但吸附到自由格已经把这件事做完了，再回缩一格 + 0.35m 到达容差，等于距目标 **0.6m 就宣布"到了"**然后站着不动。
   → 删掉回缩，`goal_tol` 默认改为 `grid_res`（0.25m）。实测 L 形走廊从停滞在 0.52m 变为收敛到 0.03m。

3. **topdown 叠加图的 row/col 反了** —— habitat 的 `maps.to_grid(pos[2], pos[0], ...)` 第一个参数是 **z**（形参却叫 `realworld_x`），所以 **row↔z、col↔x**，和我原来写的正好相反。
   → 修正 `world_xz_to_pixel`，并改用 pathfinder 的每轴跨度（而不是 `calculate_meters_per_pixel` 返回的单一标量——那是两轴取 min，只有正方形场景才同时正确）来换算。与 habitat 自己的 `to_grid` 逐点比对 0 误差。

另外顺带调整：`align_deg` 限制前进瞬间的朝向误差（0.25m/步下 30° 误差只剩 0.21m，进不了 0.25m 到达圈）；直行落点被挡时先尝试换下一个航点，但目标已在 0.25m 内时绝不再转向（这正是原来"贴着物体干转"的场景）；`render_grid_map` 已删除。

### 2026-09-12 注释中文化：`nav/local_plan.py` + `test/pix2move/run_pix2move.py`

按用户要求把两个文件的**注释与 docstring** 换成中文，**代码零改动**。

- 保留原样：所有标识符、字符串字面量（含 `print()` 输出、argparse 的 `help=`、图上图例文字 `agent path` / `GOAL: heading here`）、常量、数值。因此运行行为、CLI 帮助、`--help` 的选项说明、图上文字与之前**逐字节一致**（模块 docstring 会被 `argparse` 当 `description` 用，故那一处 help 抬头变成了中文）。
- 保留原文中的实测数值与术语原文：`92742 of 307200 pixels`、`1.43 m`、`0.88 m`、世界 z 从 −1.70 算成 +2.41 的例子、`FREE=2` / `floor_h` / `mapper_local_to_world` 等反引号标识符一律不译。

验收（`harnessnav` 解释器）：

```bash
python /tmp/i18n_check.py /tmp/i18n_bak/local_plan.py nav/local_plan.py       # IDENTICAL CODE
python /tmp/i18n_check.py /tmp/i18n_bak/run_pix2move.py test/pix2move/run_pix2move.py
python -m py_compile nav/local_plan.py test/pix2move/run_pix2move.py
python /tmp/t_fix.py                                                          # 35 passed, 0 failed
```

`/tmp/i18n_check.py` 的做法：解析两份文件，剥掉模块/函数/类的 docstring（注释本来就不进 AST），再比 `ast.dump()`；相同则打印 `IDENTICAL CODE`，否则打印逐行 diff 并以退出码 1 结束。**注意**：`visit_Module`/`visit_FunctionDef` 这类 `NodeTransformer` 分发必须调用 `generic_visit`，否则只有模块层级的 docstring 会被访问到，函数 docstring 全部残留，检查会假阳性。

---

### 2026-09-12 重写：改用 Habitat 原生 navmesh，删掉自建栅格

用户要求：「最终需求就是跑 HM3D 的 baseline、拿到 SPL 等指标，尽可能用 Habitat 原生代码实现，简化整个 pix2move 测试，旧代码直接删掉，注意跨墙吸附问题。」

**删掉的**（备份在 `/tmp/pix2move_bak/`）：`nav/local_plan.py`（444 行占用栅格 + 盲区填充 + 吸附）、`nav/path_planning.py`（A*）。

**为什么整个架构是多余的**：HM3D 发布预烘焙的 `<scene>.basis.navmesh`，覆盖整个楼层平面，`sim.pathfinder.is_loaded` 为 True。原先四个 bug 类别（幻影墙、饥饿栅格、近场盲区、退化吸附）全部源于"从单帧深度自己造栅格"，而 navmesh 从一开始就知道整个楼层——这些 bug 在新架构里**在设计上不存在**，不是被修好的。

**新链路**（全部是 habitat 自己的代码）：

| 环节 | 用的 API |
|---|---|
| 吸附到可行走面 | `sim.pathfinder.snap_point()` |
| 测地路径 / 距离 | `ShortestPath` + `pathfinder.find_path()`；`sim.geodesic_distance()` |
| 离散动作 | `sim.make_greedy_follower(goal_radius=..., stop_key=0).next_action_along(goal)` |
| 指标 | `env.get_metrics()` —— `distance_to_goal` / `success` / `spl` / `soft_spl` / `collisions` |

`stop_key` 必须显式传：默认 `None` 时到达只返回 `None`、**不会真的执行 stop**，而 `Success` 判据要求 `task.is_stop_called`，不调 stop 则 success/SPL 恒为 0。

**CLI**：`--oracle`（跑标注 view point 出指标）/ 默认交互模式；`--episodes`、`--seed`、`--success-distance`（默认 1.0，HM3D ObjectNav 标准；本仓库 YAML 写的是 0.25）、`--max-offset`、`--max-detour-m`、`--max-detour-ratio`、`--no-save`。

**oracle 基线（20 集，seed 5，`--no-save`）**：`success 1.0000`、`spl 0.9608`、`soft_spl 0.9267`、`mean distance_to_goal 0.1510`、碰撞 0。这条链路端到端是通的。

**跨墙吸附守卫**（`snap_goal_to_navmesh`，用户点名要求）：`snap_point` 只按欧氏距离找最近可行走点，不知道墙，可能把目标吸到墙对面。三个检查——吸附位移 `offset`、`find_path` 连通性、测地/直角的绕行比。阈值**按实测分布定**，在 8 个 episode 的 **8203 个标注 view point** 上量的：

- `offset` 最大只有 0.22 m（标注点本来就在 navmesh 上）。
- `excess = 测地 − 直线`：中位数 1.19 m、p90 **6.46 m**、p99 **36.8 m**。→ **绕行几米到几十米是室内常态**，用绝对绕路量当判据会误杀 p75 以上的合法目标（旧的 `excess > 4` 误杀约四分之一）。
- `ratio`（直线 ≥ 1 m）：p99 = 3.08、最大 3.29。→ `ratio > 4.0` 在那 8203 个点上**零误杀**，同时仍抓得住实测的跨墙案例（直线 0.85 / 测地 3.44 → ratio 4.05）。

两个判据覆盖**不相交**的距离区间（`euclid < 0.5 m` 用绝对量，否则用比值），重叠会让"目标在房子另一头、要绕 20 m"被绝对判据误杀。oracle 模式传 `guard=False`：view point 是数据集给的标准答案，拿跨墙检查去否决标注没有道理。

**已知残留**（实测确认无解，不是没做）：墙两侧离目标一样近时（实测 1.55 m vs 测地 5.96 m，ratio 3.84 < 4.0）最近的可行走点本来就是二选一，守卫放行。试过的判据都不成立——`try_step_no_sliding`（habitat 自己移动用的函数）同房间落点偏差 1.3–2.5 m、跨墙 1.24 m，没有区分度；沿直线采样 `is_navigable` 两边命中率都是 0.17/0.19。代价有界：吸附点仍在原目标 0.33 m 内，多走 4.41 m，而且 offset/euclid/geodesic/detour 全都打印出来可供人工判断。

**验收**：

```bash
python test/pix2move/run_pix2move.py --oracle --episodes 20 --seed 5 --no-save   # spl 0.9608
python /tmp/t_guard.py        # 34 passed —— 守卫逻辑（假 pathfinder，不依赖仿真）
python /tmp/t_guard_live.py   # 10 passed —— 守卫在真 navmesh 上（含真实跨墙案例）
```

---

### 2026-09-12 修复：「跨房间移动报 REJECTED：与 agent 不在同一连通片」

用户反馈：跨房间移动直接报 `REJECTED: 与 agent 不在同一连通片（被墙或家具隔开）`，并给出解法方向——「请你尝试使用 目标位置 与 当前位置的连线上的点，投影到 nav 上，直接保证不会穿墙」。

**先量，再改。** 在 12837 个标注 view point 上扫了一遍（`/tmp/t_island.py`）：

```
find_path 失败 4634 (36.1%)
  其中目标在**别的**小岛: 4634
  其中目标在**同一个**小岛却仍失败: 0
```

**36.1% 的目标与 agent 不在同一个小岛上，而所有失败无一例外都是跨岛。** 也就是说这不是吸附吸错了——HM3D 的 navmesh 是真的分片（每个场景 1–3 个互不连通的岛），目标真的在另一个岛上，**没有任何纯 navmesh 方法能走过去**（habitat 自己的 `distance_to_goal` 对这种目标也返回 inf，ObjectNav 基准就是这么算的）。之前的做法是原地拒绝，用户看到的就是"跨房间就报错"，而目标常常就在隔壁、画面里看得见。

**改法**（就是用户提的那条）：沿 `agent → goal` 连线采样，从目标端往回扫（`t = 1 → 0`），第一个既在 navmesh 上、又能 `find_path` 到 agent 的采样点就是沿这条线能走到的最远处（`farthest_reachable_along`）。它**天然不会穿墙**：越过墙的采样点落在另一个小岛上，`find_path` 直接失败，于是被跳过。机器人朝目标走、停在能走到的边界上，`info["fallback"]=True`、`info["advance"]` 报告推进比例。

实测推进（4634 个跨岛目标）：中位数 **39%**、p75 79%、p90 84%，完全没有推进（<1%）的占 **0.0%**，能一步到位（>99%）的也是 **0.0%**——符合"目标真在另一个岛上"的判断。

**回退也过绕行守卫。** 光有上面的回退还不够：连线上越过墙的那个点虽然**可达**，却可能要绕很远。实测 60 个跨岛目标里有 **15 个**是这种（直线 2.9 m / 测地 **24.4 m** / ratio **8.5**）——照走等于原地绕一大圈。所以把绕行判据（`detour_reject`，就是原来的第三个检查，抽成函数）也传进采样循环，一路往回收，收到一个既走得到又不绕远的点为止。

**顺手挖出一个更好的解释**：那 15 个被拒的回退点，**无一例外是"楼上/楼下"**。在某个场景上扫 300 个同岛随机点（`/tmp/t_same.py`）：

```
被拒 81 个 (27.0%)，这 81 个的 |Δy| 全部 = 2.80 m   ← 正好一层楼高
放行点的 |Δy| p50 = 0.00
同层(|Δy|<0.3)且 3 m 内：21 个，被拒 0 个
```

所以守卫根本不是在误杀同层目标，而是在**当楼层探测器用**。于是拒绝理由里直接把这个说出来：`…，高差 2.80 m —— 目标多半在楼上/楼下`，比甩一个"绕行比 6.84"有用得多。`detour_reject` 因此多了个 `dy` 参数（默认 0，|dy| > 1.5 时才提）。

**实测验证**（`/tmp/t_fallback.py`，16 项全过）：

- 60 个跨岛目标：**拒绝 0 个**，全部给了回退点；回退点全在 agent 所在岛上、全部 `find_path` 可达、全部比 agent 更靠近目标、ratio 全部 ≤ 2.86。
- 回退点的投影参数全在 `[0, 1]`（没跑到 agent 身后、也没越过目标），垂直吸附偏移 ≤ 1.29 m。
- 挑推进最多的那个（advance 94%）实机 `pursue`：**146 步走到位 0.07 m**，`arrived=True`，episode 没被结束，距离目标从 10.48 m 到 **1.01 m**。
- 同层同岛目标 0/30 被拒；跨层目标被拒时明说高差。

**oracle 基线未变**：`success 1.0000`、`spl 0.9608`、`soft_spl 0.9267`、`mean distance_to_goal 0.1510`，与改动前逐位一致（oracle 走 `guard=False`，且它瞄的是最近 view point，正常不触发回退）。

**验收**：

```bash
python test/pix2move/run_pix2move.py --oracle --episodes 20 --seed 5 --no-save   # spl 0.9608
python /tmp/t_guard.py        # 57 passed —— 守卫 + 回退逻辑（假 pathfinder，不依赖仿真）
python /tmp/t_fallback.py     # 16 passed —— 回退在真 navmesh 上（60 个跨岛目标 + 实机 pursue）
```

---

### 2026-09-14 改版：Tk 窗口 → 网页界面（play2nav 首次进本文档）

**起因**：`play2nav` 上一轮交付的是一个 **Tk 窗口**，但这台机器**没有图形界面**（`DISPLAY` 未设、
无 Xvfb），所以那层代码从写出来到删除**一次都没有真正运行过**——它是整个项目里唯一未经执行的
路径。改成**网页**同时解决两件事：界面终于能被使用（在你自己电脑的浏览器里打开），而且它落在
**可以被自动验证**的 HTTP 层上。

三个已确认的决定：**删掉 `ui.py`**（不保留 Tk 模式）、**网页里带实时俯视小地图**、
**监听 `0.0.0.0`**（直接访问 `http://10.12.208.214:8080`；页面上没有登录，同网段任何人都能驱动）。

**新增**：`play2nav/web.py`（Flask 路由 + 帧槽 + 意图标志）、`play2nav/page.html`（单页界面）、
`play2nav/test_webserver.py`（网页层端到端自检，43 项）。**删除**：`play2nav/ui.py`。
**修改**：`app.py`（`run_gui` → `run_web` + `_web_loop`；新增 `--host/--port/--jpeg-quality`；
`_record_step` 里递增 `frame_seq`）、`__init__.py`、README、`keystate.py` 与 `test_keystate.py`
的措辞（改成界面无关）、`CLAUDE.md`。

**线程模型**（这块最要紧的约束，细节见 `play2nav/README.md`）：`habitat.Env` 不是线程安全的，
所以 **sim 循环在进程主线程**、独占 `env`；**Werkzeug 在 daemon 线程**、**永不接触 `env`**，
只改按键状态（一把锁）与置意图标志。意图用**标志**而不是队列（`/stop` 会被重复 POST，队列要
再去重；标志天然幂等）。状态发布是**整体替换一个只含基本类型的 dict**，`/state` 只读一次引用
——不阻塞、不撕裂、够不到模拟器。用 `make_server(...)` 而不是 `app.run()`：**没有 reloader**
（reloader 会 fork 出第二个进程，等于开两个 `habitat.Env`）。

**顺带修掉三个既有真 bug**（前两个正是"那层从没跑过"的证据）：

1. **`frame_seq` 从不递增**。它只在 `reset_display_cache()`（只被 `start_episode` 调用）里 +1，
   而唯一读者是界面的刷新判断——所以**那个窗口每集只会画出第一帧，之后整集都不更新**。
   → 改到 `_record_step()` 里递增（"产生了一帧新画面"的唯一事件），`reset_display_cache` 随之删除。
2. **`cv2.imencode` 要 BGR，而 `nav.rgb` 是 RGB**。不转的话红蓝互换，画面看着仍然"像那么回事"，
   **没有参照图根本发现不了**。→ `COLOR_RGB2BGR`，并用 `/frame.jpg` 与同集 `rgb.mp4` 第 0 帧
   的比对来守（实测平均差 **2.30**/255）。
3. **小地图画布每步被重建两次**。`topdown_canvas()` 实测 25 ms（步数少）到 100 ms（2000 步，
   轨迹折线随步数线性增长），而 `_record_step()` 每步已经调用过它一次。→ 存下画布给网页复用，
   缩放+编码只要约 9 ms。

**一个重要的实测发现（推翻了原计划里的一条判据）**：换场景时 `env.reset()` 要在原生代码里把
glb + navmesh 重新装进模拟器，这段时间 habitat-sim **占着 GIL**，于是**整个 Python 进程**都
停摆——不只是 HTTP 线程。实测一次换场景让 `/state` 整整 **3 秒以上**不应答，同进程里一个只做
`sleep` 的心跳线程也被饿住 **3.73 秒**（探针时间线：`finished x2 -> loading x1 -> 超时 x1 ->
loading x1 -> navigating x22`，卡顿完全落在 `loading` 窗口内）。

所以原计划里"换场景期间每 0.25 秒打一次 `/state`，**每一次都必须在 2 秒内返回**"这条**做不到**，
不是线程模型的问题，是解释器被冻住了。要消掉这几秒只能把模拟器挪到另一个**进程**里。自检里
因此改成断言：**换场景结束后服务端立刻恢复**（其后 22 次取样全部成功且 < 0.5 秒），并额外断言
**卡顿与心跳被饿是同一回事**（`3.73s vs 3.00s`），把"服务端自己慢"这种情况排除掉。用户可见的
表现是"换场景时画面和状态停住几秒"，而页面在冻结**之前**就已经收到了 `phase=loading`
（「正在加载场景…」），所以不会误导。

**另一个只有真机手打才会现形的 bug（已修）**：服务端的 5 秒死人开关原本按"最后一次收到**任何**
请求"计时，而页面自己每 300 ms 就在轮询 `/state`——于是那个计时被无限刷新，开关**永远不会
触发**。实测按住 `w` 之后只轮询 `/state`，agent 一路走到 72 步都没停。改成只认
**`POST /key`（心跳）**之后正常。自检里也补了一条：按住后只轮询不发 `/key`，键必须在第 5 秒
被松开。**这类"两个机制互相抵消"的 bug，自动自检没覆盖到，是手打真机才发现的**（见坑 43）。

**验收**（全部本机实测）：

```bash
python play2nav/test_webserver.py     # 43/43 通过
python play2nav/test_keystate.py      # 13/13 通过（改动只碰了措辞）
python play2nav/app.py --headless --scripted "w,w,w,a,a,w,w,w,p" --stage val \
    --episodes 1 --selfcheck          # 逐字未变，含 goal ref 4.5912 == 4.5912
```

`test_webserver.py` 的关键结果：长按 0.9 秒走 4 步（判据 3～10）、松手后静置 4 步再过 1 秒仍
4 步（不卡键）、连按 5 次只多 1 步、回退 `seq` 被明确拒绝且 agent 不动、空闲 3 秒**恰好 1 帧**、
两个并发流读者各拿 6 帧、`/frame.jpg` 与 `rgb.mp4` 第 0 帧平均差 2.30/255、`/state` 的
`step_idx` 与磁盘 `metrics.json.num_steps` 一致、换场景真的换了场景、`/quit` 后服务端停止应答
且 `session_summary.json` 有 2 集落在不同目录。另外 stderr 里 `GET /state` 之类的 access log
行数为 **0**（`_QuietRequestHandler` 只掐 `log_request`，报错与 traceback 照常输出）。

**仍然只能由人确认的**：浏览器的渲染与真实按键事件（本机没有浏览器、没有显示器）。清单在
`play2nav/README.md` 的「我没能验证的部分」。

---

### 2026-09-14 修复：网页卡顿 / 按 w 不动 / 按住 1 秒才动 / 输入检测栏 / 画面不许跳跃

**起因**（用户五条，逐条对应下面五节）：页面按一下要等 1~2 秒才更新视图；按 `w` 根本没向前走；
希望改成"按住 1 秒后才持续移动"；左侧「状态」下面加一栏「输入检测」滚动显示按的键（卡住提示
也放这里）；显示的画面严格不许跳跃、必须流式输出。

**1）卡顿：两个叠在一起的原因，都实测过**

- **连接排队**。俯视图原本是"每出一帧就换一次 `<img src="/topdown.jpg?seq=N">`"，页面每 300 ms
  从 `/state` 的 `frame_seq` 变化驱动刷新，等于**每秒新开约 7 条 TCP 连接**。而浏览器对同一
  域名**只允许 6 条并发连接**，其中一条被 MJPEG 常占着——按键 POST 排在图片请求后面等着。
  → 俯视图改成**第二条 MJPEG 流** `/topdown_stream`，不再逐帧开连接（也正好满足"不许跳跃"）。
- **Nagle + 延迟 ACK**。Werkzeug 推一帧分 4 次 `write()`（`wbufsize = 0`，无缓冲），Nagle 会把
  后面的小包压住等前一个的 ACK。→ `_QuietRequestHandler.disable_nagle_algorithm = True`。
- **服务端每步的开销**。实测 `env.step` **52.8 ms**，而 `topdown_canvas()` **58.4 ms**，其中
  `colorize_topdown_map` 一个人就 **48.7 ms**（整张地图 + 迷雾，一步之内几乎不变）。→ 上色结果
  缓存、每 `--topdown-refresh`（默认 0.2 s）才重算，中间每步只 `canvas.copy()`（0.3 ms）；并且
  **第一视角画面在俯视图与 mp4 编码之前就推出去**（`Navigator._emit_frame()` 放在 `_record_step()`
  之前），可见延迟不再被最贵的那段拖住。
- **带宽**：实测 640×480 q80 = **27 KB/帧**，320×240 q70 = **7 KB**。默认改成
  `--display-scale 0.5 --jpeg-quality 70`（约 4 倍省），`--fps` 现在只管俯视图。

**2）按 w 不动的真相：出生点被顶住了，不是 bug**

实测那个 episode：连发 **20** 次 `move_forward` 总共只挪 **0.4501 m**（9 步那次也是 0.4501，
**头两步就走满、之后饱和**），`collisions.count = 19/20`。所以"按 w 没反应"是真的——agent 卡在
出生点旁边的家具上。这与 CLAUDE.md 里"出生点可能顶着家具，遇到走不动就先转身"是同一件事，
只是这次要**把它讲出来**：新增 `Navigator.stuck_hint()`，同时出现在顶栏横幅、输入检测的实时行、
以及输入检测的**滚动事件流**（能看出是哪一步开始撞住的）。

**判据踩了一个坑（由自检抓住）**：第一版把 habitat 的 `collisions.is_collision` 也当判据
（`last_collided or blocked_steps >= 2`），结果在**正常行走**时一路误报——自检里第一集按住 `w`
每步走满 **0.2915 m**，提示却出现了。原因是那个标志是**接触**标志（`sim.previous_step_collided`
一类），在有家具的房间里擦到椅腿、门框就置位，**与有没有走动无关**。→ 判据只留**位移**：
连续两步都 `< BLOCKED_DELTA_M = 0.02 m`（标称一步 0.25 m）。`is_collision` 降级为旁证，在实时行
里写作「（有接触）」。

**3）按住门槛**：`HOLD_DELAY_S = 1.0`（`--hold-delay 0` 退回旧行为）。**最要紧的不变量**：
`KeyState.press()` 对**已经按住**的键**不得**重置计时起点——前端每秒一次的心跳会把按住集合
重发一遍，若每次都把 `_pressed_at` 推到现在，按住时长永远停在 1 秒以内，门槛**永远跨不过去**，
表现就是"按多久都不动"。这条有专门的单测（`test_heartbeat_does_not_reset_hold_clock`）守着。
界面会在实时行里显示倒计时，松手时事件流写明"只按了 0.25s，不够 1.0s 门槛 → 没有产生动作"。

**4）输入检测栏**：左侧「状态」下新增一栏，三层——实时行（按住倒计时 / 上一动的结果与位移 /
走不动提示）、事件流（按下、松开、清空、乱序丢弃、死人开关，逐条带相对时间戳，终端式滚动）、
诊断行（服务端推送帧率与带宽、两条流的读者数、JPEG 质量与显示比例、`/state` 往返耗时）。
服务端侧只在**状态变化**时记事件（松开那条不记就说不清"为什么没动"），连续重复的消息合并。

**5）画面不许跳跃**：两个画面都是**常开的 MJPEG 流**（`/stream` + `/topdown_stream`），浏览器用
原生 `<img>` 吃，`page.html` 顶部注释里写明了**不许**改成 fetch+blob 手动解码。

**顺带修的第六件（改完后必须重启服务才生效，重启途中最容易踩到）**：`SIGTERM` 原先直接终止
进程、**不走 `finally`**，于是正在写的 mp4 停在半截（实测 `rgb.mp4` 只剩 48 字节，只有文件头，
`metrics.json` 根本没有）。→ 在 `main()` 里把 `SIGTERM` 转成 `KeyboardInterrupt`，`kill` /
`pkill` 就和 Ctrl-C 走同一条收尾路径（`run_web` 的 `finally` → `nav.finish("quit_early")`），
并捕获 `KeyboardInterrupt` 免得一次正常中断甩出一段 traceback。**实测**：`kill -TERM` 之后
`metrics.json 1301B / rgb.mp4 217KB / topdown.mp4 22KB / trajectory.png 82KB + session_summary.json`
全部写完，退出码 130。用户可见的收益是「改完代码重启服务不会毁掉正在录的那一集」。

**自检自己的一个 bug（被这条新断言抓住）**：判断"这一集走不走得动"原本用**整段窗口的最大位移**，
而出生点被顶住时 agent 仍能先迈出一两步（0.4501 m 是两步就走满的），于是把过程**判反**——末两次
位移已经是 `[0.0, 0.0]` 了，却因为窗口里有过 0.2001 m 而走"没被挡住"那一支。→ 改成看**末两次
取样**，并把判据抽成 `check_stall_reporting()`，两集都跑、正反两向都断言。

**验收**（全部本机实测）：

```bash
python play2nav/test_webserver.py     # 57/57 通过（原 43 项 + 按住门槛/轻点无效/走不动双向）
python play2nav/test_keystate.py      # 21/21 通过（原 13 项 + 按住门槛 8 项）
python play2nav/app.py --headless --scripted "w,w,w,a,a,w,w,w,p" --stage val \
    --episodes 1 --selfcheck          # 逐字未变，含 goal ref 4.5912 == 4.5912
# 真实启动命令的冒烟测试（自检走的是进程内 make_server，不是这条 CLI 路径）
python play2nav/app.py --port 18099 --episodes 1   # / 200 + /frame.jpg + /topdown.jpg + /state 全通
```

**注意**：改完代码后 **8080 上原有的服务进程跑的还是旧代码**（Python 不热加载）。实测当时
`8080` 上挂着一个更早启动的实例，必须重启才会生效——`Ctrl-C` 或 `kill <pid>` 都会优雅落盘，
**`kill -9` 会毁掉正在录的那一集**。
```

`test_webserver.py` 新增的关键结果：按住 0.8 s 时 `0 -> 0`（一步都没走）、越过门槛后读数严格
递增 `[0,1,2,3]`、轻点 `w` 0.25 s 一步不走且事件流写明原因、第一集每步 0.29 m 时提示**未出现**
（`is_collision` 期间为真也不报）、第二集末两次位移 `[0.0, 0.0]` 时提示**出现**且同时进了实时行
与滚动事件流、网页画面 320×240 而录像仍是 640×480。

**仍然只能由人确认的**：浏览器渲染是否**真的连绵不断**（我只能验到"HTTP 上每出一帧就推、
空闲不推"），以及真实按键的手感。清单已更新在 `play2nav/README.md` 的「我没能验证的部分」。

---

### 2026-09-14 修正：轻触走一步 / 按住 1 秒转连续；「撞住」改按 `env.step` 前后位置判

用户的反馈：「延迟比上一步好很多」（保持）＋ 两点纠正：

- **`--hold-delay` 被理解反了**。上一版做成了"按住满 1 秒才出动作、轻点无效"，用户要的是
  **轻触一下走一步、按住满 1 秒才转连续移动**——"不是每次移动都要按超过 1s"。
- **"走不动"的判据过于严格**。要求直接用 `env.step` 的结果判：**前后位置相同**就提示
  「碰到障碍物，无法移动」。

1. **轻触一步 + 按住转连续**（[play2nav/keystate.py](play2nav/keystate.py)）。按下当刻先出
   **一步**：`_tap_used[key]` 标记这一步有没有发出去，`active_key()` 取走时置位（因此它
   **会改状态**，docstring 里写明了），每个键每按一次只给一步。连续输出仍由 `HOLD_DELAY_S`
   决定。两条不变量不变：心跳**不得**重置计时起点，也**不得**补发那一步（否则表现为"按多久
   都只走一步"）。前端文案、`web.py::_release_text`、`_input_live`、`--hold-delay` 的 help
   同步改。
2. **撞住只看位置**（[play2nav/app.py](play2nav/app.py)）。`blocked_steps`（连续两步位移
   < `BLOCKED_DELTA_M = 0.02`）整块删掉，换成
   `last_stuck = last_action == "move_forward" and last_delta_m <= STUCK_EPS_M`（1e-4 m）。
   依据是 `allow_sliding` 关着时被挡住就**一步都不走**：实测"连发 6 次 `move_forward`，后 5 步
   位移都是 **0.0000**"，而正常一步是 0.25 m，中间没有别的取值。所以不需要经验阈值，也**不再
   需要连撞两步才报**——轻点 `w` 撞上就立刻提示。`collisions.is_collision` 降为旁证（实时行里
   的「（有接触）」），理由见坑 46。
3. **自检跟着改**（[play2nav/test_webserver.py](play2nav/test_webserver.py)）：5a 从"按住
   0.8 s 一步都不许走"改成"轻触走出**恰好一步**"，5d 从"轻点一步都不走"改成"轻点走且**只走
   一步**"，步数区间改 5~12（轻触 1 + 连续若干）。`check_stall_reporting` 的"不该报"那半边
   改成**逐次取样**判——第一集在窗口末尾正好撞住时，只看末项会让这半边**没人验**（实测就是
   这么发生的：两个 episode 都走了末项那个分支）。

4. **取键必须排在动作间隔闸门之后**（`_web_loop`）。`active_key()` 现在会**吃掉**轻触那一步，
   而原来的写法是"先取键、再判断 `(now - last_action) >= args.action_interval`"——被闸门挡掉的
   那一步就**白丢了**：在离上一步 0.15 s 以内轻点一下会毫无反应。改成先过闸门再取键。这类
   "查询带副作用"的接口，调用点必须保证返回值不会被丢弃。

实测（`test_webserver.py` **60/60**、`test_keystate.py` **22/22**、无头回归与改动前**逐字
一致**：`goal ref 4.5912 == 4.5912`、`selfcheck OK (max diff 1px, 64 points)`、`steps 9
path 0.45 m`）：轻触 0.75 s `0 -> 1`；越过门槛后 `[1,2,3,4]` 严格递增；按住 1.9 s 共 5 步
（轻触 1 + 连续 4）；松手后 5 步不再变；轻点 0.26 s `6 -> 7`；连点两下 `7 -> 9`；第一集 4 次
取样、第二集 6~7 次取样里**凡有位移的一次都没误报**（最大 0.2915 m），两集末次位移 0.0 时
提示出现且三处都在。

### 2026-09-14 修复：「两次按中一个键后画面才更新」，以及画面必须追平最后一次输入

用户反馈：**「两次按中一个键后画面才会更新。怀疑画面与当前 obs 不匹配，显示的是上一步动作
执行后的结果」**，并要求「如果画面落后，请持续更新到最后一次输入的结果」。

**先说没复现的部分**（当时的第一嫌疑是画面管线）：探针按"客户端收到的最后一帧 vs 当场重编码
的 `nav.rgb` / `nav.last_topdown`"逐字节比对，8 次取样**两个画面都完全一致**（不能用帧的 md5
判身份——被挡住时相邻两步的画面逐字节相同，那样比会得出"负数延迟"这种荒唐结论）。`frame_seq`
9/9 全部推了出去，publish → 客户端约 **1 ms**。所以**没有一个固定的"落后一步"**。

真正的成因在**按键状态机**，两类，都能用探针稳定复现（修前 → 修后）：

| 场景 | 修前 | 修后 |
|---|---|---|
| 极短轻触（30 ms）整段落在上一步的执行期内 | `忙碌窗口内的极短轻触: 期望 2 步，实际 1 步` | 2 步 |
| 连点两下，第二下距松手 0.00 s / 0.05 s | `期望 2 步，实际 1 步`（≥0.10 s 才正常） | 2 步 |

两者都表现为**「点了没反应，再点一下才动」**——而那"再点一下"让画面变了，看起来就像"画面比
输入落后一步"。修法四条：

1. **轻触的那一步改成"离散事件"**（[play2nav/keystate.py](play2nav/keystate.py)）。新
   `pending_tap()` **不再要求取键那一刻 `is_held`**，欠着就一定发得出去，上限
   `TAP_PENDING_S = 2.0`（换场景要 3~8 秒，换之前那几下不该在换完之后补出来）。sim 线程走一步
   要 200 ms 上下（上色 + 录像编码），一次 30 ms 的轻触完全可能整段落在这一步的执行期内——旧
   写法要求"取键时还按着"，那一下就**被静静丢掉**。
2. **心跳必须显式标出来**：前端心跳带 `hb: true`，服务端 `press(..., heartbeat=True)` 对它
   什么都不改；而**不带 `hb` 的按下，哪怕距上一次松手只有几毫秒，也必须是新的一次按下**
   （重新计时、重新拿到一步）。两者在时序上无法区分，靠 `RELEASE_GRACE_S` 猜就会把宽限期内
   的第二下吞成心跳 = "连点两下只走一步"。心跳同时**不再让已松手的键复活**（否则一个迟到的
   在途心跳会把松手后已经开始走路的机器人重新按下去，一路走到死人开关）。
3. **限速只推迟、不吞**（[play2nav/app.py](play2nav/app.py) 的 `_TopdownGate`）。旧写法在
   限速窗口内到达时直接推进了"已发布帧号"，那一帧就**永远丢了**——画面停在旧帧上，要等用户
   再按一下才更新，正是用户描述的另一半成因。另外给第一视角加了**每 tick 兜底补推**
   （判据 `emit_seq` 与已推帧号是否相等），并让编码失败不再把整个会话带下去。
4. **让落后看得见**（[play2nav/page.html](play2nav/page.html)）：诊断行新增
   `画面帧 第一视角 已推/当前 · 俯视图 已推/当前`，两个数相等才算追平，落后会标 `⚠`；输入
   检测实时行新增「`w` 轻触的一步已收下，待发（sim 线程正在执行上一步）」。

顺带修掉一个**新写的指示器的 bug**：第 4 条那对帧号里，俯视图那一路忘了在真正发布的地方写
`published["topdown"]`，指示器会永远显示"落后"（假警报）。是自检里的
`pub_topdown_seq == frame_seq` 断言抓出来的。

实测：`test_webserver.py` **70/70**（新增 6 条闸门纯逻辑 + 「宽限期内的第二下 = 第二步」+
「5 次 30 ms 轻触 = 5 步」+ 两条画面收敛）；`test_keystate.py` **28/28**（新增 5 条：轻触扛过
松手、2 秒后不再补、宽限期内第二下、心跳不复活、`debug` 列出欠着的那一步）；轻触探针
**8/8**（上表两行都翻过来了，且「轻触 + 心跳重发 = 1 步」没有回归）；无头脚本回归与改动前
**逐字一致**（`goal ref 4.5912 / 4.5912 -> 一致`、`selfcheck OK (max diff 1px, 64 points)`、
`steps 9 path 0.45 m`）。

**注意**：改动只在磁盘上，正在 8080 上跑的那个服务（pid 2274599，15:26:36 启动）**先于**这些
改动，必须重启才会生效（别 `kill -9`，会把正在写的 mp4 弄坏）。

---

### 2026-09-14 修复：画面**真的**错开一帧 —— 每帧必须发两段 MJPEG part

用户第二次确认，描述得没有歧义：**「每次显示的是上一个动作完成后的 obs……先输入 a，画面根本
不动，再次输入 w，画面变为输入 a 后左转的画面，再输入 d，画面变为前进一帧的画面，而不是右转
的画面」**。上一节修的是**按键状态机**（那是真的、也确实在，探针能稳定复现），但**不是这一个
现象**——上一节末尾"没有一个固定的落后一步"的结论只对**字节层**成立。

**这次能证死，是因为发现本机装了浏览器**（`/usr/bin/google-chrome`，145）。用
`--headless=new --remote-debugging-port=N --remote-debugging-port … --remote-allow-origins=*`
起无头 Chrome，CDP 上 `Page.captureScreenshot` 定时截图（`websocket-client` 在 **base python**
里，`harnessnav` 那个 env 没有——所以探测脚本用 `python3` 跑、像素比对用 `harnessnav` 的
python 跑），再把截图与服务端**真正推出去的那份 JPEG 字节**（在 `publish_rgb` 上挂钩子存盘）
按像素对齐。修前，真 Chrome、真 `page.html`：

| 测量 | 修前 |
|---|---|
| 每次按键后静置 1.5 s，浏览器**画出来**的是哪一步 | 一律是**上一步**（8 次按键错 8 次） |
| 画面上屏发生的时刻 | **正好是下一帧被推出去的那一刻**（例：pub_004 在 65.09 s 推送 → 65.2 s 才画出 pub_003 的内容） |
| 最后一帧 | 直到**流结束**（`/quit` 断开）才画出来 |
| 字节层 | 一直是健康的：publish → 客户端收到 **1~2 ms**，帧号映射全对 |

**成因与修法**：浏览器要等到**更后面一段 part** 到达，才把当前这一帧提交上屏；而这条流在 agent
站着不动时**是静默的**（不推未变化的帧，见上一节的设计），所以这一帧会一直悬着、直到下一次按键
产生新帧——**"按一下不动、再按一下才显示上一个动作的结果"就是这么来的**。用最小 MJPEG 服务做
对照实验（`/push?n=K` 推一帧、静置、截图、与该帧比像素），五种写法：

| 写法 | 静置后画的是最新一帧吗 |
|---|---|
| v1 现状：boundary 只在下一帧的开头发 | 否，推迟一帧 |
| v2 发完帧立刻补下一个 boundary（part 自己结束） | 否，推迟一帧 |
| v3 v2 再把下一段的首部一起发出去（boundary 不再有歧义） | 否，推迟一帧 |
| v4 去掉 `Content-Length`，靠紧随其后的 boundary 界定 | 否，推迟一帧 |
| **v6 同一帧连发两遍（第二遍逐字节相同）** | **是** |
| v7 第二遍隔 80 ms 再发 | 是（但没必要，v6 就够） |

所以 [play2nav/web.py](play2nav/web.py) 的 `_mjpeg` 里**一帧 `yield` 两次**（同一个 `part`
对象，第二遍只是把第一遍顶上去），代价是这一步的字节数翻倍（320×240 约 35 KB → 70 KB）。

**验证**（真 Chrome + 真页面，`--host 127.0.0.1 --port 8099 --out /tmp/...`）：

- 按 6 次 `a`，每次静置后把页面里 `#rgb` 那块裁下来，与**同一时刻**服务端 `/frame.jpg` 逐像素
  比：**差 0.00**（与别的步的帧差 **32~50**），6/6 全对；
- 只发一遍的旧代码在同一个实验里 6/6 全错（"画的是第 N-1 步"）；
- 页面自身的诊断行 `画面帧 第一视角 7/7 · 俯视图 7/7`（无 `⚠`）；
- 排版、两条画面、输入检测栏都正常（这条以前只能靠用户看）。

自检：`test_webserver.py` **74/74**（新增 4 条纯逻辑：一帧两段、两段逐字节相同、两段都带
`Content-Length`、停机即收工；"空闲 3 秒恰好 1 帧"改判为"只推 1 帧、且同一帧两段"——一帧是两段
part，判"推了几帧"要看 `frames_in()` 而不是 `count_frames()`）；`test_keystate.py` **28/28**
未动。

**注意**：8080 上那个服务（pid 2293067，15:47:42 启动）仍然**先于**本次改动，必须重启才会生效。

### 2026-09-14 点云 Occupancy Map + 环视全景 / BEV

从 StateNav `statenav_agent.py` 的 `update_map_memory` / `rotate_panoramic` / `render_bev_map` 抽成库，**只做记忆与可视化，不参与规划**（规划仍是 navmesh）。

| 文件 | 内容 |
|---|---|
| `nav/occupancy.py` | `OccupancyMap`：RGB-D 反投影（**相机**位姿）→ 层高切片 → Open3D 体素融合；`scan_around` 原地右转一圈（可选低头再扫地面）；`render_bev`；全景拼图/落盘 |
| `test/occupancy/run_occupancy.py` | a. 出生点环视并保存全景+BEV；b. 前进若干步后再环视一次 |

约定：

- 反投影用相机位姿 + `geometry.translate_to_world`（habitat 世界系，**不要**再套 `habitat_rotation` 的 Y/Z 置换——那是 pix2move 局部系用的）。
- 深度按传感器 `min_depth`/`max_depth` 清哨兵，避免 `min_depth` 假墙。
- 环视 12 张、先存再转，序号 **0 = 未转动的当前观测**。默认保存 `1,3,5,7,9,11`；`--pano-even` 保存 `0,2,4,6,8,10`。`--no-merge` 分别落盘，默认拼一张；两种方式都在图上写 `Direction N`。
- BEV `--bev-mode {color,bw,both}`（默认 both）：彩色点云或白点黑底占用。红色三角 = 当前朝向；青色射线/红字与全景序号对齐（序号 `i` 是正前方起顺时针 `i*30°`）；**历史**环视点为蓝色空心圆，旁标从 0 起的蓝色序号；**当前**环视点只留红三角。圆之间蓝线 + 移动细轨迹。
- 环视时不把原地转向的抖动记进轨迹（转身 xz 会抖几毫米，画出来会在节点旁长毛刺）。

验证（`h1zeeAwLh9Z` ep27，`--seed 3`）：两圈都出图，第二次 BEV 上两个蓝圈 + 连线。产物在 `test/occupancy/out/`。

```bash
python test/occupancy/run_occupancy.py
python test/occupancy/run_occupancy.py --pano-even --no-merge --bev-mode bw
```

---

## 启动指令

```bash
cd /home/xsuper/hc_workplace/HarnessNav
conda activate harnessnav

# M1 分割测试（图片放 test/seg_test/img/）
python test/seg_test/run_seg.py --text "chair"                 # 两个后端都跑 + 对比图
python test/seg_test/run_seg.py --text "chair" --backend glee

# M0 基线：跑标注目标，出 SPL 等指标（推荐先跑这个）
python test/pix2move/run_pix2move.py --oracle --episodes 20 --seed 5 --no-save

# M0 交互式
python test/pix2move/run_pix2move.py
python test/pix2move/run_pix2move.py --seed 0 --backend glee --glee-threshold 0.15

# play2nav：人用键盘玩。起一个网页服务，终端会打印网址，在**你自己的电脑**上用浏览器打开
# 轻触一下走一步，按住满 1 秒转持续移动（--hold-delay 0 退回「一按下就持续动」）
# 按了没反应看左侧「输入检测」；顶到家具时会提示「碰到障碍物，无法移动…先转身」
python play2nav/app.py                              # 默认 --stage val --episodes 20，监听 0.0.0.0:8080
python play2nav/app.py --host 127.0.0.1             # 只给本机 / SSH 转发访问（页面上没有登录）

# play2nav 的网页层端到端自检（不需要浏览器，约 2 分钟）
python play2nav/test_webserver.py

# 点云占用图：环视全景 + BEV（出生点一圈，走几步再一圈）
python test/occupancy/run_occupancy.py
python test/occupancy/run_occupancy.py --pano-even --no-merge --bev-mode bw
```

`--oracle` 模式：瞄准 episode 标注里**测地距离最近**的 view point（`success` 就是按 view point 判的），用 `sim.make_greedy_follower` 走过去，每集打印 habitat 的 `distance_to_goal` / `success` / `spl` / `soft_spl` / `collisions`，最后给平均值。

交互模式：每轮落盘 `obs/{stem}_{N}.png` + `_map.png`，终端提示 `target | left|right|forward|up|down | stop=quit`。

- **输入语义目标词**（如 `chair`，任意开放词表短语都行）→ 分割 → 反投影 → 吸附 → 自动跟随前进。走到目标 `goal_radius` 内就停下，**但不结束 episode**（`pursue(..., stop_at_goal=False)`），可以接着找下一个目标。
- **目标在另一个 navmesh 小岛上时不再报错**：会打印 `partial: ... 朝目标推进到 N% 处就停`，然后朝目标走、停在能走到的边界上（HM3D 的 navmesh 分片，实测 36.1% 的标注目标跨岛，真的过不去——habitat 自己的 `distance_to_goal` 对这类目标也返回 `inf`）。
- **手动微调**：`left` / `right` 转 30°、`forward` 前进 0.25 m、`up` / `down` 抬头低头，一条指令一个动作。这些是**保留字**，在送去做分割**之前**拦截。
- **`stop` 退出**。

同名帧只落盘一次（`saved_index` 游标），没匹配到实例时终端仍打印 `--- frame ..._N.png ---` 但不重写文件。

判断是否正常：每轮打印的 `navmesh distance to goal X -> Y` 应下降；`_map.png` 上**青色**是 navmesh 测地路径、**洋红**是 agent 走过的轨迹、**黄点**是反投影得到的原始目标点（在物体上）、**红环**是吸附到 navmesh 后的实际导航目标。

常用参数：`--backend {glee,gdino_sam}`、`--seed N`、`--stage {train,val,val_mini}`、`--episodes N`、`--max-actions N`（交互模式每轮最多走几步，默认 5）、`--glee-threshold 0.3`、`--out test/pix2move/obs`、`--no-save`。

`--success-distance` 默认 **1.0 m**（HM3D ObjectNav 标准）。本仓库 YAML 里写的是 0.25 m，启动时会打印 `success : within 1.00 m of a view point (YAML says 0.25; ...)` 提示这处覆盖。用 0.25 m 时几乎不可能 stop 在那么小的范围内，success/SPL 会恒为 0。

**排障：**
- 输入 `left` 却被当成目标去分割了 → 保留字拦截没生效（`MANUAL_ACTIONS`）。**GLEE 是开放词表，阈值调低时什么词都能匹出一个实例**（实测 `left` 匹到 score 0.186、`wall` 匹到 0.264），所以这些词必须在分割之前拦掉
- **走到目标跟前突然整个会话结束** → `pursue` 到达时真的发了 stop 动作，而 habitat 的 stop 会终止任务。交互模式必须传 `stop_at_goal=False`（oracle 评测相反，**必须**发 stop，否则 `task.is_stop_called` 为假、success/SPL 恒为 0）
- `no instance matched 'xxx'` → 该物体在当前视野里，或被挡/太近/太远。默认 episode（GLAQ4DNUx5U ep=25）出生点几乎贴着柜子，`toilet` / `wall` 都匹配不到；`--glee-threshold 0.15` 能匹配到 `cabinet` / `door`，或换 `--seed 0`
- `REJECTED: 吸附点不在 navmesh 上` / `吸附位移 N m > 2.5 m` → 反投影点离任何可行走面都太远（多半不是地面，是吊灯之类），换目标重试
- `REJECTED: 与 agent 不在同一连通片` → 目标和 agent 之间 navmesh 不连通（HM3D 的 navmesh 分片，实测 36.1% 的标注目标都跨岛）。**现在不再原地拒绝**：会沿 agent→goal 连线推进到能走到的边界，打印 `partial: ... 朝目标推进到 N% 处就停` 然后照常走。只有在**连线上一个既可达又不绕远的点都没有**时才会真的 REJECTED
- `REJECTED: 绕行比 N > 4.0`，带 `高差 X m —— 目标多半在楼上/楼下` → 十有八九是楼上/楼下（实测被拒点的 |Δy| 清一色 2.80 m）。同层目标基本不会撞这条：实测同层 3 m 内 21 个点零误杀。不带高差提示的才是真的跨墙，见上文「已知残留」
- `跟随器无法从当前位置规划到目标（GreedyFollowerError）` → 同一个连通片问题。回退点已经保证可达，正常不该再看到；看到就是回退点被 `snap_point` 挪过了头，报 bug
- `navmesh 没有加载` → 该场景缺 `.basis.navmesh` 文件

---

## 环境

conda 环境 **`harnessnav`**（从 `statenav` 克隆，Python 3.9、torch 2.5.1+cu121，已含 habitat-sim/lab、opencv、open3d）。在克隆基础上补装：

- `detectron2==0.6` — 源码编译（GLEE 依赖）
- `segment-anything==1.0`
- `groundingdino==0.1.0` — 源码编译自 `thirdparty/GroundingDINO`
- `timm`、`pathfinding`（M0 的 A* 需要）、`addict`、`yapf`、`pycocotools`

编译命令（CUDA 12.2）：

```bash
CUDA_HOME=/usr/local/cuda-12.2 FORCE_CUDA=1 MAX_JOBS=4 pip install . --no-build-isolation --no-deps
```

---

## 网络与镜像（重要，下次直接照抄）

**HF 模型下载** —— 直连 huggingface.co 不通，必须走镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="hf_..."          # 已存在 /home/xsuper/.hf_env（chmod 600，仓库外）
source /home/xsuper/.hf_env
```

**GitHub** —— 直连 github.com 与 `dl.fbaipublicfiles.com` 均超时。改用：

```
https://ghproxy.net/https://github.com/<owner>/<repo>
https://ghproxy.net/https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>
```

**PyPI** —— 清华镜像（已在 `~/.pip/pip.conf`）：`https://pypi.tuna.tsinghua.edu.cn/simple`

**Conda** —— 清华 conda-forge/main 镜像（已在 `~/.condarc`）

**权重直链（hf-mirror）：**

| 模型 | URL |
|---|---|
| GroundingDINO | `hf-mirror.com/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth` |
| SAM v1 | `hf-mirror.com/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth` |
| GLEE | `hf-mirror.com/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE_SwinL_Scaleup10m.pth` |
| CLIP | `hf-mirror.com/openai/clip-vit-base-patch32/<file>` |
| BERT | `hf-mirror.com/bert-base-uncased/<file>` |
| detectron2 | 经 ghproxy clone 后源码编译 |

**大文件下载必须用断点续传+重试**（镜像会中途掐断，且 wrapper 脚本仍以 0 退出，具有误导性）：

```bash
curl -sSL -C - --retry 3 --retry-delay 5 --connect-timeout 30 \
     --speed-time 60 --speed-limit 10240 -o "$out" "$url"
# 外层 for i in 1..8 循环，每轮用 stat -c%s 对比预期大小
```

---

## 踩过的坑（下次避免）

1. **GitHub 直连超时** → 用 `ghproxy.net` 前缀；权重走 `hf-mirror.com`。
2. **大文件下载被掐断**（GLEE 停在 291MB/1.44GB，SAM 停在 130MB/2.56GB），且 wrapper 退出码仍是 0。→ 上面那条 `-C -` 续传 + 重试循环，必须显式校验文件大小。
3. **detectron2 editable 安装失败**（`ModuleNotFoundError: No module named 'torch'`）—— 其 `setup.py` 的 `develop` 会重新调 pip 且不带 `--no-build-isolation`。→ 用非 editable + `--no-deps` 安装。
4. **`groundingdino.util.inference` 拖入 `supervision` → `deprecate`**，而 `deprecate` 装不上。→ 放弃它，在 `gdino_sam_backend.py` 里自己写推理，只 import `build_model` / `SLConfig` / `clean_state_dict` / `get_phrases_from_posmap` / `datasets.transforms`。
5. **GroundingDINO 运行时联网找 `bert-base-uncased`** → 手动下载到 `model/bert-base-uncased/`，并设 `args.text_encoder_type` 指向本地。现已完全离线可跑。
6. **GLEE 代码里硬编码了作者机器的 CLIP 路径** → 改成读环境变量 `GLEE_CLIP_PATH`，默认 `model/clip-vit-base-patch32`（见 `thirdparty/GLEE/glee/models/glee_model.py`）。
7. **`pathfinding` 包根本没装**（StateNav 里 `path_planning` 是死代码，其 mapper 并不调用它，所以一直没暴露）。→ M0 需要，已 `pip install pathfinding`。
8. **`pathfinding` 的 Grid 语义容易反**：`walkable = weight > 0`，即 **0 = 不可通行**。所以成本图必须"自由格为正、障碍为 0"，且自由格不能用 1（会被 `planmap[planmap==1]=10` 放大成近乎阻挡）。本项目自由格用 **2**。
9. **`quaternion` 库的约定是 `[w,x,y,z]`**，而 habitat-sim 的 `AgentState.rotation` 是 `[x,y,z,w]` 数组。传错会静默得到错误旋转矩阵 → `nav/transform.py::as_quaternion` 专门处理。
10. **转向正负号别推导，要实测**：真机标定得 `turn_left` 使朝向角变化 **−30°**（turn_sign=−1）。工程中已实现自动标定（`run_pix2move.py::calibrate`），不依赖推导。
11. **`/home` 磁盘曾接近满（97–98%）** → 大文件尽量放 `/DATA_HDD`（1.7TB 可用）。
12. 渲染细节：信息条文字在竖版图上会被截断 → `_fit_text` 缩小/截断；贴边的 box 标签被裁 → `_draw_tag` 钳制进画面。
13. **`sim.set_agent_state` 要的是机身位姿，不是相机位姿。** 本项目 YAML 里传感器挂在 `position: [0, 0.88, 0]`，机身又已在高度 0.88，所以相机在 1.76m。把 `sensor_states['rgb'].position` 喂回去会把 agent 抬高 0.88m 且**不可逆**——而反投影又必须用相机位姿。两者要分开（`body_pose()` / `sensor_pose()`），别图省事混用。
14. **`maps.to_grid(realworld_x, realworld_y, ...)` 的形参名是错的/误导的。** habitat 自己调用时传的是 `to_grid(pos[2], pos[0], ...)`，即第一个参数是 **z**。返回的 `(a_x, a_y)` 又直接用来索引 `map[a_x, a_y]`，所以最终 **row↔z、col↔x**。另外它的 `grid_size` 是按 `map.shape` 每轴各算一次的，而 `maps.calculate_meters_per_pixel` 返回的是两轴的 **min**（单一标量）——非正方形场景下这个标量并不等于两轴的换算系数。要么用 `map.shape` 逐轴算，要么直接用 `to_grid`。
15. **"停在物体旁边"不要靠把终点沿路径回缩实现。** 吸附到最近自由格已经完成了这件事，再回缩一格会让"到达判据"把 0.6m 外当成已到达，表现为"走到目标附近就站着不动"。控制器里任何提前终止条件都要和 `grid_res` 一起核算实际误差。
16. **改朝向的符号/幅度要闭环验证，不能只看单步。** 0.25m/步、30°/步这种离散动作，角度误差和位置误差会互相放大：30° 的朝向误差让一步前进在目标方向上只剩 0.21m 的有效位移。合成闭环测试（模拟器 + 控制器同构，`/tmp/t_nonsim.py`）能低成本地覆盖多个朝向、L 形走廊、贴墙目标这些真机上不好复现的场景。
17. **`nvidia-smi` / EGL 报 driver-library version mismatch**（内核模块 595.84 vs 用户态 595.91）会让 habitat-sim 起不来：`Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 5 EGL devices in total`，而此时 `torch.cuda.is_available()` 仍为 `True`，容易误判成代码问题。**重启机器**即可。
18. **深度传感器用 `min_depth` 编码"这条射线没打到任何东西"，不是用 0 或 `max_depth`。** HM3D 网格有洞，射空非常常见——实测一帧 30% 的像素读数**恰好** `min_depth`。反投影这些点会在正前方生成一堵假墙。`preprocess_depth` 的默认 `lower_bound=0.1` 拦不住它们，必须按传感器实际的 `min_depth` / `max_depth` 设边界（见 `clean_depth()`）。**排查手法**：打印 `depth == min_depth` 的像素数和它们的行分布，如果散布全图而不是集中在某块，就是哨兵值而不是真实近物。
19. **相机脚下那片地面物理上看不见，别把它当障碍。** 高 h 的相机，最低成像光线落到 `h·fy/cy` 处（本项目 0.88×1.621≈1.43m），更近的地面永远落在画面下沿之外。单帧建图时这片"没观测到"的区域正好是 agent 自己站的地方——判成障碍的话 A\* 连起点都不通行，只能靠硬挖格子兜底，而那个补丁又会变成全图唯一的自由区域，导致"吸附目标"永远吸回自己脚下。正确做法是**显式建模这个盲区并当作可走地面**，但要在障碍覆盖**之前**填充，且限制在相机方位角范围内。
20. **栅格范围不能只由观测点云决定。** 一旦去掉哨兵点（坑 18），最近的可见地面就在 agent 前方 1.43m，`pad_cells` 那点余量根本盖不住 agent 自己的位置，起点会掉到图外。范围要取"点云 ∪ 需要额外推理的区域（盲区圆盘、强制自由点）"的包围盒。
21. **画在千像素级地图上的标记，别用固定像素尺寸。** habitat 的 topdown 图约 1000px 见方、agent 图标半径 32px——6px 的点看不见，8px 的环整个被 agent 图标盖住。标记尺寸要按图幅算，并且**最后绘制**。排查"图上没有标记"时，先确认标记是不是画了但被别的东西盖住/太小。
22. **"帧号不递增"多半是重复落盘，不是命名逻辑错。** 主循环里任何 `continue` 分支（没匹配到实例、目标不可达）都不消耗动作数，如果落盘写在循环开头就会反复重写同一个 `_N`。用 `saved_index` 之类的游标挡住即可。多动作一步连跳（`_4` 直接到 `_6`）是**正常**的：名字里的数字是累计动作数，不是帧序号。
23. **`mapper_local_to_world(local_pos, initial_position)` 的第二个参数要传 `habitat_translation(origin_world)`，不是 `origin_world`。** 它只对偏移量做 `(x,z,y)→(x,y,z)` 的重排，原点按局部序直接相加。传世界序原点会把原点的 y/z 也换一次，结果偏出去十几米。**危险之处在于它不报错**——画在图上就是画到画外，`to_px` 返回 `None`，标记静默消失。凡是"坐标算出来但画不出来/对不上"的问题，先做一次 `habitat_translation()` 的往返比对确认坐标系约定。
24. **`env.reset()` 会让之前拿到的一切句柄失效，多 episode 必须每集重取。** 踩了两次，两次都很难从报错本身看出原因：
    - `env.sim.make_greedy_follower(...)` 的返回值持有 agent 的 body 引用，reset（尤其是换场景）后场景图重建，旧引用直接失效 —— 下一步就抛 `InvalidAttachedObject: Attached Object is invalid`。
    - **`env.sim.pathfinder` 更阴**：旧对象**不报错**，还能正常返回结果，只是那些结果是**上一个场景**的 navmesh 给的。表现是 `snap_point` 出界、`is_navigable` 返回 False（当初被误读成"吸附点不在 navmesh 上"）、`find_path` 说目标不可达——于是明明有解的 episode 被判成无解。**凡是 `env.reset()` 之后还用着 reset 之前取的东西，先怀疑这一条。**
25. **`habitat.Env(config)` 只构造，不会自动 reset。** 不 reset 就 `env.step` 会 `AssertionError: Cannot call step before calling reset`，而且 `env.current_episode` 在 reset 之前也是无效的。
26. **argparse 的默认值会盖掉函数签名里的默认值，两处必须一起改。** 跨墙守卫的阈值在函数里写 3.0/4.0、在 `add_argument` 里却还留着旧的 2.0/2.0，于是 CLI 跑起来用的是更紧的那个，正常目标被大面积误杀。现在 `/tmp/t_guard.py` 里有一条断言专门比对这两处是否一致。
27. **有 open-vocabulary 分割在后头时，交互指令词必须当保留字提前拦。** `left` / `right` 这类词送进 GLEE 不会被拒绝，而是**匹配出一个低分实例然后一本正经走过去**——阈值越低越离谱（0.15 时 `left` 匹到 0.186）。这类 bug 不报错、只是行为诡异，很难从输出看出原因。凡是"输入了 X 但机器人做的不是 X"，先查有没有被下游的开放词表吃掉。
28. **`env.step(HAB_STOP)` 会终止 episode —— 这和"走到目标旁边看看"是冲突的。** `stop` 是 habitat 的一等动作，任务收到它就结束（`Success` 也依赖 `task.is_stop_called`）。所以同一个 `pursue` 函数在两种模式下需求正好相反：oracle 评测**必须**发 stop（否则 success/SPL 恒为 0），交互模式**必须**不发（否则到一次目标整个会话就没了）。用 `stop_at_goal` 显式分开。
29. **habitat 深度传感器给的是 `(H, W, 1)`，不是 `(H, W)`。** `depth[y, x]` 取出来是长度 1 的数组，`float()` 转换看着能work但会发 `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar`，未来版本直接报错。`nav/geometry.py` 和 `perception/base.py` 各自在内部 `depth[:, :, 0]` 了一次，所以在 `clean_depth` 里统一压维，别让每处自己猜形状。
30. **HM3D 的 navmesh 是分片的，`find_path` 失败不代表"吸附错了"。** 每个场景 1–3 个互不连通的岛，实测 **36.1%** 的标注 view point 与 agent 不同岛，而**所有** 4634 次 `find_path` 失败**全部**是跨岛、同岛零失败。所以"目标不可达"经常是**真的**不可达——habitat 自己的 `distance_to_goal` 对这类目标也返回 `inf`。遇到 `find_path` 返回 False，先问"是不是在另一个岛上"，别急着怀疑坐标或吸附。想验证就 `pf.get_island(p)` 比对两侧。**推论**：交互式场景下直接拒绝体验很差（目标往往就在隔壁、看得见），应该沿连线推进到能走到的边界。
31. **"沿 agent→goal 连线采样、投影到 navmesh"时，越过墙的那个采样点可能是"可达但荒唐"的。** 它落在另一个岛上时 `find_path` 会失败（这步天然挡住了穿墙），但它也可能**同岛可达却要绕 24 m**（实测 60 个跨岛目标里 15 个，直线 2.9 / 测地 24.4）。所以采样循环里必须**同时**带上绕行判据，一路往回收。只按 `find_path` 是否成功来挑点，会挑到一个"走过去比站着不动还亏"的位置。
32. **绕行守卫抓到的绝大多数是"楼上/楼下"，不是"墙的另一侧"。** 实测某场景 300 个同岛随机点被拒 81 个，这 81 个的 `|Δy|` **清一色 2.80 m**（正好一层楼），而**同层 3 m 内 21 个点零误杀**。所以它不是在误杀正常目标，而是顺带当了楼层探测器——拒绝理由里把 `高差 X m —— 目标多半在楼上/楼下` 直接打出来，比只报一个绕行比有用得多。排查"守卫太严"时，先按 `|Δy|` 分组看。
33. **`env.reset()` 期间整个进程会僵住——不只是你说的那个线程。** `env.reset()` 在 habitat-sim 的原生代码里重新装 glb + navmesh，这段时间它**占着 GIL**，于是所有 Python 线程一起停摆。实测换场景时 `/state` 整整 3 秒不应答，而同进程里一个只做 `sleep` 的心跳线程被饿住 3.73 秒。**所以别把"换场景期间 HTTP 必须一直有响应"当成线程模型的验收判据**，那条做不到；能验的是"换场景**结束后**立刻恢复"。要消掉这几秒只能把模拟器挪到另一个**进程**。
34. **同进程里量 HTTP 延迟，量到的多半是自己的饿。** 第一版自检在客户端线程里量 `/state` 的响应间隔，报出"3.19 秒空档"，看起来像服务端卡死——其实是测量方自己没被调度到（见坑 33）。要分清"服务端慢"还是"解释器冻了"：**用独立进程发请求**，或者**同时量一个纯 Python 心跳线程**。两者一对比，责任方就清楚。（`test_webserver.py` 里 `_poll_probe` 与 `_GapWatch` 就是干这个的。）
35. **`urllib` 读不了 MJPEG 流。** 空闲时读超时是这条流的**正常状态**（画面不变就不推），而 `http.client` 在 chunked 读取途中撞上一次 socket 超时后，会把响应对象标成不可再读，抛 `OSError: cannot read from timed out object`——正常路径被当成致命错误。→ 自己开裸 socket 发 `GET /stream`，超时就 `continue`。（裸 socket 拿到的是 chunked 原始字节，`--frame` 边界有可能被 chunk 界劈开，所以要先剥帧界再数帧。）
36. **`Response(mimetype="text/html; charset=utf-8")` 会产出**双份** charset**（Werkzeug 见到 `text/` 还会再补一次，响应头变成 `text/html; charset=utf-8; charset=utf-8`）。→ 已经带 charset 的值要用 `content_type=` 传。
37. **Werkzeug 每个响应都带 `Connection: close`**（`serving.py` 里写死的），所以浏览器**并行**发出的按键请求**不保证到达顺序**——`keyup` 早于 `keydown` 到达就会永久卡住一个键。→ 前端给每个按键事件带单调 `seq`，服务端丢弃 `seq <= last_seq`。（别改成前端串行发送，那会给每次按键加延迟。）
38. **`navigator.sendBeacon` 的 Content-Type 是 `text/plain`。** `/keys/clear` 若无条件 `request.get_json()` 会直接 400——而这条恰恰是整个设计里**最要紧**的请求（失焦/关页面时不清键，agent 会一直往前走）。→ 那个路由**完全不碰请求体**。
39. **按钮会抢焦点**：点完「进入下一个场景」后焦点留在按钮上，按空格/回车会**再触发一次**（第二次 `/next`）。→ 点击处理里 `blur()`（等价于 Tk 时代的 `focus_force()`）并 `preventDefault` 空格/回车；页面里不要出现任何 `<input>`。
40. **`frame_seq` 应该在哪递增**：它曾经只在换集时 +1，于是界面整集只画第一帧。这个 bug 一直藏在"从未真正运行过"的 Tk 层里（见 2026-09-14 那节）——**没被跑过的代码里的 bug 不会报错，只会等着被继承**。判据：递增点应当是"产生了一帧新画面"的唯一事件（`_record_step`）。
41. **`cv2.imencode` 要 BGR，而 habitat 的 RGB 观测是 RGB。** 不转换就红蓝互换，而画面看着仍然"像那么回事"，**没有参照图根本发现不了**。→ `cvtColor(..., COLOR_RGB2BGR)`，并且**验证方式**是拿网页那张 JPEG 与同集 `rgb.mp4` 的第 0 帧比（同源，互换会表现为巨大差异，别的原因都不会）。
42. **"松手后立刻读 `/state` 相等"不是判卡键的判据。** 松手的一瞬间可能正好有一个动作已经在 sim 线程里发出去了，它落地会把步数再抬 1。→ 静置一个动作间隔（> `--action-interval`）后再看**两次读数是否相等**（卡键是每秒约 6 步，区分度很大）。同理，"回退 `seq` 被丢弃"要断言响应体里的 `{"ok": false, "why": "stale-seq"}`——那是确定的；"步数没动"只是旁证。
43. **"多久没收到请求就松开按键"这种死人开关，会被页面自己的轮询废掉。** 页面每 300 ms 打一次 `/state`，如果把"收到任何请求"当存活信号，那个计时被无限刷新，开关**永远不会触发**（实测：按住 `w` 之后只轮询 `/state`，agent 一路走到 72 步都没停）。→ 计时的对象必须是**"最后一次按键事件（`POST /key`）"**：心跳（按住期间每秒重发一次按住集合）才是存活的真实信号。**这条恰恰是自动自检没覆盖、靠真机手打才发现的**——自检里补了一条：按住后只轮询 `/state` 不发 `/key`，键必须在第 5 秒被松开。
44. **浏览器每域名只允许 6 条并发连接，其中一条被 MJPEG 常占着。** 所以"每出一帧就换一次 `<img src>`"的轮询写法（每秒约 7 条新连接）会把**按键 POST 排到图片请求后面**——表现是"按一下要等 1、2 秒画面才动"，而不是报错。→ 变化频繁的画面一律走**常开的 MJPEG 流**，不要用"轮询 + 换 src"。同类问题排查顺序：先看浏览器开发者工具里那条连接是不是一直 pending。
45. **Werkzeug 推一帧分 4 次 `write()`（`wbufsize = 0`，无缓冲），默认开着 Nagle。** 小包被压住等前一个的 ACK，撞上延迟 ACK 就是每帧固定的几十毫秒停顿。→ `RequestHandler.disable_nagle_algorithm = True`。**凡是"流式输出却一顿一顿"的，先关 Nagle。**
46. **"接触" != "没走动"。** habitat 的 `collisions.is_collision` 反映的是这一步有没有**碰到东西**：在有家具的房间里正常直行擦到椅腿、门框也会置位（实测每步走满 **0.2915 m** 时它为真）。拿它当"卡住"的判据会一路误报。→ 判断"走不动"只能按**位移**；`is_collision` 当旁证。（这条是自检抓出来的：新加的"正常走得动时不该误报"直接失败。）
47. **"连续按住 X 秒才生效"这类闸门，绝不能被心跳/自动重复重置计时起点。** 前端每秒把"当前按住集合"重发一遍，若 `press()` 每次都对已按住的键把 `_pressed_at` 推到现在，按住时长永远停在 1 秒以内，门槛**永远跨不过去**——现象是"按多久都不动"，而代码看起来完全正常。→ `press()` 只在**真正重新按下**时重置，有单测守着。
48. **判断"这一集走不走得动"不能用整段窗口的最大位移。** 出生点被家具顶住时 agent 仍能**先迈出一两步**再饱和（实测 20 个前进总共只挪 0.4501 m，是头两步就走满的），拿窗口最大值当判据会把过程**判反**：末两次位移已经是 0.000 了，却因为窗口里有过 0.2001 而判成"没被挡住"。→ 看**末两次取样**。**自检脚本自己也会写错，新断言第一次就该拿真实数据核对一遍。**
49. **`env.step` 之外最贵的一环是 `colorize_topdown_map`（实测 48.7 ms，整张地图 + 迷雾）。** 它一步之内几乎不变，每步重算是纯浪费；而且**画面推送必须排在它和 mp4 编码之前**，否则可见延迟被最贵的两段拖着。→ 上色缓存 + `--topdown-refresh`，并把 `_emit_frame()` 放在 `_record_step()` 前面。
50. **"轻触一下"和"按住"是两件事，别做成一个闸门。** 把它做成"按住满 X 秒才出动作"看起来符合"防误触"，实际是**短按完全没反馈**——用户想要的恰恰相反：**轻触 = 走一步**（想精确挪一格就轻点），**按住满 X 秒 = 转连续**。实现要点是"那一步每个键只给一次"（`_tap_used`，取走即置位），而**心跳既不能重置计时、也不能补发那一步**——这两条任一写错，现象都是"按多久都只走一步"，而代码看着完全正常。另有一处调用顺序陷阱：取键的函数带了"吃掉那一步"的副作用，**它必须排在速率闸门之后**，否则在小于动作间隔的窗口里轻点会被闸门连人带步一起丢掉（写这一版时就踩了，靠读调用点发现，自检里补了"连点两下 = 两步"）。
51. **"这一步有没有动"只需要看前后位置是否相同，别自己设阈值、更别要求"连续 N 步"。** `allow_sliding` 关着时被挡住就是**硬停**：实测连发 6 次 `move_forward`，后 5 步位移**精确为 0.0000**；正常一步 0.25 m，中间没有过渡值 ⇒ 不需要经验阈值、不需要累计步数（累计反而让"撞上了"要等两步才说）。判据能直接用 habitat 自己的执行结果（`env.step` 前后的机身坐标），就不要另造测量。

52. **「轻触的那一步」是离散事件，不是"按住"状态的一部分——它不能要求"取键那一刻键盘还按着"。** sim 线程走一步要 200 ms 上下（俯视图上色 + mp4 编码），一次 30 ms 的轻触，按下与松手完全可能**整段落在这一步的执行期内**；要求"取键时还按着"，这一下就被静静丢掉。现象是**「点了没反应，要再点一下才动」**，而每一处代码单看都对。→ 欠着就一定发得出去（`pending_tap()` 不看 `is_held`），只给一个上限（`TAP_PENDING_S = 2.0`：换场景要 3~8 秒，换之前的几下不该在换完之后补出来）。探针判据：**期望 2 步、实际 1 步**，一眼就能分辨。
53. **心跳和"真的又按了一下"在时序上无法区分，只能显式标出来。** 前端每秒把"当前按住集合"重发一遍，而用户连点两下时第二下可能距上一次松手只有几毫秒——**两者长得一模一样，要的行为却相反**（前者什么都不该改，后者必须重新计时、重新拿到一步）。靠宽限期去猜就会把第二下吞成心跳 = **"连点两下只走一步"**。→ 心跳带 `hb: true`，服务端分开处理。推论：**任何"重复事件要与新事件区分"的协议都必须带标记，别指望时间窗**。
54. **限速/去重这类闸门只许"推迟"，不许"吞"。** 俯视图限速那版写法在窗口内到达时**直接推进了"已发布帧号"**，那一帧就永远丢了——画面停在旧帧上，要等用户再按一下才更新（和坑 52 合起来就是用户报的那一整现象）。→ 窗口内的新帧记下来、窗口一到补发（`_TopdownGate`）；同一类风险的另一半由"每 tick 兜底补推 + 编码失败不回写已推帧号"兜住。
55. **"已推帧号"这种指示量，必须在**真正发布的那一处**同步更新。** 加画面帧指示器时，第一视角那一路写了 `published["rgb"]`，俯视图那一路**漏了**——指示器于是永远显示"落后"，是个**假警报**：它本身不报错，只会让人一直以为画面有问题。→ 断言必须直接比对语义（`pub_topdown_seq == frame_seq`），不能只看"有没有这个字段"。
56. **MJPEG：一帧要发两段 part，浏览器才会把它上屏。** 浏览器（实测 Chrome 145，有头无头一样）要等到**更后面一段 part** 到达，才提交当前这一帧；而"站着不动就静默、不推未变化的帧"的流（正是本项目的做法）会让这一帧**一直悬着**，直到下一次按键——现象就是**「按一下键画面不动、再按一下才显示上一个动作的结果」**。对照实验里，把 boundary 紧跟帧后发、连下一段首部一起发、去掉 `Content-Length` **全都没用**，只有"同一帧连发两遍"（内容逐字节相同）立刻就好，代价是字节数翻倍。推论两条：**（a）连续流（30 fps）里永远发现不了这个性质**（最多晚 33 ms），所以"帧率高的场景验证过了"不能推出"低帧率/事件驱动也一定对"；**（b）"字节 1 ms 就到了"不能证明"画面对"**——传输层健康与上屏是两件事。
57. **"这层我没法验证"这个前提，动手之前先查一遍。** "浏览器渲染我没法验证"憋了好几轮（README 里甚至写成了给用户的验收清单），直到发现本机**装了 `google-chrome`**：`--headless=new --remote-debugging-port=N --remote-allow-origins=*` 起无头，CDP `Page.captureScreenshot` 定时截图，就能把"画出来的像素"当数据用（`websocket-client` 在 **base python**、`cv2` 在 `harnessnav` env，两边分开跑）。判定手法：把页面里那张 `<img>` 的 `getBoundingClientRect()` 拿回来裁图，与服务端**当刻**的 `/frame.jpg` 比像素——同一帧差 0.00，错一帧差 30~50，一眼可辨。**能自己量的事，别让用户当测量仪**。
58. **"画面的身份"不能用 md5 比"相邻两步是否相同"来判。** 被挡住时相邻两步的 obs 逐字节相同，用 md5 判身份会得出"画面超前于动作"这种荒唐结论（上一轮踩过）；这轮又踩了它的反面：按键序列如果做成"左右左右"交替，画面本来就只有**两种**朝向、帧帧循环，按"帧号"标出的差异全是噪声。→ 验证脚本里**每一步都要产生一个可区分的画面**（连续同向转身，或直接放箭头/数字），再谈"画的是第几步"。

---

## 尚未开始（等后续）

- P1：OpenAI 兼容 VLM adapter + Planner/Mover prompt（默认 Qwen-VL），产出写 `/DATA_HDD/hc/harness_outputs/`
- 基于感知的完整 baseline：分割接进评测回路，跑 val 出成功率/SPL
- 后端推理成本/效果消融（GLEE vs GroundingDINO+SAM）
- 碰撞后重规划

---

## M3 P0（2026-09-15）

双系统 ObjectNav 规划落地：System 1 Planner 做节点级决策，System 2 Mover 在当前 FOV 多腿逼近；Harness 管 FSM、ScanNode、Skill 与建图。底层仍是 navmesh greedy follower（`nav/goto.py`），不接视觉 SLAM、不用占用栅格 A*。

**设计**：[`docs/m3_harness.md`](docs/m3_harness.md)、[`docs/m3_protocol.md`](docs/m3_protocol.md)。

**代码**

- `nav/goto.py`：从 pix2move 抽出反投影、跨墙吸附、`pursue` / `pursue_leg`、对准 yaw
- `nav/occupancy.py`：射线雕刻 FREE/OCC/UNKNOWN 栅格、`extract_frontiers`、BEV 绿点前沿与节点叠加
- `harness/`：`state` / `memory` / `loop` / 脚本 Planner·Mover / skills（Depth Look Recall Verify）

**可视化测试指令**见 [`test/test.md`](test/test.md)。Skills 按 skill 分目录（`test/skills/look/output` 等），memory 分 `nodes/` 与 `traceback/`。

点云高度切片相对**机身**保留 `[y-0.4, y+1.6]`，否则相机在 0.90 m 时旧公式会切掉地板以上几乎所有点。GLEE 同类经 mask IoU=0.5 NMS 后最多 3 个。

P0 脚本 Planner：GLEE(goal) → Verify/semantic；否则 door；否则 leftover 最多的扇区；否则 TraceBack。Mover 选仍大于阈值里最近的候选。

## M3 P1（2026-09-15）

真实 VLM Planner（`vlm/` + vLLM OpenAI 接口）接入 Harness。主入口 [`run_HarnessNav.py`](run_HarnessNav.py)。Arrived 后 Harness 发 `HAB_STOP`。每集终端流水同步写入 `epXXX/debug.txt`。产物 `/DATA_HDD/hc/harness_outputs/p1/<run_id>/`。

Find 时看到目标则 Verify。一致性只看 VLM 两视角。Miss 后若再次看到目标，状态回到 Find。

产物每集含 `debug.txt`、`topdown.mp4`（habitat `top_down_map`）、`metrics.json`（含 `time_cost`、`tool_counts`）。

```bash
python run_HarnessNav.py --episodes 10 --seed 5 --base-url http://127.0.0.1:8711/v1 --model Qwen-VL
```

验收（`20260915_145618`，seed=5，10 val）：mean success **0.80**，mean SPL 0.39。失败 ep003 Miss、ep009 Confirmed 未进半径。

BEV 只画实际走过的轨迹，不再画节点间连线；节点序号与扇区红字同字号。

## Observe + GLEE 收口（2026-09-16）

ScanNode 不再跑 GLEE、不再用分割升 Find。每拍 Planner 先 Observe 六向 `goal_find` / `landmark` / `room_type`；任一 `goal_find` 则 Find；Find 且未 Confirmed 则终态必须 Verify。GLEE 只留 semantic Mover 选点与 Depth。Verify 侧移不靠 mask，一致性只看 VLM 两视角。所有 VLM 通讯解析失败最多重试 5 次，用尽则本集 `aborted`，评测继续下一集。

## 前沿视线、PlannerIn 瘦身、Planner 自主 Stop（2026-09-17）

- 前沿：`extract_frontiers` 用占用栅格 Bresenham 丢掉穿 `OCC` 的质心（或退到可见 FREE–UNKNOWN 边界）；画面候选比较像素深度与几何深度，挡住的绿点不进 Mover。跟随仍是 navmesh greedy follower，frontier 每腿 `max_steps=30`。
- MakePlan / Verify 选定朝向后：若实际走动、到达或被挡住，该向记为已探索；对准后一步未动则不改。
- 每拍结束后 Summary VLM 用英文写 `history[].summary`（VLM 输入输出一律英文）。
- 回合内不读 `get_metrics`。Confirmed 可发 `Stop`；`Arrived` 表示已经 HAB_STOP。官方指标只在集结束后记成绩单。
- Mover 子目标：传感器 `depth_m ≤ 0.5`。

## Planner prompt 补全（2026-09-17）

`vlm/prompts/planner.txt` 按三步写全，回合称 loop。Observe 仍只回六向 JSON；工具 / 终态各写清参数与门控。步骤 2、3 必须先输出英文 `reasoning:` 再调工具或给终态 JSON。不写 Harness/Mover 后台分工。产物 `planner.txt` 在每次 Tool Call 与 Final Action 前写入该段 reasoning。Summary 吃进本拍完整 Planner 交互（Observe、各轮输出与 reasoning、工具结果、终态、执行结果），压成 2–4 句 History。

## 分割阈值放松与单候选直选（2026-09-18）

语义分割一次推理后按 0.25 → 0.225 → 0.20 → 0.175 → 0.15 最多五档过滤，直到 NMS 仍有实例。`run_HarnessNav.py --backend glee|gdino_sam`。Mover 仅在候选多于 1 个时调 VLM，唯一候选直接走。

## 多 Agent 并行（2026-09-18）

`run_multi_agent.py`：``val_epi.txt`` 的 100 / 1000 两档各自用对应 ``num_episode_sample`` 扫一遍，不再把 1000 的前缀当 100。worker 按同一档长 ``reset``，任务取该档前 ``--episodes`` 个。旧缓存无 ``format=num_episode_sample`` 时丢弃 ``[n=100]``。

## 语义落脚改为命中点→相机（2026-09-18）

`foothold_from_hit` 不再从机身往目标射线（会停在门框）。改为从语义命中格沿视线走向相机，离开 OCC/inflate 后取第一格 FREE 作为 A* 终点；失败再以命中点为圆心 ``nearest_walkable``。

## 探索点过滤与规则选点（2026-09-21）

提取探索点时丢掉开口短于约 0.6 米的家具口袋；最少连通格子改为 8；近处点合并，每个朝向最多 2 个。近距离行走不再问大模型：探索模式在该朝向占用图点中取路径最长者；识别物体按 0.7 置信度加 0.3 归一化深度。一步未动不把该朝向记成已探索，且转回本圈扫描朝向。未能移动也保存第一视角图；流水日志写第几圈。

## 语义与探索点分工（2026-09-21）

规划提示词写明：有标志物或与目标相关的物体时用语义模式搜当前房间、靠近目标；只有地面、门框、走廊或空开口时用探索点穿过门框或沿走廊。语义查询词不得是门、门口、走廊、地面、墙。主循环校验拒绝这些词。语义对准后逐档降低阈值仍无实例时，本圈把「无法有效识别到物体…请尝试其他的方案」写回规划器再规划，不重新环视，最多回退两次，流水组件为 `retry`。脚本规划器不再对门做语义规划。


`run_multi_agent.py` 不再在每个 Agent 进程里加载 GLEE。主进程先起 ``--seg-num`` 个分割 worker（默认 10，摊在 ``--gpu`` / ``--seg-gpu`` 默认 ``0,1``），共享请求队列；Agent 经 `perception/queue_backend.py` 的 `QueueSegBackend` 入队等 mask。`segment_relax` 仍在 Agent 侧。协议单测：`python test/seg_test/test_queue_backend.py`。

## 评测日志与摘要（2026-09-21）

评测入口在导入仿真库之前压低 Habitat 加载场景的初始化输出，并丢掉缺少语义网格的告警。整次运行结束后在实验目录写 `brief_summary.txt`：成功率、到目标距离、路径效率均值，以及总耗时与每集平均用时。

## 规划器上下文压缩与 Stop 闸门（2026-09-21）

Planner 每圈只送两张图：六向拼成一张后缩放到 960×480，俯视图同比缩小且不超过 960×480。Summary 只吃 Observe、短工具结果、终态与执行，不再塞完整对话。`history` 只保留节点摘要字符串。分割放松三档：0.5 → 0.4 → 0.3。跟随最多 3 段，到不到子目标都停。单测：`python test/harness/test_ctx_compress.py`。

## 去掉 Stop 深度闸门（2026-09-21）

规划器输出 `Stop` 即发仿真停止并结束本集，不再要求本圈测深小于 0.9 米，也不再因深度不足拒绝后中止。

## 丢掉占用图走不通的探索点（2026-09-21）

`extract_frontiers` 从机身搜不到有限路径则丢弃该簇，不再写成 `1000+欧氏`。跟随只在有限测地距离的点里取最远；路径已为无穷则不调用 `pursue_occupancy`。跟随第一步 A* 失败立即返回，避免每步打满搜索。单测：`python test/occupancy/test_frontier_reach.py`。

