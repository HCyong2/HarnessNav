# M3 通讯协议

JSON 一律 `snake_case`。`pano_id ∈ {0,2,4,6,8,10}`。`xyz=[x,y,z]` 米。`uv=[u,v]`。`yaw` / `world_yaw` 弧度。图不进 JSON。P0 脚本与 P1 VLM 共用字段。评测 `success_distance_m=1.0`。

## PlannerIn

附件固定 7 张：`Direction {0,2,4,6,8,10}` + `Topdown`。Look/Recall 再追加。

```json
{
  "goal": "toilet",
  "state": "Unseen",
  "current_node_id": 3,
  "allowed_tools": ["Depth", "Look", "Recall"],
  "allowed_actions": ["MakePlan", "TraceBack"],
  "views": [
    {"pano_id": 0, "goal_find": false, "landmark": "kitchen", "room_type": "kitchen", "unexplored": false},
    {"pano_id": 2, "goal_find": true, "landmark": "plant in doorway", "room_type": "hallway", "unexplored": true}
  ],
  "history": [
    {
      "node_id": 0,
      "visit_count": 1,
      "views": [
        {"pano_id": 0, "goal_find": false, "landmark": "sofa", "room_type": "living room", "unexplored": true}
      ],
      "summary": "Observed Direction 10 as a living room that may contain a plant; advanced with semantic exploration in that heading."
    },
    {
      "node_id": 3,
      "visit_count": 1,
      "summary": ""
    }
  ]
}
```

`state`：`Unseen | Find | Confirmed | Arrived | Blocked`。每拍先 Observe 六向 `goal_find` / `landmark` / `room_type`；Harness 用 `goal_find` 置 Find。`unexplored` 由占用图 leftover 扇区写入；Planner 对本节点发出带 `pano_id` 的 MakePlan/Verify/Locate 后该向粘性为 false。当前节点的 views 只出现在顶栏，history 里同 `node_id` 不再重复贴 views。回合内不把 `success_distance_m`、habitat metric、hint、leftover 坐标喂给 Planner。Blocked 时另有 `blocked_type` / `traceback_node_ids` / `locate_count`。

## Observe

```json
{"views":[{"pano_id":0,"goal_find":false,"landmark":"corridor","room_type":"hallway"}]}
```

六向必须齐全。VLM 解析失败最多重试 5 次，用尽则本集 `aborted`。

## Planner 输出（每次恰好一个对象）

```json
{"action": "Depth", "pano_id": 4, "object": "chair", "instance_id": null}
{"action": "Look", "look": "down"}
{"action": "Recall", "node_id": 2, "pano_id": 4, "query": "where was the door"}
{"action": "MakePlan", "pano_id": 4, "mode": "semantic", "object_query": "sofa", "plan": "Approach the sofa in this room."}
{"action": "TraceBack", "node_id": 2}
{"action": "Verify", "pano_id": 4}
{"action": "Locate", "pano_id": 4}
```

`mode=frontier` ⇒ `object_query=null`；`mode=semantic` ⇒ `object_query` 非空，且不得是门、门口、走廊、地面、墙等通道说法。不在 `allowed_*` 中的 `action` 为 `planner_violation`。语义规划对准后第一帧分割仍空时，本圈把失败说明写回规划器再要一次终态，不重新环视，最多回退 2 次。Planner 不发 Stop。

## Depth / Look / Recall / Verify 输出

```json
{"ok": true, "instances": [{"id": "chair_1", "uv": [120, 200], "depth_m": 2.4, "geodesic_m": 2.8, "score": 0.41}]}
{"ok": true, "action": "down", "image_label": "Look down"}
{"ok": false, "error": "pitch_limit"}
{"ok": true, "node_id": 2, "query": "...", "image_labels": ["Recall node=2 dir=4"], "node_public": {}}
{"ok": true, "consistency": true, "vlm_same": true, "pano_id": 4}
```

`Look.action`：`up|down|left|right`。

## MoverIn + Mover 输出 + MoverReport

```json
{
  "goal": "toilet",
  "mode": "semantic",
  "object_query": "sofa",
  "plan": "Approach the sofa in this room.",
  "near_m": 0.5,
  "leg_index": 2,
  "candidates": [{"id": "sofa_1", "uv": [200, 180], "depth_m": 2.1, "score": 0.44}]
}
```

```json
{"id": "sofa_1"}
```

```json
{
  "status": "miss",
  "mode": "semantic",
  "object_query": "sofa",
  "chosen_ids": ["sofa_1"],
  "legs": 2,
  "dist_moved_m": 3.4,
  "last_goal_xyz": [1.1, 0.88, -2.0]
}
```

`status`：`ok | miss | blocked | arrived_subgoal`。

## 记忆

Frontier：`{"fid":"F0","xyz":[1.2,0.88,-3.4],"world_yaw":1.52,"geodesic_m":3.1}`。画面内每腿重编 `F1…`。

Node：`node_id` 只增不改号；`pano` 键为 `"0"…"10"` 的 `{rgb_path, depth_path}`；内部 `leftover_frontiers` / `explored_dirs`；`views`；`summary`（Summary VLM 覆盖写入）。

Edge：`{"src":1,"dst":2,"geodesic_m":4.8,"visits":1}`。

History：`node_id`、`visit_count`、`summary`；远程节点另带上次 `views`（含 `unexplored`）。无 `is_current`。

## 状态机门控

| 状态 | allowed_tools | allowed_actions |
|---|---|---|
| Unseen | Depth Look Recall | MakePlan TraceBack |
| Find | 无 | Verify |
| Confirmed | Depth Look Recall | MakePlan TraceBack Locate |
| Arrived | 无 | 无 |
| Blocked type1 | Look Depth | MakePlan TraceBack |
| Blocked type2 | Look Depth | MakePlan TraceBack Locate |

Planner 不发 Stop。Verify 仅 `pano_id`；Locate 仅 `pano_id`。type2 时 PlannerIn 含 `traceback_node_ids`。
