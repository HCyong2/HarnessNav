"""Harness 主循环：ScanNode、Planner、Mover、建图。"""

import json
import math
import os
import shutil
import time

import numpy as np

from harness.memory import NODE_MATCH_M, NodeGraph
from harness.overlay import (depth_text_map, draw_annotated, frontier_candidates,
                             path_geodesic_ok, pick_mover_candidate,
                             semantic_candidates)
from harness.protocol import (BLOCKED_DIST_M, MAX_LOCATE_ATTEMPTS, MAX_LOCATE_LEGS,
                             MAX_MOVER_LEGS, MAX_MOVER_RETRIES, PANO_IDS, dump_json,
                             jsonable, make_mover_miss_caption, make_seg_retry_caption,
                             validate_planner_action)
from harness.skills import restore_pitch, run_depth, run_look, run_recall
from harness.state import NavState, allowed_for, needs_bev
from harness.topdown_rec import TopdownRecorder, attach_step_capture
from nav.goto import (STUCK_EPS_M, body_position, body_yaw_env, face_pano,
                      foothold_from_hit, occupancy_path_length, pixel_to_world,
                      pursue_occupancy, turn_to_yaw)
from nav.occupancy import (EVEN_PANO_INDICES, HAB_STOP, NEAR_FRONTIER_GEO_M,
                           OccupancyMap, VLM_BEV_MAX_WH, VLM_PANO_WH, clean_depth,
                           concat_panorama, fit_within, frontiers_in_dir,
                           planner_sector_dirs, resize_exact, save_rgb, sensor_pose,
                           to_rgb_uint8)
from nav.transform import habitat_camera_intrinsic
from perception.base import empty_result, mask_center_pixel, segment_relax
from vlm.log import RunLog
from vlm.planner import READONLY
from vlm.retry import VlmRetryExhausted, retry_call
from vlm.summary import compress_bundle, summarize
from vlm.verify_views import vlm_verify_pair

MAX_SKILLS = 3
# 前沿点与语义共用：选点前「已近」闸门、跟随后到达、pursue 停步半径。
SUBGOAL_NEAR_M = 0.35
FRONTIER_LEG_STEPS = 30
SEMANTIC_LEG_STEPS = 8
TOOL_NAMES = ("Depth", "Look", "Recall", "Verify", "MakePlan", "TraceBack",
              "Locate", "Stop")


def _mover_exec_fail(report):
    """一步未走的 Mover 执行失败（看不见候选 / 语义空）。"""
    if not isinstance(report, dict):
        return False
    st = report.get("status")
    legs = int(report.get("legs") or 0)
    return st in ("seg_empty", "miss", "lost") and legs == 0


class EpisodeAbort(Exception):
    """本集因 VLM 失败提前结束。"""

    def __init__(self, reason):
        """记录原因。

        Args:
            reason (str): 失败说明。
        """
        self.reason = str(reason)
        super().__init__(self.reason)


def _short_json(obj, limit=400):
    """一行 JSON 摘要。"""
    text = json.dumps(jsonable(obj), ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


class Harness:
    """一次 episode 的编排器。"""

    def __init__(self, env, config, out_dir, goal=None, backend=None,
                 success_distance_m=1.0, glee_threshold=0.2,
                 planner=None, mover=None, log=None, debug=True):
        """初始化。

        Args:
            env: ``habitat.Env``，调用方已 ``reset``。
            config: habitat 配置。
            out_dir (str): 本集可视化目录。
            goal (str, optional): 目标类。
            backend: 分割后端。
            success_distance_m (float): 到达半径；停止闸门与 Depth 共用。
            glee_threshold (float): 旁路阈值（后端自身决定）。
            planner: ``VlmPlanner``。
            mover: ``VlmMover``。
            log (RunLog, optional): 终端/debug 流水。
            debug (bool): 为真时落盘图片、planner.txt、topdown；否则只写 episode.json。
        """
        self.env = env
        self.config = config
        self.out_dir = out_dir
        self.debug = bool(debug)
        self.backend = backend
        self.success_distance_m = float(success_distance_m)
        self.intrinsic = habitat_camera_intrinsic(config)
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        spec = sensors.depth_sensor
        self.min_depth = float(spec.min_depth)
        self.max_depth = float(spec.max_depth)
        self.occ = OccupancyMap.from_config(config)
        self.graph = NodeGraph()
        self.planner = planner
        self.mover = mover
        self.log = log if log is not None else RunLog()
        self.state = NavState.UNSEEN
        self._mover_retries_used = 0
        self.pitch_steps = 0
        self.current_id = None
        self.last_node_id = None
        self.steps_used = 0
        self.max_steps = int(config.habitat.environment.max_episode_steps)
        self.goal = goal or str(getattr(env.current_episode, "object_category", "chair"))
        self.abort_reason = None
        self.scan_count = 0
        self.verify_count = 0
        self.look_count = 0
        self.confirmed_xyz = None
        self.stop_issued = False
        self.blocked_from = None
        self.confirmed_node_floor = None
        self.locate_count = 0
        self.last_sector_dirs = set()
        self.tool_counts = {k: 0 for k in TOOL_NAMES}
        self.topdown = None
        self._move_ticks = 0
        self._scan_tool_log = []
        self.vlm_tmp = os.path.join(out_dir, ".vlm")
        self.planner_log_path = os.path.join(out_dir, "planner.txt")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self.vlm_tmp, exist_ok=True)
        if self.debug:
            with open(self.planner_log_path, "w", encoding="utf-8") as f:
                f.write("")

    def _save_debug_rgb(self, path, image):
        """仅 debug 模式写可视化 RGB。"""
        if self.debug:
            save_rgb(path, image)

    def _slog(self, component, message):
        """打流水；本圈扫描开始后带圈号。"""
        if component != "Harness" and int(self.scan_count) > 0:
            message = f"loop{self.scan_count} {message}"
        self.log.emit(component, message)

    def _abort(self, reason, node_info=None):
        """VLM 用尽重试，提前结束本集。"""
        del node_info
        self.abort_reason = str(reason)
        self._slog("Harness", reason)
        try:
            if not getattr(self, "_scan_logged", False):
                self._append_planner_scan(
                    getattr(self, "_scan_tool_log", []), None,
                    final_reasoning=self._planner_reason())
                self._scan_logged = True
        except Exception:
            pass
        raise EpisodeAbort(reason)

    def _vlm_try(self, fn, label):
        """VLM 调用最多 ``VLM_MAX_TRIES`` 次。"""
        try:
            return retry_call(fn, label)
        except VlmRetryExhausted as exc:
            self._abort(str(exc))

    def _planner_in_text(self):
        """本拍 PlannerIn JSON。"""
        payload = getattr(self.planner, "payload", None)
        if payload is None:
            return "(empty)"
        return json.dumps(jsonable(payload), ensure_ascii=False, indent=2)

    def _planner_reason(self):
        """上一轮 Planner 回复里的 reasoning 正文。"""
        getter = getattr(self.planner, "last_reasoning", None)
        if callable(getter):
            return getter() or ""
        return ""

    def _append_planner_scan(self, tool_log, final_action, final_reasoning=None):
        """把本拍 Planner 对话追加到 ``planner.txt``。

        PlannerIn 已含 ``views``，不再另写 Observe 块。
        """
        blocks = [f"Scan {self.scan_count}", "PlannerIn",
                  self._planner_in_text(), ""]
        for i, item in enumerate(tool_log or [], 1):
            name = item.get("action") or ""
            args = {k: v for k, v in (item.get("args") or {}).items()
                    if k != "reasoning"}
            blocks.append(f"Planner Tool Call {i}")
            reason = (item.get("reasoning") or "").strip()
            if reason:
                blocks.append("reasoning:")
                blocks.append(reason)
            blocks.append(f"{name} {json.dumps(jsonable(args), ensure_ascii=False)}")
            blocks.append(json.dumps(jsonable(item.get("result")), ensure_ascii=False))
            blocks.append("")
        blocks.append("Planner Final Action")
        reason = (final_reasoning or "").strip()
        if reason:
            blocks.append("reasoning:")
            blocks.append(reason)
        if final_action is None:
            blocks.append("(none)")
        else:
            action = dict(final_action)
            action.pop("reasoning", None)
            blocks.append(json.dumps(jsonable(action), ensure_ascii=False))
        blocks.append("")
        if not self.debug:
            return
        with open(self.planner_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(blocks).rstrip() + "\n\n")

    def _append_planner_retry(self, caption, new_action=None, reasoning=None):
        """把本圈分割失败回退写入 ``planner.txt``。"""
        if not self.debug:
            return
        blocks = ["Retry", caption, ""]
        reason = (reasoning or "").strip()
        if reason:
            blocks.append("reasoning:")
            blocks.append(reason)
            blocks.append("")
        if new_action is not None:
            blocks.append("Planner Final Action")
            action = dict(new_action)
            action.pop("reasoning", None)
            blocks.append(json.dumps(jsonable(action), ensure_ascii=False))
            blocks.append("")
        with open(self.planner_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(blocks).rstrip() + "\n\n")

    def _mover_retry_caption(self, plan, report):
        """按失败类型生成写回规划器的说明。"""
        if (report or {}).get("status") == "seg_empty":
            return make_seg_retry_caption(plan)
        return make_mover_miss_caption(plan, report)

    def _append_verify_log(self, slim, view_paths):
        """把 Verify 结果追加到 ``planner.txt``。"""
        if not self.debug:
            return
        body = dict(slim)
        body["views"] = [os.path.basename(p) for p in view_paths]
        text = (
            f"Verify {self.verify_count}\n"
            + json.dumps(jsonable(body), ensure_ascii=False, indent=2)
            + "\n\n"
        )
        with open(self.planner_log_path, "a", encoding="utf-8") as f:
            f.write(text)

    def _canon_verify_action(self, action):
        """去掉 Verify 多余字段，只保留 pano_id。"""
        if not isinstance(action, dict) or action.get("action") != "Verify":
            return action
        action = dict(action)
        action.pop("instance_id", None)
        return action

    def _allowed(self):
        """当前状态对应的工具与终态白名单。"""
        return allowed_for(self.state, blocked_from=self.blocked_from)

    def _enter_confirmed(self):
        """进入 Confirmed。"""
        self.state = NavState.CONFIRMED
        self.blocked_from = None
        if self.confirmed_node_floor is None and self.current_id is not None:
            self.confirmed_node_floor = int(self.current_id)

    def _apply_motion_outcome(self, dist_moved_m, source_action, allow_enter_blocked=True):
        """按位移更新 Blocked 进出；Verify 不得进入 Blocked。

        Args:
            dist_moved_m (float): 本段位移。
            source_action (str): ``MakePlan`` / ``Locate`` / ``TraceBack``。
            allow_enter_blocked (bool): 为假时只允许脱困、不新进 Blocked。
        """
        dist = float(dist_moved_m or 0.0)
        if self.state == NavState.BLOCKED:
            if dist >= BLOCKED_DIST_M - 1e-9:
                if self.blocked_from == "Confirmed":
                    self._enter_confirmed()
                    self._slog("Harness", "Blocked type2 脱困 → Confirmed")
                else:
                    self.state = NavState.UNSEEN
                    self.blocked_from = None
                    self._slog("Harness", "Blocked type1 脱困 → Unseen")
            return
        if not allow_enter_blocked:
            return
        if source_action not in ("MakePlan", "Locate"):
            return
        if dist >= BLOCKED_DIST_M - 1e-9:
            return
        if self.state == NavState.UNSEEN:
            self.blocked_from = "Unseen"
            self.state = NavState.BLOCKED
            self._slog("Harness", f"Unseen → Blocked type1 dist={dist:.3f}")
        elif self.state == NavState.CONFIRMED:
            self.blocked_from = "Confirmed"
            self.state = NavState.BLOCKED
            self._slog("Harness", f"Confirmed → Blocked type2 dist={dist:.3f}")

    def _count(self, name):
        """工具调用 +1。"""
        if name in self.tool_counts:
            self.tool_counts[name] += 1

    def _obs(self):
        """当前 RGB 与清洗深度。"""
        obs = self.env.sim.get_sensor_observations()
        return to_rgb_uint8(obs["rgb"]), clean_depth(obs["depth"], self.min_depth, self.max_depth)

    def _segment(self, rgb, text):
        """分割并按 0.5→0.4→0.3 放松阈值后 NMS。

        Returns:
            tuple: ``(SegResult, 采用的阈值)``。
        """
        if self.backend is None or not text:
            h, w = rgb.shape[:2]
            return empty_result(height=h, width=w), 0.0
        result, thr = segment_relax(self.backend, rgb, text)
        name = getattr(self.backend, "name", "seg")
        self._slog("Seg", f"{name} thr={thr:.3f} n={len(result)}")
        return result, float(thr)

    def scan_node(self, look_down_floor=True):
        """环视建图并落盘本拍全景 / BEV。"""
        self.pitch_steps = restore_pitch(self.env, self.pitch_steps)
        yaw = body_yaw_env(self.env)
        images = self.occ.scan_around(self.env, rotate_times=12,
                                      look_down_floor=True, mark_node=False)
        xyz = body_position(self.env)
        pano_rgbs = {}
        for pid in PANO_IDS:
            if pid >= len(images):
                continue
            pano_rgbs[pid] = images[pid]
        frontiers = self.occ.extract_frontiers(
            agent_xyz=xyz, pf=self.env.sim.pathfinder, node_yaw=yaw)
        sector_dirs = planner_sector_dirs(frontiers, xyz, yaw, self.occ)
        self.last_sector_dirs = set(sector_dirs)
        nid, revisit = self.graph.upsert_scan(
            xyz, yaw, pano_rgbs, {}, sector_dirs, last_plan=None, summary=None)
        if self.last_node_id is not None:
            self.graph.add_edge(self.last_node_id, nid, self.env, occ=self.occ)
        self.current_id = nid
        self.last_node_id = nid
        self.scan_count += 1
        bev_full = self._render_bev(frontiers=frontiers)
        pano_full = concat_panorama(images, EVEN_PANO_INDICES)
        self._save_debug_rgb(
            os.path.join(self.out_dir, f"bev_{self.scan_count}.png"), bev_full)
        self._save_debug_rgb(
            os.path.join(self.out_dir, f"pano_{self.scan_count}.png"), pano_full)
        if self.debug:
            occ_dbg = self.occ.render_occ_debug(xyz, frontiers=frontiers)
            self._save_debug_rgb(
                os.path.join(self.out_dir, f"Occ_{self.scan_count}.png"), occ_dbg)
        pano_vlm = os.path.join(self.vlm_tmp, f"scan{self.scan_count}_pano.jpg")
        bev_vlm = os.path.join(self.vlm_tmp, f"scan{self.scan_count}_bev.jpg")
        save_rgb(pano_vlm, resize_exact(pano_full, VLM_PANO_WH[0], VLM_PANO_WH[1]))
        save_rgb(bev_vlm, fit_within(bev_full, VLM_BEV_MAX_WH[0], VLM_BEV_MAX_WH[1]))
        self._slog(
            "ScanNode",
            f"node={nid} revisit={revisit} sector_dirs={sorted(sector_dirs)} "
            f"state={self.state.value}")
        return {
            "node_id": nid, "revisit": revisit, "images": images, "yaw": yaw, "xyz": xyz,
            "frontiers": frontiers, "sector_dirs": sorted(sector_dirs), "views": [],
            "scan_dir": self.vlm_tmp, "bev_path": bev_vlm, "pano_vlm": pano_vlm,
        }

    def _packup(self, node_info, tool_log=None):
        """组装瘦身 PlannerIn。"""
        del tool_log
        return {
            "goal": self.goal,
            "state": self.state.value,
            "current_node_id": node_info["node_id"],
            "views": node_info.get("views") or [],
            "history": self.graph.history(node_info["node_id"]),
        }

    def _attach_unexplored(self, views, node_info):
        """把近距路径扇区与已选朝向合成 ``unexplored``。"""
        sector_dirs = {int(x) for x in (node_info.get("sector_dirs") or [])}
        node = self.graph.nodes[node_info["node_id"]]
        explored = {int(x) for x in (node.get("explored_dirs") or [])}
        for v in views:
            pid = int(v["pano_id"])
            v["unexplored"] = (pid in sector_dirs) and (pid not in explored)
        return views

    def apply_skill(self, action, node_info):
        """执行 Depth / Look / Recall。

        Returns:
            tuple: ``(result_dict, extra_image_paths)``。
        """
        name = action.get("action")
        extra = []
        if name in ("Depth", "Look", "Recall"):
            self._count(name)
        if name == "Depth":
            pid = int(action.get("pano_id") or 0)
            query = action.get("object") or self.goal
            face_pano(self.env, pid, node_yaw=node_info["yaw"])
            out = run_depth(self.env, self.backend, query, action.get("instance_id"),
                            self.min_depth, self.max_depth,
                            occ=self.occ, intrinsic=self.intrinsic)
            out["pano_id"] = pid
            if self._is_goal_query(query):
                for inst in out.get("instances") or []:
                    fh = inst.get("foothold_xyz")
                    if fh is not None:
                        self.confirmed_xyz = list(fh)
                        break
            # 回写 Planner 只留测地距离
            slim_inst = []
            for inst in out.get("instances") or []:
                if not isinstance(inst, dict):
                    continue
                slim_inst.append({
                    "id": inst.get("id"),
                    "geodesic_m": inst.get("geodesic_m"),
                })
            slim_out = {"ok": bool(out.get("ok", True)), "pano_id": pid,
                        "instances": slim_inst}
            if out.get("error") is not None:
                slim_out["error"] = out.get("error")
            return slim_out, extra
        if name == "Look":
            look = action.get("look") or "down"
            result, self.pitch_steps = run_look(self.env, look, self.pitch_steps)
            rgb, _ = self._obs()
            self.look_count += 1
            self._save_debug_rgb(
                os.path.join(self.out_dir, f"look{self.look_count}.png"), rgb)
            # 临时目录供本圈 VLM 附加图（回合结束会删）
            tmp = os.path.join(self.vlm_tmp, f"scan{self.scan_count}_look_{look}.png")
            save_rgb(tmp, rgb)
            extra.append(tmp)
            result["image_path"] = tmp
            return result, extra
        if name == "Recall":
            out = run_recall(self.graph, int(action.get("node_id") or 0),
                             pano_id=action.get("pano_id"), query=action.get("query") or "")
            images = out.pop("images", None) or []
            for i, img in enumerate(images):
                if img is None:
                    continue
                path = os.path.join(self.vlm_tmp, f"scan{self.scan_count}_recall_{i}.png")
                save_rgb(path, img)
                extra.append(path)
            out["image_paths"] = extra
            return out, extra
        return {"ok": False, "error": "unknown_skill"}, extra

    def _apply_observe(self, views, node_info):
        """用六向 goal_find 更新 Find，并写入节点。"""
        views = self._attach_unexplored(views, node_info)
        node_info["views"] = views
        found = any(bool(v.get("goal_find")) for v in views)
        # Blocked / Confirmed / Arrived 不因 Observe 改写；type1 脱困后已是 Unseen
        if self.state not in (NavState.CONFIRMED, NavState.ARRIVED, NavState.BLOCKED):
            if found:
                self.state = NavState.FIND
            elif self.state == NavState.FIND:
                self.state = NavState.UNSEEN
        node = self.graph.nodes[node_info["node_id"]]
        node["views"] = views
        found_ids = [v["pano_id"] for v in views if v.get("goal_find")]
        self._slog("Observe", f"goal_find={found_ids} state={self.state.value}")

    def _mark_selected_pano(self, action, node_info):
        """MakePlan / Verify / Locate 选定的朝向粘性标已探索。"""
        name = action.get("action")
        if name not in ("MakePlan", "Verify", "Locate"):
            return
        if action.get("pano_id") is None:
            return
        pid = int(action["pano_id"])
        self.graph.mark_explored(node_info["node_id"], pid)
        for v in node_info.get("views") or []:
            if int(v.get("pano_id")) == pid:
                v["unexplored"] = False

    def _propose_ok(self, tools, actions, require_verify):
        """带校验的一轮 propose。"""
        def once():
            action = self.planner.propose()
            if not isinstance(action, dict) or "action" not in action:
                raise ValueError("Planner 未给出合法 action")
            err = validate_planner_action(action, tools, actions)
            if err:
                raise ValueError(err)
            name = action.get("action")
            if require_verify and name not in READONLY and name != "Verify":
                raise ValueError("Find 且未 Confirmed，终态必须是 Verify")
            self.planner.commit_pending()
            return action

        return self._vlm_try(once, "Planner propose")

    def _planner_until_terminal(self, node_info, tools, actions, require_verify):
        """在已有对话上继续，直到给出终态。

        Args:
            node_info (dict): 本圈节点。
            tools (list): 允许的只读工具。
            actions (list): 允许的终态。
            require_verify (bool): Find 且未核实则终态必须核实。

        Returns:
            dict: 终态动作。
        """
        tool_log = self._scan_tool_log
        n_skill = int(getattr(self, "_scan_n_skill", 0) or 0)
        extra_imgs = int(getattr(self, "_scan_extra_imgs", 0) or 0)
        last_action = None
        final_reasoning = ""
        budget = MAX_SKILLS + 1
        for _ in range(budget):
            action = self._propose_ok(tools, actions, require_verify)
            name = action["action"]
            if name in READONLY:
                reason = self._planner_reason()
                if n_skill >= MAX_SKILLS:
                    self._abort("Planner 只读工具次数用尽仍未给出终态", node_info)
                if name == "Recall" and extra_imgs >= 2:
                    result = {"ok": False, "error": "image_quota"}
                    try:
                        self.planner.feed_skill(action, result, [])
                    except Exception as exc:
                        self._abort(f"VLM 回写 Skill 失败: {exc}", node_info)
                    n_skill += 1
                    tool_log.append({
                        "action": name,
                        "args": {k: v for k, v in action.items() if k != "action"},
                        "result": result,
                        "reasoning": reason,
                    })
                    continue

                self._slog("Planner", f"调用 {name} {_short_json({k: v for k, v in action.items() if k != 'action'})}")
                result, extra = self.apply_skill(action, node_info)
                extra_imgs += len(extra)
                n_skill += 1
                slim = {k: v for k, v in result.items() if k != "views"}
                self._slog(name, f"返回 {_short_json(slim)}")
                tool_log.append({
                    "action": name,
                    "args": {k: v for k, v in action.items() if k != "action"},
                    "result": slim,
                    "reasoning": reason,
                })
                try:
                    self.planner.feed_skill(action, slim, extra)
                except Exception as exc:
                    self._abort(f"VLM 回写 Skill 失败: {exc}", node_info)
                last_action = action
                continue
            last_action = self._canon_verify_action(action)
            final_reasoning = self._planner_reason()
            self._slog("Planner", _short_json(last_action))
            break
        else:
            self._abort("Planner 未给出终态", node_info)

        self._scan_n_skill = n_skill
        self._scan_extra_imgs = extra_imgs
        self._scan_final_reasoning = final_reasoning
        self.pitch_steps = restore_pitch(self.env, self.pitch_steps)
        turn_to_yaw(self.env, float(node_info["yaw"]))
        return self._canon_verify_action(last_action)

    def planner_step(self, node_info):
        """Observe 后最多 3 次只读工具，再给出终态 action。"""
        if self.planner is None:
            self._abort("未配置 VLM Planner")
        tool_log = []
        self._scan_tool_log = tool_log
        self._scan_logged = False
        self._scan_final_reasoning = ""
        self._scan_n_skill = 0
        self._scan_extra_imgs = 0
        self._mover_retries_used = 0
        tools0, actions0 = self._allowed()
        payload = self._packup(node_info, tool_log)
        try:
            self.planner.start(
                payload, node_info["pano_vlm"],
                tools=tools0, actions=actions0,
                blocked_from=self.blocked_from)
        except Exception as exc:
            self._abort(f"VLM 组装对话失败: {exc}", node_info)

        views = self._vlm_try(self.planner.observe, "Observe")
        self._apply_observe(views, node_info)
        tools, actions = self._allowed()
        self.planner.set_allowed(
            tools, actions, state=self.state, blocked_from=self.blocked_from)
        if needs_bev(self.state, blocked_from=self.blocked_from):
            try:
                self.planner.feed_bev(node_info.get("bev_path"))
            except Exception as exc:
                self._abort(f"VLM 追加俯视图失败: {exc}", node_info)
        payload = self._packup(node_info, tool_log)
        self.planner.feed_observe(views, payload)
        require_verify = self.state == NavState.FIND

        last_action = self._planner_until_terminal(
            node_info, tools, actions, require_verify)
        self._append_planner_scan(
            tool_log, last_action,
            final_reasoning=self._scan_final_reasoning)
        self._scan_logged = True
        return last_action, None

    def do_traceback(self, node_id):
        """直达旧节点，失败则沿边分段走。"""
        target = self.graph.nodes.get(int(node_id))
        if target is None or int(node_id) == self.current_id:
            return {"status": "reject", "node_id": node_id}
        start = body_position(self.env)
        max_geo = min(15.0, 0.25 * max(self.max_steps - self.steps_used, 1))
        geo = occupancy_path_length(self.occ, start, target["xyz"])
        if not math.isfinite(geo) or geo > max_geo:
            return {"status": "reject", "node_id": node_id, "reason": "too_far"}
        via_graph = False
        dist = 0.0
        out = pursue_occupancy(self.env, self.occ, target["xyz"], max_steps=80,
                               on_step=self._on_move, success_dist=NODE_MATCH_M)
        dist += out["dist_moved_m"]
        self.steps_used += out["steps"]
        arrived = float(np.hypot(body_position(self.env)[0] - target["xyz"][0],
                                 body_position(self.env)[2] - target["xyz"][2])) < NODE_MATCH_M
        if not arrived:
            via_graph = True
            path = self.graph.graph_path(self.current_id, int(node_id))
            for nid in path:
                waypoint = self.graph.nodes[nid]["xyz"]
                out = pursue_occupancy(self.env, self.occ, waypoint, max_steps=40,
                                       on_step=self._on_move, success_dist=NODE_MATCH_M)
                dist += out["dist_moved_m"]
                self.steps_used += out["steps"]
        turn_to_yaw(self.env, float(target["yaw"]))
        arrived = float(np.hypot(body_position(self.env)[0] - target["xyz"][0],
                                 body_position(self.env)[2] - target["xyz"][2])) < NODE_MATCH_M
        return {"status": "ok" if arrived else "miss", "node_id": int(node_id),
                "revisit": True, "via_graph": via_graph, "dist_moved_m": dist}

    def _render_bev(self, frontiers=None):
        """画当前占用图 BEV。"""
        xyz = body_position(self.env)
        if frontiers is None:
            frontiers = self.occ.extract_frontiers(agent_xyz=xyz, pf=self.env.sim.pathfinder)
        nodes, _edges = self.graph.overlays()
        return self.occ.render_bev_from_env(
            self.env, sector_labels=EVEN_PANO_INDICES, rotate_times=12,
            frontiers=frontiers, node_overlays=nodes)

    def _on_move(self, env):
        """途中融合。"""
        self.occ.record_body(body_position(env))
        self._move_ticks += 1
        if self._move_ticks % 5 == 0:
            self.occ.integrate_from_env(env)

    def _subgoal_dist_m(self, body, xyz):
        """机身到子目标的到达距离：占用图测地优先，不通则水平欧氏。

        Args:
            body: 机身位置。
            xyz: 子目标世界坐标。

        Returns:
            tuple: ``(dist_m, how)``，``how`` 为 ``geodesic`` 或 ``euclid``。
        """
        goal = np.asarray(xyz, dtype=np.float64).reshape(3)
        geo = occupancy_path_length(self.occ, body, goal)
        if path_geodesic_ok(geo):
            return float(geo), "geodesic"
        euc = float(np.hypot(goal[0] - body[0], goal[2] - body[2]))
        return euc, "euclid"

    def _candidate_stand_xyz(self, cand, depth, sensor_pos, sensor_rot, body):
        """候选可走落脚点；已有 ``xyz`` 则直接用，否则反投影再吸附。

        Args:
            cand (dict): 候选。
            depth (np.ndarray): 深度图。
            sensor_pos: 相机位置。
            sensor_rot: 相机旋转。
            body: 机身位置。

        Returns:
            np.ndarray | None: 形状 ``(3,)``。
        """
        if cand.get("xyz") is not None:
            return np.asarray(cand["xyz"], dtype=np.float64).reshape(3)
        uv = cand.get("uv")
        if uv is None or depth is None:
            return None
        world = pixel_to_world(
            uv[0], uv[1], depth, self.intrinsic, sensor_pos, sensor_rot)
        if world is None:
            return None
        stand = foothold_from_hit(self.occ, sensor_pos, world, body)
        if stand is None:
            return None
        return np.asarray(stand, dtype=np.float64).reshape(3)

    def run_mover(self, plan, frontiers, max_legs=None, stop_on_geodesic=False):
        """对准规划朝向后选点并多段逼近；探索与语义均在首次选点后锁定世界坐标。

        前沿点与语义共用 ``SUBGOAL_NEAR_M``：选点前已近、跟随后到达均按占用图测地
        （不通时退回水平欧氏）。

        Args:
            plan (dict): 含 ``pano_id`` / ``mode`` / ``object_query``。
            frontiers (list): 探索点。
            max_legs (int, optional): 腿数上限；缺省 MakePlan 用 3。
            stop_on_geodesic (bool): 每腿后若测地达标则程序发 Stop（Locate）。

        Returns:
            dict: mover 报告；不直接改 Blocked/Miss 状态机。
        """
        node = self.graph.nodes[self.current_id]
        node_yaw = float(node["yaw"])
        pano_id = int(plan["pano_id"])
        face_pano(self.env, pano_id, node_yaw=node_yaw)
        mode = plan["mode"]
        query = plan.get("object_query")
        chosen, legs, dist = [], 0, 0.0
        last_xyz = None
        status = "ok"
        if max_legs is None:
            max_legs = MAX_MOVER_LEGS
        tag = "object" if mode == "semantic" else "frontier"
        locked_xyz = None
        locked_id = None
        while legs < max_legs and not self.env.episode_over:
            rgb, depth = self._obs()
            pos, rot = sensor_pose(self.env)
            h, w = rgb.shape[:2]
            seg_kept = None
            if mode == "semantic":
                result, thr = self._segment(rgb, query)
                cands, seg_kept = semantic_candidates(result, depth, query)
            else:
                sector = frontiers_in_dir(
                    frontiers, pano_id, node["xyz"], node_yaw, occ=self.occ)
                near_sector = [
                    fr for fr in sector
                    if path_geodesic_ok(fr.get("geodesic_m"))
                    and float(fr["geodesic_m"]) < NEAR_FRONTIER_GEO_M
                ]
                overlay = frontier_candidates(
                    near_sector, self.intrinsic, pos, rot, (h, w),
                    max_n=max(len(near_sector), 1), drop_occluded=False)
                cands = []
                for c in overlay:
                    item = {
                        "id": c["id"],
                        "xyz": c["xyz"],
                        "fid": c.get("fid"),
                        "geodesic_m": c.get("geodesic_m"),
                        "depth_m": c["depth_m"],
                        "score": 1.0,
                        "uv": c["uv"],
                    }
                    cands.append(item)
                if legs == 0:
                    self._slog(
                        "Mover",
                        f"探索候选 sector={len(sector)} near={len(near_sector)} "
                        f"drawn={len(cands)}")
            drawn = [c for c in cands if c.get("uv") is not None]
            stem = f"scan{self.scan_count}_leg{legs}_{tag}"
            in_path = os.path.join(self.vlm_tmp, f"{stem}_in.png")
            annotated = draw_annotated(
                rgb, mode, drawn, seg_kept,
                write_depth=(mode == "semantic"))
            save_rgb(in_path, annotated)
            self._save_debug_rgb(
                os.path.join(self.out_dir, f"{stem}_in.png"), annotated)
            if not cands and locked_xyz is None:
                if (mode == "semantic" and legs == 0
                        and int(getattr(self, "_mover_retries_used", 0) or 0) < MAX_MOVER_RETRIES
                        and hasattr(self.planner, "feed_plan_retry")
                        and not stop_on_geodesic):
                    retry_n = int(self._mover_retries_used) + 1
                    retry_path = os.path.join(
                        self.out_dir, f"scan{self.scan_count}_retry{retry_n}_in.png")
                    self._save_debug_rgb(retry_path, annotated)
                    status = "seg_empty"
                    self._slog("Mover", "无候选，准备本圈回退")
                    report = {
                        "status": status,
                        "mode": mode,
                        "object_query": query,
                        "chosen_ids": chosen,
                        "legs": legs,
                        "dist_moved_m": dist,
                        "last_goal_xyz": last_xyz,
                        "seg_thr": float(thr),
                        "seg_n": int(len(result)),
                    }
                    self._slog("Mover", f"pursue legs={legs} dist={dist:.2f} status={status}")
                    return report
                status = "lost" if stop_on_geodesic else "miss"
                self._slog("Mover", f"无候选 {status}")
                break
            body = body_position(self.env)
            if locked_xyz is not None:
                dist_m, how = self._subgoal_dist_m(body, locked_xyz)
                if dist_m <= SUBGOAL_NEAR_M:
                    status = "arrived_subgoal"
                    self._slog(
                        "Mover",
                        f"锁定子目标已在阈值内 {locked_id} how={how} "
                        f"dist={dist_m:.2f}")
                    break
                geo_lock = occupancy_path_length(self.occ, body, locked_xyz)
                cand = {
                    "id": locked_id or "locked",
                    "xyz": locked_xyz,
                    "geodesic_m": geo_lock if path_geodesic_ok(geo_lock) else None,
                }
            else:
                near = []
                for c in cands:
                    stand = self._candidate_stand_xyz(c, depth, pos, rot, body)
                    if stand is None:
                        continue
                    dist_m, _how = self._subgoal_dist_m(body, stand)
                    if dist_m <= SUBGOAL_NEAR_M:
                        near.append(c)
                if near:
                    status = "arrived_subgoal"
                    self._slog("Mover", f"子目标已在阈值内 {near[0]['id']}")
                    break
                cand = None
                if (mode == "frontier" and self.mover is not None
                        and hasattr(self.mover, "pick") and len(cands) > 1):
                    mover_in = {
                        "goal": self.goal,
                        "mode": mode,
                        "object_query": query,
                        "plan": plan.get("plan") or "",
                        "near_m": SUBGOAL_NEAR_M,
                        "leg_index": legs,
                        "candidates": [
                            {k: c[k] for k in ("id", "uv", "depth_m", "score", "geodesic_m")
                             if k in c}
                            for c in cands
                        ],
                        "depth_map": depth_text_map(cands),
                    }
                    try:
                        picked = self.mover.pick(mover_in, ego_path=in_path)
                        pid = picked.get("id")
                        cand = next((c for c in cands if c.get("id") == pid), None)
                        self._slog("Mover", f"VLM 选 {pid} n={len(cands)}")
                    except Exception as exc:
                        self._slog("Mover", f"VLM 选点失败，回退规则: {exc}")
                        cand = None
                if cand is None:
                    cand = pick_mover_candidate(mode, cands)
                if cand is None:
                    status = "lost" if stop_on_geodesic else "miss"
                    self._slog("Mover", f"无候选 {status}")
                    break
            chosen.append(cand["id"])
            if "xyz" in cand:
                world = np.asarray(cand["xyz"], dtype=np.float64)
                snapped = world
                info = {"geodesic": occupancy_path_length(self.occ, body_position(self.env), world),
                        "euclid": float(np.hypot(world[0] - body_position(self.env)[0],
                                                 world[2] - body_position(self.env)[2])),
                        "offset": 0.0}
            else:
                world = pixel_to_world(cand["uv"][0], cand["uv"][1], depth,
                                       self.intrinsic, pos, rot)
                if world is None:
                    status = "lost" if stop_on_geodesic else "miss"
                    break
                snapped = foothold_from_hit(self.occ, pos, world, body_position(self.env))
                info = {"geodesic": None, "euclid": None, "offset": None}
                if snapped is not None:
                    info["geodesic"] = occupancy_path_length(
                        self.occ, body_position(self.env), snapped)
                    info["euclid"] = float(np.hypot(
                        snapped[0] - body_position(self.env)[0],
                        snapped[2] - body_position(self.env)[2]))
                    info["offset"] = float(np.hypot(snapped[0] - world[0], snapped[2] - world[2]))
            if snapped is None:
                status = "lost" if stop_on_geodesic else "miss"
                break
            geo_now = info.get("geodesic")
            if mode == "frontier" and not path_geodesic_ok(geo_now):
                self._slog("Mover", f"不可达 {cand['id']} geodesic=inf 不跟随")
                status = "miss"
                break
            if locked_xyz is None:
                locked_xyz = np.asarray(snapped, dtype=np.float64).reshape(3).tolist()
                locked_id = cand.get("id")
            if mode == "frontier":
                geo_v = cand.get("geodesic_m")
                self._slog(
                    "Mover",
                    f"跟随 {cand['id']} geodesic="
                    f"{None if geo_v is None else round(float(geo_v), 2)} "
                    f"locked={locked_xyz is not None}")
            else:
                depth_v = cand.get("depth_m")
                self._slog(
                    "Mover",
                    f"跟随 {cand['id']} conf={float(cand.get('score') or 0):.3f} "
                    f"depth_m="
                    f"{None if depth_v is None else round(float(depth_v), 2)} "
                    f"score={float(cand.get('pick_score') or 0):.3f} "
                    f"locked={locked_xyz is not None}")
            last_xyz = np.asarray(snapped, dtype=np.float64).reshape(3).tolist()
            leg_steps = FRONTIER_LEG_STEPS if mode == "frontier" else SEMANTIC_LEG_STEPS
            out = pursue_occupancy(
                self.env, self.occ, snapped, max_steps=leg_steps,
                on_step=self._on_move, success_dist=SUBGOAL_NEAR_M)
            self.steps_used += out["steps"]
            dist += out["dist_moved_m"]
            geo = info.get("geodesic")
            euc = info.get("euclid")
            off = info.get("offset")
            self._slog(
                "Mover",
                f"snap offset={None if off is None else round(float(off), 2)} "
                f"geodesic={None if geo is None else round(float(geo), 2)} "
                f"euclid={None if euc is None else round(float(euc), 2)} "
                f"steps={out['steps']} dist={out['dist_moved_m']:.2f}")
            rgb_out, _ = self._obs()
            self._save_debug_rgb(
                os.path.join(self.out_dir, f"{stem}_out.png"), rgb_out)
            legs += 1
            if stop_on_geodesic:
                foothold = self._estimate_goal_foothold()
                if foothold is not None:
                    ok, dist_m, how = self._stop_geodesic_ok(foothold)
                    if ok:
                        self.confirmed_xyz = foothold.tolist()
                        self._issue_stop()
                        self.state = NavState.ARRIVED
                        self.blocked_from = None
                        status = "stopped"
                        self._slog("Locate", f"测地达标 how={how} dist={dist_m}")
                        break
            if out["blocked"]:
                status = "blocked"
                break
            if (not out["arrived"] and int(out.get("steps") or 0) == 0
                    and float(out.get("dist_moved_m") or 0.0) < STUCK_EPS_M):
                self._slog("Mover", "占用图无路径 miss")
                status = "miss"
                break
            arrive_m, how = self._subgoal_dist_m(body_position(self.env), snapped)
            if arrive_m <= SUBGOAL_NEAR_M:
                status = "arrived_subgoal"
                self._slog(
                    "Mover",
                    f"子目标测地到达 how={how} dist={arrive_m:.2f}")
                break
            if out["dist_moved_m"] < STUCK_EPS_M:
                status = "blocked"
                break
        report = {
            "status": status,
            "mode": mode,
            "object_query": query,
            "chosen_ids": chosen,
            "legs": legs,
            "dist_moved_m": dist,
            "last_goal_xyz": last_xyz,
        }
        self._slog("Mover", f"pursue legs={legs} dist={dist:.2f} status={status}")
        if status in ("miss", "lost") and legs == 0:
            turn_to_yaw(self.env, node_yaw)
        return report

    def _is_goal_query(self, query):
        """子目标是否就是本集 navigation object。"""
        return bool(query) and str(query).strip().lower() == self.goal.strip().lower()

    def _issue_stop(self):
        """发 HAB_STOP 以结算 Success。"""
        if self.env.episode_over:
            self._slog("Harness", "本集已结束，不再发 HAB_STOP")
            return False
        self.env.step(HAB_STOP)
        self.stop_issued = True
        self._slog("Harness", "HAB_STOP")
        return True

    def _estimate_goal_foothold(self):
        """分割当前画面目标并反投影到占用图落脚点；失败则用 confirmed_xyz。"""
        rgb, depth = self._obs()
        pos, rot = sensor_pose(self.env)
        body = body_position(self.env)
        result, _thr = self._segment(rgb, self.goal)
        if len(result) > 0:
            uv = mask_center_pixel(result.masks[0], depth)
            if uv is not None:
                world = pixel_to_world(uv[0], uv[1], depth, self.intrinsic, pos, rot)
                if world is not None:
                    stand = foothold_from_hit(self.occ, pos, world, body)
                    if stand is not None:
                        return np.asarray(stand, dtype=np.float64).reshape(3)
        if self.confirmed_xyz is not None:
            return np.asarray(self.confirmed_xyz, dtype=np.float64).reshape(3)
        return None

    def _stop_geodesic_ok(self, foothold):
        """占用图测地是否进入成功半径；图有洞时用同房间直线兜底。"""
        body = body_position(self.env)
        geo = occupancy_path_length(self.occ, body, foothold, success_dist=0.35)
        limit = float(self.success_distance_m)
        if path_geodesic_ok(geo) and float(geo) <= limit + 1e-6:
            return True, float(geo), "geodesic"
        euc = float(np.hypot(foothold[0] - body[0], foothold[2] - body[2]))
        if (not path_geodesic_ok(geo)) and euc <= limit + 1e-6:
            if self.occ.grid_line_clear(body, foothold):
                return True, euc, "euclid_same_room"
        return False, (None if not path_geodesic_ok(geo) else float(geo)), "reject"

    def _append_summary(self, text):
        """把 Summary 追加到 planner.txt。"""
        if not self.debug:
            return
        body = f"Summary\n{text}\n\n"
        with open(self.planner_log_path, "a", encoding="utf-8") as f:
            f.write(body)

    def _write_history_summary(self, planner_in, views, action, tool_log, rec,
                               pano_path=None):
        """调 Summary VLM，覆盖当前节点 summary 与语义 leftover。"""
        del planner_in
        bundle = compress_bundle(
            views, tool_log, action,
            getattr(self, "_scan_final_reasoning", "") or "", rec)
        client = getattr(self.planner, "client", None)
        leftover = []
        if client is None:
            text = f"Ran {action.get('action')}; state is {self.state.value}."
            leftover = [
                str(v.get("landmark") or v.get("room_type") or f"dir {v.get('pano_id')}")
                for v in (views or []) if v.get("unexplored")
            ]
        else:
            try:
                text, leftover = summarize(
                    client, bundle, self.goal, pano_path=pano_path)
            except VlmRetryExhausted as exc:
                self._abort(str(exc))
        node = self.graph.nodes[self.current_id]
        node["summary"] = text
        node["leftover"] = list(leftover or [])
        self._append_summary(text if not leftover else f"{text}\nleftover={leftover}")
        self._slog("Summary", text if not leftover else f"{text} leftover={leftover}")

    def apply_action(self, action, node_info):
        """执行终态动作。"""
        name = action["action"]
        if name in TOOL_NAMES:
            self._count(name)
        if name == "Verify":
            action = self._canon_verify_action(action)
            self._mark_selected_pano(action, node_info)
            pano_id = int(action.get("pano_id") or 0)
            self._slog("Planner", f"调用 Verify pano_id={pano_id} goal={self.goal} "
                       f"state={self.state.value}")
            face_pano(self.env, pano_id, node_yaw=node_info["yaw"])
            rgb0, _ = self._obs()
            vid = self.verify_count
            path0 = os.path.join(self.vlm_tmp, f"verify{vid}_view0.png")
            save_rgb(path0, rgb0)
            self._save_debug_rgb(
                os.path.join(self.out_dir, f"verify{vid}_view0.png"), rgb0)
            plan = {
                "action": "MakePlan",
                "pano_id": pano_id,
                "mode": "semantic",
                "object_query": self.goal,
                "plan": "Verify approach",
            }
            # Verify 靠近不得进入 Blocked
            report = self.run_mover(plan, node_info["frontiers"],
                                    max_legs=MAX_MOVER_LEGS, stop_on_geodesic=False)
            rgb1, _ = self._obs()
            path1 = os.path.join(self.vlm_tmp, f"verify{vid}_view1.png")
            save_rgb(path1, rgb1)
            self._save_debug_rgb(
                os.path.join(self.out_dir, f"verify{vid}_view1.png"), rgb1)
            paths = [path0, path1]
            client = getattr(self.planner, "client", None)
            vlm_ok = False
            parsed = {}
            if client is not None:
                try:
                    vlm_ok, parsed = vlm_verify_pair(client, path0, path1, self.goal)
                except VlmRetryExhausted as exc:
                    self._abort(str(exc))
            slim = {
                "ok": True,
                "consistency": bool(vlm_ok),
                "vlm_same": bool(vlm_ok),
                "vlm_reason": parsed.get("reason") if isinstance(parsed, dict) else None,
                "vlm_fields": {k: parsed.get(k) for k in (
                    "goal_in_view0", "goal_in_view1", "same_instance")
                    if isinstance(parsed, dict)},
                "mover": {k: report.get(k) for k in (
                    "status", "legs", "dist_moved_m", "chosen_ids")},
                "pano_id": pano_id,
            }
            self._append_verify_log(slim, paths)
            self.verify_count += 1
            self._slog("Verify", f"consistency={vlm_ok} {_short_json(slim)}")
            if vlm_ok:
                self._enter_confirmed()
                fh = self._estimate_goal_foothold()
                if fh is not None:
                    self.confirmed_xyz = fh.tolist()
                    self._slog("Verify", f"confirmed_xyz={self.confirmed_xyz}")
            else:
                self.state = NavState.UNSEEN
                self.blocked_from = None
            return {"action": "Verify", "result": slim, "ok": bool(vlm_ok)}
        if name == "TraceBack":
            self._slog("Planner", f"TraceBack node_id={action.get('node_id')}")
            result = self.do_traceback(int(action["node_id"]))
            self._apply_motion_outcome(
                result.get("dist_moved_m"), "TraceBack", allow_enter_blocked=False)
            self._slog("TraceBack", f"返回 {_short_json(result)}")
            return {"action": "TraceBack", "result": result}
        if name == "Locate":
            self.locate_count += 1
            pano_id = int(action.get("pano_id") or 0)
            self._slog(
                "Planner",
                f"Locate pano_id={pano_id} count={self.locate_count}/{MAX_LOCATE_ATTEMPTS}")
            plan = {
                "action": "MakePlan",
                "pano_id": pano_id,
                "mode": "semantic",
                "object_query": self.goal,
                "plan": "Locate goal",
            }
            report = self.run_mover(
                plan, node_info["frontiers"],
                max_legs=MAX_LOCATE_LEGS, stop_on_geodesic=True)
            if _mover_exec_fail(report):
                return {"action": "Locate", "plan": action, "mover": report,
                        "mover_retry": True}
            self._mark_selected_pano(action, node_info)
            force_stop = self.locate_count >= MAX_LOCATE_ATTEMPTS
            if self.state != NavState.ARRIVED and force_stop:
                self._slog("Locate", "第3次 Locate 强制 Stop")
                self._issue_stop()
                self.state = NavState.ARRIVED
                self.blocked_from = None
                report = dict(report)
                report["force_stop"] = True
                report["status"] = "stopped"
            elif self.state != NavState.ARRIVED:
                legs = int(report.get("legs") or 0)
                st = report.get("status")
                allow = legs >= 1 or st == "blocked"
                self._apply_motion_outcome(
                    report.get("dist_moved_m"), "Locate", allow_enter_blocked=allow)
            return {"action": "Locate", "plan": action, "mover": report,
                    "ok": self.stop_issued, "locate_count": self.locate_count}
        if name == "MakePlan":
            self.graph.nodes[self.current_id]["last_plan"] = {
                "mode": action.get("mode"), "pano_id": action.get("pano_id"),
                "object_query": action.get("object_query"), "plan": action.get("plan"),
            }
            report = self.run_mover(action, node_info["frontiers"])
            if _mover_exec_fail(report):
                return {"action": "MakePlan", "plan": action, "mover": report,
                        "mover_retry": True}
            self._mark_selected_pano(action, node_info)
            legs = int(report.get("legs") or 0)
            st = report.get("status")
            allow = legs >= 1 or st == "blocked"
            self._apply_motion_outcome(
                report.get("dist_moved_m"), "MakePlan", allow_enter_blocked=allow)
            return {"action": "MakePlan", "plan": action, "mover": report}
        return {"action": name, "ignored": True}

    def run(self, max_scans=8):
        """跑若干个 ScanNode 回合。"""
        t0 = time.perf_counter()
        if self.debug:
            self.topdown = TopdownRecorder(self.env, self.out_dir)
            attach_step_capture(self.env, self.topdown)
            self.topdown.capture()
        else:
            self.topdown = None
            if hasattr(self.env, "_hn_topdown_rec"):
                self.env._hn_topdown_rec = None

        log = []
        aborted = False
        try:
            for _ in range(max_scans):
                if self.env.episode_over or self.stop_issued:
                    break
                node_info = self.scan_node()
                action, _chat = self.planner_step(node_info)
                rec = None
                while True:
                    action = self._canon_verify_action(action)
                    rec = self.apply_action(action, node_info)
                    if not (rec.get("mover_retry") or rec.get("seg_retry")):
                        break
                    if (int(getattr(self, "_mover_retries_used", 0) or 0) >= MAX_MOVER_RETRIES
                            or not hasattr(self.planner, "feed_plan_retry")):
                        mover = dict(rec.get("mover") or {})
                        mover["status"] = "miss"
                        name = rec.get("action") or action.get("action") or "MakePlan"
                        rec = {"action": name, "plan": rec.get("plan") or action,
                               "mover": mover}
                        node = self.graph.nodes.get(self.current_id) or {}
                        if node.get("yaw") is not None:
                            turn_to_yaw(self.env, float(node["yaw"]))
                        # 一步未走用尽 retry：不进 Blocked
                        break
                    self._mover_retries_used = int(self._mover_retries_used) + 1
                    plan = rec.get("plan") or action
                    mover = rec.get("mover") or {}
                    caption = self._mover_retry_caption(plan, mover)
                    self._slog("retry", caption.replace("\n", " | "))
                    try:
                        self.planner.feed_plan_retry(caption)
                    except Exception as exc:
                        self._abort(f"VLM 回写执行失败说明失败: {exc}", node_info)
                    tools, actions = self._allowed()
                    require_verify = self.state == NavState.FIND
                    action = self._planner_until_terminal(
                        node_info, tools, actions, require_verify)
                    self._append_planner_retry(
                        caption, action, self._scan_final_reasoning)
                rec["state"] = self.state.value
                rec["node_id"] = node_info["node_id"]
                self._write_history_summary(
                    self._packup(node_info), node_info.get("views") or [],
                    action, getattr(self, "_scan_tool_log", []), rec,
                    pano_path=node_info.get("pano_vlm"))
                log.append(jsonable({k: v for k, v in rec.items() if k != "views"}))
                if self.stop_issued or self.state == NavState.ARRIVED:
                    break
        except EpisodeAbort as exc:
            aborted = True
            self.abort_reason = exc.reason
            self._slog("Harness", f"本集提前结束: {exc.reason}")
        finally:
            if self.debug:
                try:
                    rgb_final, _ = self._obs()
                    save_rgb(os.path.join(self.out_dir, "final_obs.png"), rgb_final)
                except Exception as exc:
                    self._slog("Harness", f"final_obs 未保存: {exc}")
                try:
                    save_rgb(os.path.join(self.out_dir, "final_bev.png"), self._render_bev())
                except Exception as exc:
                    self._slog("Harness", f"final_bev 未保存: {exc}")
            if not self.stop_issued:
                if self.env.episode_over:
                    self._slog("Harness", "本集已结束，跳过代发 HAB_STOP")
                else:
                    self._slog("Harness", "回合结束代发 HAB_STOP 以结算指标")
                    self._issue_stop()
            if self.topdown is not None:
                self.topdown.close()
            shutil.rmtree(self.vlm_tmp, ignore_errors=True)
        time_cost = round(time.perf_counter() - t0, 3)
        extra = {
            "time_cost": time_cost,
            "tool_counts": dict(self.tool_counts),
            "topdown_frames": int(self.topdown.frames) if self.topdown else 0,
            "aborted": aborted,
            "abort_reason": self.abort_reason,
        }
        if self.debug:
            extra["topdown_mp4"] = "topdown.mp4"
            extra["planner_txt"] = "planner.txt"
        dump_json(os.path.join(self.out_dir, "episode.json"), {
            "goal": self.goal, "state": self.state.value, "scans": self.scan_count,
            "nodes": len(self.graph.nodes), "log": log, **extra,
        })
        return {"state": self.state.value, "scans": self.scan_count,
                "nodes": len(self.graph.nodes), "out_dir": self.out_dir, **extra}
