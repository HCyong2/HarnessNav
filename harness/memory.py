"""节点图：扫描节点、边、History 渲染。node_id 只增不改号。"""

import math

import numpy as np

from harness.protocol import PANO_IDS, jsonable
from nav.goto import geodesic_m, occupancy_path_length
from nav.occupancy import to_rgb_uint8

NODE_MATCH_M = 0.4


def refresh_unexplored(node, sector_dirs):
    """按近距路径扇区与已选朝向刷新 ``views[].unexplored``。

    Args:
        node (dict): 节点。
        sector_dirs: 可迭代的偶序号朝向（有近距路径前沿）。
    """
    dirs = {int(x) for x in (sector_dirs or [])}
    explored = {int(x) for x in (node.get("explored_dirs") or [])}
    views = node.get("views") or []
    if views:
        for v in views:
            pid = int(v["pano_id"])
            v["unexplored"] = (pid in dirs) and (pid not in explored)
        return
    node["views"] = [
        {"pano_id": pid, "unexplored": (pid in dirs) and (pid not in explored)}
        for pid in PANO_IDS
    ]


def _xyz(p):
    """``(3,)`` float64。"""
    return np.asarray(p, dtype=np.float64).reshape(3)


class NodeGraph:
    """访问过的扫描节点与可通行边。全景保存在内存，不写 ``nodes/`` 目录。"""

    def __init__(self, root_dir=None):
        """初始化空图。

        Args:
            root_dir (str, optional): 兼容旧调用；不再用于落盘。
        """
        self.root_dir = root_dir
        self.nodes = {}
        self.edges = {}
        self._next_id = 0

    def nearest(self, xyz):
        """最近节点及其欧氏水平距离。

        Args:
            xyz: 世界坐标。

        Returns:
            tuple: ``(node_id, dist)``；空图为 ``(None, inf)``。
        """
        p = _xyz(xyz)
        best, best_d = None, float("inf")
        for nid, node in self.nodes.items():
            q = _xyz(node["xyz"])
            d = float(np.hypot(p[0] - q[0], p[2] - q[2]))
            if d < best_d:
                best, best_d = nid, d
        return best, best_d

    def upsert_scan(self, xyz, yaw, pano_rgbs, pano_depths, sector_dirs, last_plan,
                    summary, room_guess=""):
        """新建或重访。距旧点 < 0.4 m 视为重访。

        Args:
            xyz: 机身位置。
            yaw (float): 扫描 yaw。
            pano_rgbs (dict): ``pano_id -> RGB``。
            pano_depths (dict): ``pano_id -> (H,W)`` 深度。
            sector_dirs: 近距路径扇区集合。
            last_plan (dict): 本节点上次计划，可空。
            summary (str): 文字摘要；``None`` 表示保留原值。
            room_guess (str): 房间猜测。

        Returns:
            tuple: ``(node_id, revisit)``。
        """
        nid, dist = self.nearest(xyz)
        revisit = nid is not None and dist < NODE_MATCH_M
        if not revisit:
            nid = self._next_id
            self._next_id += 1
            self.nodes[nid] = {
                "node_id": nid,
                "xyz": _xyz(xyz).tolist(),
                "yaw": float(yaw),
                "visit_count": 0,
                "room_guess": room_guess,
                "pano": {},
                "leftover": [],
                "explored_dirs": [],
                "views": [],
                "last_plan": last_plan,
                "summary": "" if summary is None else summary,
            }
        node = self.nodes[nid]
        node["visit_count"] = int(node["visit_count"]) + 1
        node["xyz"] = _xyz(xyz).tolist()
        node["yaw"] = float(yaw)
        if "leftover" not in node:
            node["leftover"] = []
        if last_plan is not None:
            node["last_plan"] = last_plan
        if summary is not None:
            node["summary"] = summary
        refresh_unexplored(node, sector_dirs)
        if room_guess:
            node["room_guess"] = room_guess
        pano_meta = {}
        for pid in PANO_IDS:
            key = str(int(pid))
            entry = {}
            if pid in pano_rgbs:
                entry["rgb"] = to_rgb_uint8(pano_rgbs[pid])
            if pid in pano_depths:
                entry["depth"] = np.asarray(pano_depths[pid], dtype=np.float32)
            pano_meta[key] = entry
        node["pano"] = pano_meta
        return nid, revisit

    def add_edge(self, src, dst, env, occ=None):
        """两次不同 id 之间记一条边。代价优先 occupancy 路径长。

        Args:
            src (int): 起点。
            dst (int): 终点。
            env: 用来算测地（无 occupancy 时）。
            occ: 自建占用图。
        """
        if src is None or dst is None or src == dst:
            return
        a, b = min(int(src), int(dst)), max(int(src), int(dst))
        key = (a, b)
        if occ is not None:
            geo = occupancy_path_length(occ, self.nodes[src]["xyz"], self.nodes[dst]["xyz"])
            if not math.isfinite(geo):
                geo = 1e6
        else:
            geo = geodesic_m(env, self.nodes[src]["xyz"], self.nodes[dst]["xyz"])
        if key in self.edges:
            self.edges[key]["visits"] += 1
            self.edges[key]["geodesic_m"] = geo
        else:
            self.edges[key] = {"src": a, "dst": b, "geodesic_m": geo, "visits": 1}

    def leftover_count(self, node_id):
        """该节点语义 leftover 条数；无则回退 ``unexplored`` 扇区数。"""
        node = self.nodes.get(int(node_id))
        if node is None:
            return 0
        phrases = node.get("leftover") or []
        if phrases:
            return int(len(phrases))
        views = node.get("views") or []
        return int(sum(1 for v in views if v.get("unexplored")))

    def mark_explored(self, node_id, pano_id):
        """Planner 选定某朝向后，该扇 ``unexplored`` 粘性置 false。

        Args:
            node_id (int): 节点。
            pano_id (int): 偶序号朝向。
        """
        node = self.nodes.get(int(node_id))
        if node is None:
            return
        pid = int(pano_id)
        explored = [int(x) for x in (node.get("explored_dirs") or [])]
        if pid not in explored:
            explored.append(pid)
        node["explored_dirs"] = explored
        for v in node.get("views") or []:
            if int(v.get("pano_id")) == pid:
                v["unexplored"] = False

    def best_traceback_target(self, current_id, env, max_geo, occ=None):
        """leftover 最多且 occupancy 路径不超过上限的旧节点。"""
        best, best_n = None, 0
        cur = self.nodes[int(current_id)]["xyz"]
        for nid, node in self.nodes.items():
            if nid == current_id:
                continue
            n = self.leftover_count(nid)
            if n <= 0:
                continue
            if occ is not None:
                geo = occupancy_path_length(occ, cur, node["xyz"])
            else:
                geo = geodesic_m(env, cur, node["xyz"])
            if geo > max_geo:
                continue
            if n > best_n:
                best, best_n = nid, n
        return best

    def graph_path(self, src, dst):
        """边上的最短路（按测地），不含起点。

        Args:
            src (int): 起点。
            dst (int): 终点。

        Returns:
            list: node_id 序列；不通为空。
        """
        if src == dst:
            return []
        adj = {i: [] for i in self.nodes}
        for (a, b), ed in self.edges.items():
            adj[a].append((b, float(ed["geodesic_m"] or 1e6)))
            adj[b].append((a, float(ed["geodesic_m"] or 1e6)))
        dist = {i: float("inf") for i in self.nodes}
        prev = {i: None for i in self.nodes}
        dist[int(src)] = 0.0
        used = set()
        for _ in self.nodes:
            u, best = None, float("inf")
            for i, d in dist.items():
                if i not in used and d < best:
                    u, best = i, d
            if u is None:
                break
            used.add(u)
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
        if dist[int(dst)] == float("inf"):
            return []
        path = []
        cur = int(dst)
        while cur != int(src):
            path.append(cur)
            cur = prev[cur]
            if cur is None:
                return []
        path.reverse()
        return path

    def history(self, current_id, occ=None, frontiers=None, max_nodes=5):
        """PlannerIn.history：仅最近若干节点的摘要。

        当前节点默认不写入（本圈尚无 summary、顶栏已有 views）。
        回溯再访且已有 summary 时例外，把当前节点也列入。

        Args:
            current_id (int): 当前节点。
            occ: 占用图，可选（保留兼容）。
            frontiers (list): 当前前沿，可选（保留兼容）。
            max_nodes (int): 最多保留的历史节点数。

        Returns:
            list: 按 node_id 升序；起始可为空列表。
        """
        del occ, frontiers
        rows = []
        cur = None if current_id is None else int(current_id)
        for nid in sorted(self.nodes):
            node = self.nodes[nid]
            summary = node.get("summary", "") or ""
            if cur is not None and nid == cur and not summary.strip():
                continue
            leftover = node.get("leftover") or []
            if not isinstance(leftover, list):
                leftover = []
            rows.append({
                "node_id": nid,
                "visit_count": node["visit_count"],
                "summary": summary,
                "leftover": [str(x) for x in leftover],
            })
        if max_nodes is not None and len(rows) > int(max_nodes):
            rows = rows[-int(max_nodes):]
        return rows

    def overlays(self):
        """BEV 节点与边。"""
        nodes = [{"xyz": n["xyz"], "node_id": n["node_id"]} for n in self.nodes.values()]
        edges = []
        for ed in self.edges.values():
            if ed["src"] in self.nodes and ed["dst"] in self.nodes:
                edges.append({
                    "src_xyz": self.nodes[ed["src"]]["xyz"],
                    "dst_xyz": self.nodes[ed["dst"]]["xyz"],
                })
        return nodes, edges
