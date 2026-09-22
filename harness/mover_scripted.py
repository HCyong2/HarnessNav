"""P0 脚本 Mover：在仍大于阈值的候选里取最近，差 <0.15 m 取更高分。"""


class ScriptedMover:
    """只根据 candidates 选 id。"""

    def __init__(self, success_distance_m=0.5, tie_m=0.15):
        """初始化。

        Args:
            success_distance_m (float): 子目标深度阈值。
            tie_m (float): 并列距离差。
        """
        self.success_distance_m = float(success_distance_m)
        self.tie_m = float(tie_m)

    def pick(self, mover_in):
        """选出一个候选 id。

        Args:
            mover_in (dict): ``MoverIn``。

        Returns:
            dict: ``{"id": ...}``。
        """
        cands = list(mover_in.get("candidates") or [])
        if not cands:
            return {"id": None}
        thresh = float(mover_in.get("near_m", mover_in.get(
            "success_distance_m", self.success_distance_m)))
        far = [c for c in cands if float(c["depth_m"]) > thresh]
        pool = far if far else cands
        pool = sorted(pool, key=lambda c: float(c["depth_m"]))
        best = pool[0]
        for c in pool[1:]:
            if abs(float(c["depth_m"]) - float(best["depth_m"])) < self.tie_m:
                if float(c.get("score") or 0) > float(best.get("score") or 0):
                    best = c
            else:
                break
        return {"id": best["id"]}
