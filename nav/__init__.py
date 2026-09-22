"""导航侧工具：坐标变换、几何计算、点云占用图与 occupancy A* 跟随。

2D occupancy（log-odds + 膨胀）同时用于前沿提取与离散 Habitat 动作规划；
Habitat navmesh 仅作兼容保留，主路径不再调用 greedy follower。
"""
