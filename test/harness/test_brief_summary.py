#!/usr/bin/env python
"""摘要文件与 Habitat 日志过滤。"""

import io
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from harness.protocol import write_brief_summary
from harness.quiet_sim import _FilterStream, _should_drop


def test_should_drop_habitat_noise():
    """加载场景与缺少语义网格的行应丢弃。"""
    assert _should_drop("2026-09-21 18:43:17,182 Initializing dataset ObjectNav-v1")
    assert _should_drop("2026-09-21 18:43:20,712 initializing sim Sim-v0")
    assert _should_drop("2026-09-21 18:43:21,682 Initializing task ObjectNav-v1")
    assert _should_drop(
        "[Warning]:[Scene] SemanticScene.cpp(133)::load : "
        "Semantic Scene Descriptor didn't load, semantic information unavailable")
    assert _should_drop(
        "[20:08:43:370776]:[Error]:[Scene] SemanticScene.cpp(137)::"
        "loadSemanticSceneDescriptor : SSD Load Failure! File with "
        "SemanticAttributes-provided name "
        "`/DATA_HDD/hc/dataset-3d/data/scene_datasets/hm3d_v0.2/val/"
        "00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.scn` exists but failed to load.")
    assert _should_drop(
        " Failure! File with SemanticAttributes-provided name "
        "`/DATA_HDD/hc/dataset-3d/data/scene_datasets/hm3d_v0.2/val/"
        "00876-mv2HUxq3B53/mv2HUxq3B53.basis.scn` exists but failed to load.")
    assert _should_drop("Gym has been unmaintained since 2022")
    assert not _should_drop("=== ep000 success=1.0 spl=0.5 ===")
    assert not _should_drop("progress 3/10")


def test_filter_stream_keeps_real_lines():
    """过滤流只丢掉 Habitat 噪音。"""
    buf = io.StringIO()
    stream = _FilterStream(buf)
    stream.write("Initializing dataset ObjectNav-v1\n")
    stream.write("agent=0 tasks=2\n")
    stream.write("No semantic information available\n")
    stream.flush()
    assert buf.getvalue() == "agent=0 tasks=2\n"


def test_write_brief_summary():
    """写出成功率、距离、路径效率与耗时。"""
    records = [
        {"success": 1.0, "distance_to_goal": 0.5, "spl": 0.8,
         "soft_spl": 0.7, "time_cost": 10.0},
        {"success": 0.0, "distance_to_goal": 3.5, "spl": 0.0,
         "soft_spl": 0.1, "time_cost": 20.0},
    ]
    means = {
        "success": 0.5,
        "distance_to_goal": 2.0,
        "spl": 0.4,
        "soft_spl": 0.4,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_brief_summary(tmp, records, means, 35.2, n_assigned=2)
        text = open(path, encoding="utf-8").read()
    assert "n=2" in text
    assert "sr=0.5000" in text
    assert "dtg=2.0000" in text
    assert "spl=0.4000" in text
    assert "total_time_s=35.2" in text
    assert "mean_episode_time_s=15.0" in text


if __name__ == "__main__":
    test_should_drop_habitat_noise()
    test_filter_stream_keeps_real_lines()
    test_write_brief_summary()
    print("ok")
