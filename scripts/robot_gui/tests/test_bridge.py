"""RobotBridge 로직 테스트 — ROS2 없이 (_publish 스텁, setup_ros 미호출)."""
from __future__ import annotations

import time

import pytest

from robot_gui.bridge_node import RobotBridge
from robot_gui.descriptor import load_descriptor


class FakeBridge(RobotBridge):
    def __init__(self, descriptor, **kw):
        super().__init__(descriptor, **kw)
        self.published: list[tuple] = []

    def _publish(self, topic, type_str, fields):
        self.published.append((topic, type_str, fields))

    def force_connected(self):
        self._last_state_t = time.monotonic()


class FakeClip:
    num_frames = 100
    fps = 50.0
    def frame_33d(self, idx):
        return [float(idx)] * 33
    def standing_33d(self):
        return [0.0] * 33


def _bridge():
    d = load_descriptor("p73_walker")   # 브랜치=인터페이스, profile 불필요
    b = FakeBridge(d, clip_loader_cls=lambda path: FakeClip())
    b.force_connected()
    return b


def test_module_imports_without_ros():
    assert RobotBridge is not None


def test_command_gating_blocks_before_connect():
    b = FakeBridge(load_descriptor("p73_walker"), clip_loader_cls=lambda p: FakeClip())
    with pytest.raises(PermissionError):
        b.send_command("torque_on")


def test_bringup_sequence_and_fsm():
    b = _bridge()
    assert b.send_command("torque_on")["state"] == "READY"
    assert b.published[-1] == ("p73/guiCommand", "std_msgs/String", {"data": "torqOn"})
    r = b.send_command("set_mode", 7)
    assert r["state"] == "RUNNING"
    assert b.published[-1] == ("p73/taskCommand", "p73_msgs/TaskCmd", {"task_mode": 7})


def test_set_mode_blocked_without_torque():
    b = _bridge()
    with pytest.raises(PermissionError):
        b.send_command("set_mode", 7)


def test_ik_mode_not_running():
    b = _bridge()
    b.send_command("torque_on")
    b.send_command("set_mode", 4)
    assert not b.fsm.mode_running


def test_play_blocked_until_mode_running():
    b = _bridge()
    b.motion_select("dummy.pkl")
    b.send_command("torque_on")
    with pytest.raises(PermissionError):
        b.motion_play()


def test_motion_tick_standing_then_frame():
    b = _bridge()
    b.motion_select("dummy.pkl")
    b.send_command("torque_on")
    b.send_command("set_mode", 7)
    b._motion_tick()
    assert b.published[-1][2]["data"] == [0.0] * 33
    b.motion_play()
    b._motion_tick()
    assert b.published[-1][0] == "/p73/motion_cmd"
    assert b.published[-1][2]["data"] != [0.0] * 33


def test_motion_tick_silent_when_not_running():
    b = _bridge()
    b.motion_select("dummy.pkl")
    n = len(b.published)
    b._motion_tick()
    assert len(b.published) == n


def test_snapshot_shape():
    b = _bridge()
    snap = b.snapshot()
    assert snap["robot"] == "p73_walker"
    assert snap["branch"] == "walker_motion"
    assert "can" in snap and "play" in snap["can"]


def test_init_pose_publishes_poscmd_after_torque():
    b = _bridge()
    with pytest.raises(PermissionError):
        b.send_command("init_pose")        # 토크 전엔 불가
    b.send_command("torque_on")
    b.send_command("init_pose")
    topic, typ, fields = b.published[-1]
    assert topic == "p73/posCommand" and typ == "p73_msgs/PosCmd"
    assert len(fields["position"]) == 32 and fields["traj_time"] == 3.0


def test_snapshot_can_includes_init_pose():
    b = _bridge()
    assert "init_pose" in b.snapshot()["can"]


def test_descriptor_public_dict_endpoint_source():
    # /descriptor 가 내보내는 dict (UI 가 이것만 보고 렌더)
    b = _bridge()
    pub = b.d.to_public_dict()
    assert pub["joints"] and pub["modes"]
