"""robot_gui.descriptor 단위 테스트 (순수 로직, ROS2 불필요)."""
from __future__ import annotations

import pytest

from robot_gui.descriptor import (
    ControlAction,
    DescriptorError,
    RobotDescriptor,
    list_robots,
    load_descriptor,
    merge_robot_profile,
)


def test_load_walker():
    d = load_descriptor("p73_walker")
    assert d.name == "p73_walker"
    assert d.num_joints == 13
    assert d.joint_names[0] == "L_HipRoll"
    assert d.joint_names[-1] == "WaistYaw"


def test_walker_in_registry():
    assert "p73_walker" in list_robots()


def test_branch_travels_with_yaml():
    # 브랜치 = 인터페이스: robot.yaml 이 현재 브랜치를 운반 (단일 install)
    d = load_descriptor("p73_walker")
    assert d.branch == "walker_motion"


def test_control_actions():
    d = load_descriptor("p73_walker")
    torq = d.control("torque_on")
    assert torq.topic == "p73/guiCommand"
    assert torq.build_message() == {"data": "torqOn"}


def test_set_mode_parameterized():
    d = load_descriptor("p73_walker")
    assert d.control("set_mode").build_message(7) == {"task_mode": 7}


def test_set_mode_requires_value():
    d = load_descriptor("p73_walker")
    with pytest.raises(DescriptorError):
        d.control("set_mode").build_message()


def test_modes():
    d = load_descriptor("p73_walker")
    ids = {m.id: m.label for m in d.modes}
    assert ids[4] == "IK (float)"
    assert ids[7] == "RL motion mimic"


def test_motion_spec_and_pkg_expand():
    from robot_gui.descriptor import _expand
    d = load_descriptor("p73_walker")
    assert d.motion.topic == "/p73/motion_cmd"
    assert d.motion.rate_hz == 50
    # {PKG} 가 p73_cc 절대경로로 치환되어 motion_data 를 가리킴
    expanded = [_expand(x) for x in d.motion.dirs]
    assert expanded and all("{PKG}" not in x for x in expanded)
    assert any(x.endswith("/motion_data") for x in expanded)


def test_to_public_dict_for_ui():
    d = load_descriptor("p73_walker")
    pub = d.to_public_dict()
    assert pub["name"] == "p73_walker"
    assert {"id": 7, "label": "RL motion mimic"} in pub["modes"]
    assert pub["joints"][0] == "L_HipRoll"
    assert pub["branch"] == "walker_motion"
    assert "{PKG}" not in pub["viz"].get("urdf", "")


def test_unknown_action_raises():
    with pytest.raises(DescriptorError):
        load_descriptor("p73_walker").control("nope")


def test_unknown_robot_raises():
    with pytest.raises(DescriptorError):
        load_descriptor("no_such_robot")


def test_missing_required_keys():
    with pytest.raises(DescriptorError):
        RobotDescriptor.from_dict({"name": "x"})


def test_empty_payload_builds_empty():
    assert ControlAction(topic="t/e", type="std_msgs/Empty").build_message() == {}


def test_init_pose_multifield_message():
    d = load_descriptor("p73_walker")
    msg = d.control("init_pose").build_message()
    assert d.control("init_pose").topic == "p73/posCommand"
    assert len(msg["position"]) == 32           # PosCmd float64[32]
    assert msg["position"][1] == 0.18           # L_HipPitch init
    assert msg["traj_time"] == 3.0
    assert msg["gravity"] is False


def test_no_install_tree_field():
    # 2026-06-03 단일 install 마이그레이션 → install_tree 미사용
    d = load_descriptor("p73_walker")
    assert d.install_tree is None
    assert d.branch == "walker_motion"


def test_merge_profile_overlay_pure():
    # profile 은 한 브랜치 내 sub-variant 용 (선택적). 순수 merge 동작 확인.
    robot = {"name": "r", "joints": {"names": ["j0"]},
             "controls": {"torque_on": {"topic": "r/g", "type": "std_msgs/String", "payload": "on"}}}
    profile = {"name": "p", "robot": "r", "branch": "exp1",
               "controls": {"set_phase": {"topic": "r/ph", "type": "std_msgs/Float32", "field": "data"}},
               "modes": [{"id": 8, "label": "exp"}]}
    merged = merge_robot_profile(robot, profile)
    d = RobotDescriptor.from_dict(merged)
    assert "torque_on" in d.controls and "set_phase" in d.controls
    assert d.profile == "p" and d.branch == "exp1"
    assert d.modes[0].label == "exp"
