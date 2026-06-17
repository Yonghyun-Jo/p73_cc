"""transport.MotionTransport + fsm.ControlFSM 단위 테스트 (순수 로직)."""
from __future__ import annotations

from robot_gui.fsm import CONNECTED, DISCONNECTED, READY, RUNNING, ControlFSM
from robot_gui.transport import MotionTransport


def test_transport_starts_standing():
    t = MotionTransport(num_frames=100, fps=50)
    assert t.emitting_standing
    assert t.current_frame == 0


def test_play_advances_frame():
    t = MotionTransport(num_frames=100, fps=50)
    t.play()
    t.advance(0.1)
    assert t.current_frame == 5


def test_loop_wraps():
    t = MotionTransport(num_frames=10, fps=50, loop=True)
    t.play()
    t.advance(10.0)
    assert 0 <= t.current_frame < 10


def test_no_loop_clamps_and_stops():
    t = MotionTransport(num_frames=10, fps=50, loop=False)
    t.play()
    t.advance(100.0)
    assert t.current_frame == 9
    assert t.emitting_standing


def test_pause_holds():
    t = MotionTransport(num_frames=100, fps=50)
    t.play(); t.advance(0.2); t.pause()
    f = t.current_frame
    t.advance(1.0)
    assert t.current_frame == f


def test_stop_resets():
    t = MotionTransport(num_frames=100, fps=50)
    t.play(); t.advance(0.5); t.stop()
    assert t.current_frame == 0 and t.emitting_standing


def test_seek():
    t = MotionTransport(num_frames=100, fps=50)
    t.seek(40); assert t.current_frame == 40
    t.seek(99999); assert t.current_frame == 99


def test_select_resets():
    t = MotionTransport(num_frames=100, fps=50)
    t.play(); t.advance(0.5)
    t.select(num_frames=30, fps=30)
    assert t.num_frames == 30 and t.current_frame == 0 and t.emitting_standing


def test_empty_clip_safe():
    t = MotionTransport(num_frames=0, fps=50)
    t.play(); t.advance(1.0)
    assert t.current_frame == 0 and t.emitting_standing


def test_fsm_flow():
    f = ControlFSM()
    assert f.state == DISCONNECTED and not f.can("torque_on")
    f.set_connected(True)
    assert f.state == CONNECTED and f.can("torque_on") and not f.can("set_mode")
    f.set_torque(True)
    assert f.state == READY and f.can("set_mode") and not f.can("play")
    f.set_mode_running(True)
    assert f.state == RUNNING and f.can("play")


def test_fsm_torque_off_drops_mode():
    f = ControlFSM()
    f.set_connected(True); f.set_torque(True); f.set_mode_running(True)
    f.set_torque(False)
    assert f.state == CONNECTED and not f.can("play")


def test_fsm_disconnect_resets():
    f = ControlFSM()
    f.set_connected(True); f.set_torque(True); f.set_mode_running(True)
    f.set_connected(False)
    assert f.state == DISCONNECTED and not f.torque_on and not f.mode_running


def test_fsm_mode_requires_torque():
    f = ControlFSM()
    f.set_connected(True); f.set_mode_running(True)
    assert not f.mode_running and f.state == CONNECTED


def test_fsm_init_pose_requires_torque():
    # 실제 순서: Torque On → Init Pose. 자세 이동은 토크 On 이후만.
    f = ControlFSM()
    f.set_connected(True)
    assert not f.can("init_pose") and not f.can("zero_pose")
    f.set_torque(True)
    assert f.can("init_pose") and f.can("zero_pose")
