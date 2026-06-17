"""fsm.py — 제어 상태 머신 (순수 로직, ROS2 불필요).

상태는 3개 불리언(connected/torque_on/mode_running)에서 파생, UI 버튼 게이팅(can())의
단일 진실 소스. DISCONNECTED → CONNECTED → READY(torque on) → RUNNING(mode start)
"""
from __future__ import annotations

DISCONNECTED = "DISCONNECTED"
CONNECTED = "CONNECTED"
READY = "READY"
RUNNING = "RUNNING"


class ControlFSM:
    def __init__(self) -> None:
        self.connected = False
        self.torque_on = False
        self.mode_running = False

    def set_connected(self, ok: bool) -> None:
        self.connected = bool(ok)
        if not ok:
            self.torque_on = False
            self.mode_running = False

    def set_torque(self, on: bool) -> None:
        self.torque_on = bool(on) and self.connected
        if not self.torque_on:
            self.mode_running = False

    def set_mode_running(self, on: bool) -> None:
        self.mode_running = bool(on) and self.torque_on

    @property
    def state(self) -> str:
        if not self.connected:
            return DISCONNECTED
        if self.mode_running:
            return RUNNING
        if self.torque_on:
            return READY
        return CONNECTED

    def can(self, action: str) -> bool:
        if action in ("torque_on", "torque_off", "init_yaw", "state_estimate",
                      "safety_reset"):
            return self.connected
        # 로봇을 움직이는 명령(자세 이동/모드)은 토크 On 이후
        if action in ("set_mode", "init_pose", "zero_pose"):
            return self.torque_on
        if action in ("play", "pause", "stop", "seek", "select", "loop", "tracker"):
            return self.mode_running
        return self.connected
