"""transport.py — 모션 재생 상태 머신 (순수 로직, ROS2 불필요).

motion_gui.py 의 transport 의미를 추출: play/pause/stop/seek/loop + 정지 시 STANDING.
시간 진행(advance)은 dt 를 받는다 → 브리지가 wall-clock 또는 sim_time delta 를 먹여
sim_time 페이싱을 구현. 클립의 실제 프레임→33D 변환은 브리지의 ClipLoader 몫.
"""
from __future__ import annotations


class MotionTransport:
    def __init__(self, num_frames: int = 0, fps: float = 50.0, loop: bool = True):
        self.num_frames = int(num_frames)
        self.fps = float(fps)
        self.loop = bool(loop)
        self.playing = False
        self.elapsed = 0.0

    def select(self, num_frames: int, fps: float) -> None:
        self.num_frames = int(num_frames)
        self.fps = float(fps)
        self.playing = False
        self.elapsed = 0.0

    def play(self) -> None:
        if self.num_frames > 0:
            self.playing = True

    def pause(self) -> None:
        self.playing = False

    def stop(self) -> None:
        self.playing = False
        self.elapsed = 0.0

    def seek(self, frame: int) -> None:
        if self.num_frames <= 0:
            return
        frame = max(0, min(int(frame), self.num_frames - 1))
        self.elapsed = frame / self.fps

    def set_loop(self, loop: bool) -> None:
        self.loop = bool(loop)

    def advance(self, dt: float) -> None:
        if not self.playing or self.num_frames <= 0:
            return
        self.elapsed += dt
        total = self.num_frames / self.fps
        if self.loop:
            self.elapsed %= total
        elif self.elapsed >= total:
            self.elapsed = total
            self.playing = False

    @property
    def current_frame(self) -> int:
        if self.num_frames <= 0:
            return 0
        idx = int(self.elapsed * self.fps)
        if self.loop:
            return idx % self.num_frames
        return min(idx, self.num_frames - 1)

    @property
    def emitting_standing(self) -> bool:
        return not self.playing
