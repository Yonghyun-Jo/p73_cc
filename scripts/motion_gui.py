#!/usr/bin/env python3
"""Motion data control GUI for P73 Walker.

Standalone PyQt5 window that controls motion data playback.
Publishes /p73/motion_cmd (19D) for policy and /p73/ghost_state (20D) for MuJoCo ghost.

Usage:
    walker
    python3 ~/ros2_ws/src/p73_cc/scripts/motion_gui.py
"""

import glob
import os
import pickle
import sys

import numpy as np

# ROS 2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# Qt
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)


# ── Motion math (from motion_cmd_publisher.py) ──────────────────────

def quat_rotate_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    t = 2.0 * np.cross(np.array([-x, -y, -z]), v)
    return v + w * t + np.cross(np.array([-x, -y, -z]), t)


def extract_yaw_quat(q_wxyz):
    w, x, y, z = q_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _quat_mul_wxyz(a, b):
    """Hamilton product of two quaternions in wxyz convention."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def euler_from_quat(q_wxyz):
    w, x, y, z = q_wxyz
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def compute_motion_cmd(root_pos, root_rot_wxyz, root_vel_w, root_ang_vel_w, dof_pos):
    yaw_quat = extract_yaw_quat(root_rot_wxyz)
    root_vel_local = quat_rotate_inverse(yaw_quat, root_vel_w)
    roll, pitch, _ = euler_from_quat(root_rot_wxyz)
    cmd = np.zeros(19)
    cmd[0] = root_vel_local[0]
    cmd[1] = root_vel_local[1]
    cmd[2] = root_pos[2]
    cmd[3] = roll
    cmd[4] = pitch
    cmd[5] = root_ang_vel_w[2]
    cmd[6:19] = dof_pos[:13]
    return cmd


# ── Motion data loader ──────────────────────────────────────────────

class MotionData:
    """Loads and precomputes a motion pkl file."""

    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.fps = int(data["fps"])
        self.root_pos = np.array(data["root_pos"], dtype=np.float64)
        # xyzw → wxyz
        rot_raw = np.array(data["root_rot"], dtype=np.float64)
        self.root_rot = np.zeros_like(rot_raw)
        self.root_rot[:, 0] = rot_raw[:, 3]
        self.root_rot[:, 1] = rot_raw[:, 0]
        self.root_rot[:, 2] = rot_raw[:, 1]
        self.root_rot[:, 3] = rot_raw[:, 2]
        self.dof_pos = np.array(data["dof_pos"], dtype=np.float64)
        self.num_frames = len(self.root_pos)

        # ── Align motion data to robot spawn direction ──
        # Motion data faces -Y (yaw≈-90°), robot spawns facing +X (yaw=0°).
        # Rotate +90° around Z, then anchor frame 0 to robot's side (y=+0.5m).
        self._align_to_robot(yaw_offset_deg=90.0, ghost_origin=np.array([0.0, 0.5, 0.0]))

        dt = 1.0 / self.fps
        self.root_vel = np.zeros_like(self.root_pos)
        self.root_vel[1:] = (self.root_pos[1:] - self.root_pos[:-1]) / dt
        self.root_vel[0] = self.root_vel[1]

        self.root_ang_vel = np.zeros((self.num_frames, 3))
        for i in range(1, self.num_frames):
            r0, p0, y0 = euler_from_quat(self.root_rot[i - 1])
            r1, p1, y1 = euler_from_quat(self.root_rot[i])
            self.root_ang_vel[i] = np.array([r1 - r0, p1 - p0, y1 - y0]) / dt
        self.root_ang_vel[0] = self.root_ang_vel[1]

    def _align_to_robot(self, yaw_offset_deg: float, ghost_origin: np.ndarray):
        """Rotate all root_pos/root_rot by yaw_offset around Z, then translate so frame 0 starts at ghost_origin."""
        theta = np.radians(yaw_offset_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # 2D rotation matrix for x,y
        rot2d = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Rotate positions (x,y only, keep z)
        xy_rotated = (rot2d @ self.root_pos[:, :2].T).T  # (N, 2)
        self.root_pos[:, 0] = xy_rotated[:, 0]
        self.root_pos[:, 1] = xy_rotated[:, 1]

        # Translate so frame 0 xy = ghost_origin xy
        offset_xy = ghost_origin[:2] - self.root_pos[0, :2]
        self.root_pos[:, 0] += offset_xy[0]
        self.root_pos[:, 1] += offset_xy[1]

        # Rotate quaternions: q_new = q_z(theta) * q_old
        # q_z = (cos(theta/2), 0, 0, sin(theta/2)) in wxyz
        qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
        for i in range(self.num_frames):
            self.root_rot[i] = _quat_mul_wxyz(qz, self.root_rot[i])

    def get_motion_cmd(self, frame: int) -> np.ndarray:
        f = np.clip(frame, 0, self.num_frames - 1)
        return compute_motion_cmd(
            self.root_pos[f], self.root_rot[f],
            self.root_vel[f], self.root_ang_vel[f], self.dof_pos[f],
        )

    def get_ghost_state(self, frame: int) -> np.ndarray:
        f = np.clip(frame, 0, self.num_frames - 1)
        ghost = np.zeros(20)
        ghost[0:3] = self.root_pos[f]
        ghost[3:7] = self.root_rot[f]
        ghost[7:20] = self.dof_pos[f, :13]
        return ghost

    def get_standing_cmd(self) -> np.ndarray:
        cmd = np.zeros(19)
        cmd[2] = self.root_pos[0, 2]
        cmd[6:19] = self.dof_pos[0, :13]
        return cmd


# ── ROS node ────────────────────────────────────────────────────────

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_gui")
        self.cmd_pub = self.create_publisher(Float64MultiArray, "/p73/motion_cmd", 10)
        self.ghost_pub = self.create_publisher(Float64MultiArray, "/p73/ghost_state", 10)

        # Subscribe to taskCommand for auto-start
        try:
            from p73_msgs.msg import TaskCmd
            self.task_sub = self.create_subscription(TaskCmd, "/p73/taskCommand", self._task_cb, 10)
            self._use_taskcmd = True
        except ImportError:
            from std_msgs.msg import UInt32
            self.task_sub = self.create_subscription(UInt32, "/p73/taskCommand", self._task_uint_cb, 10)
            self._use_taskcmd = False

        self.cc_mode_activated = False  # set by callback, read by GUI

    def _task_cb(self, msg):
        if 5 <= msg.task_mode < 10:
            self.cc_mode_activated = True

    def _task_uint_cb(self, msg):
        if 5 <= msg.data < 10:
            self.cc_mode_activated = True

    def publish_cmd(self, cmd: np.ndarray):
        try:
            msg = Float64MultiArray()
            msg.data = cmd.tolist()
            self.cmd_pub.publish(msg)
        except Exception:
            pass

    def publish_ghost(self, ghost: np.ndarray):
        try:
            msg = Float64MultiArray()
            msg.data = ghost.tolist()
            self.ghost_pub.publish(msg)
        except Exception:
            pass


# ── GUI ─────────────────────────────────────────────────────────────

class MotionGuiWindow(QWidget):
    MOTION_DATA_DIR = os.path.expanduser("~/ros2_ws/src/p73_cc/motion_data")

    def __init__(self, node: MotionNode):
        super().__init__()
        self.node = node
        self.motion: MotionData | None = None
        self.playing = False
        self.current_frame = 0
        self.time_acc = 0.0
        self.policy_dt = 0.02  # 50Hz, matches IsaacLab decimation=4 * sim_dt=0.005
        self._slider_pressed = False

        self.setWindowTitle("P73 Motion Control")
        self.setMinimumWidth(420)
        self._build_ui()
        self._load_file_list()

        self._shutting_down = False

        # 50Hz publish timer
        self._pub_timer = QTimer(self)
        self._pub_timer.timeout.connect(self._tick)
        self._pub_timer.start(20)

        # 10ms ROS spin timer
        self._ros_timer = QTimer(self)
        self._ros_timer.timeout.connect(self._ros_spin)
        self._ros_timer.start(10)

    def _ros_spin(self):
        if self._shutting_down:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0)
        except Exception:
            self.shutdown()

    def shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self._pub_timer.stop()
        self._ros_timer.stop()
        QApplication.instance().quit()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # File selector
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Motion:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        file_row.addWidget(self.file_combo, 1)
        layout.addLayout(file_row)

        # Transport controls
        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.chk_loop = QCheckBox("Loop")
        self.chk_loop.setChecked(True)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.chk_loop)
        layout.addLayout(btn_row)

        # Timeline slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.sliderPressed.connect(self._slider_press)
        self.slider.sliderReleased.connect(self._slider_release)
        self.slider.valueChanged.connect(self._slider_changed)
        layout.addWidget(self.slider)

        # Status
        self.lbl_status = QLabel("Frame: 0 / 0  |  Time: 0.0s / 0.0s")
        layout.addWidget(self.lbl_status)

        # Auto-start
        self.chk_auto = QCheckBox("Auto-start on CC mode (task_mode 5~9)")
        self.chk_auto.setChecked(True)
        layout.addWidget(self.chk_auto)

    def _load_file_list(self):
        self.file_combo.clear()
        pkls = sorted(glob.glob(os.path.join(self.MOTION_DATA_DIR, "*.pkl")))
        if not pkls:
            # Fallback to isaaclab location
            alt = os.path.expanduser(
                "~/isaaclab5.2/isaaclab_walker_motion/source/isaaclab_walker_motion/"
                "isaaclab_walker_motion/assets/data/p73_walker/motion_data"
            )
            pkls = sorted(glob.glob(os.path.join(alt, "*.pkl")))
        for p in pkls:
            self.file_combo.addItem(os.path.basename(p), p)
        if pkls:
            self._load_motion(pkls[0])

    def _on_file_changed(self, idx):
        path = self.file_combo.itemData(idx)
        if path:
            self._load_motion(path)

    def _load_motion(self, path: str):
        self.playing = False
        self.current_frame = 0
        self.time_acc = 0.0
        self.motion = MotionData(path)
        self.slider.setMaximum(self.motion.num_frames - 1)
        self.slider.setValue(0)
        self._update_label()
        self.node.get_logger().info(
            f"Loaded: {os.path.basename(path)} ({self.motion.num_frames} frames, "
            f"{self.motion.num_frames / self.motion.fps:.1f}s)"
        )

    # ── Transport ──

    def _on_play(self):
        if self.motion:
            if not self.playing:
                self.time_acc = self.current_frame / self.motion.fps
            self.playing = True

    def _on_pause(self):
        self.playing = False

    def _on_stop(self):
        self.playing = False
        self.current_frame = 0
        self.time_acc = 0.0
        self.slider.setValue(0)

    def _slider_press(self):
        self._slider_pressed = True

    def _slider_release(self):
        self._slider_pressed = False
        self.current_frame = self.slider.value()
        self.time_acc = self.current_frame / self.motion.fps if self.motion else 0.0

    def _slider_changed(self, val):
        if self._slider_pressed:
            self.current_frame = val
            self._update_label()

    # ── Main loop (50Hz) ──

    def _tick(self):
        if self._shutting_down or not self.motion:
            return

        # Auto-start on CC mode
        if self.chk_auto.isChecked() and self.node.cc_mode_activated:
            self.node.cc_mode_activated = False
            self.time_acc = 0.0
            self.current_frame = 0
            self.playing = True
            self.node.get_logger().info("CC mode detected — auto-starting motion from frame 0")

        if self.playing:
            # Advance by real time: 0.02s per tick × 30fps = 0.6 frames/tick
            # Matches IsaacLab: policy_dt(0.02) × motion_fps(30) = 0.6 frames/step
            self.time_acc += self.policy_dt
            self.current_frame = int(self.time_acc * self.motion.fps) % self.motion.num_frames
            if not self.chk_loop.isChecked() and self.time_acc * self.motion.fps >= self.motion.num_frames:
                self.current_frame = self.motion.num_frames - 1
                self.playing = False

        # Publish
        if self.playing or self._slider_pressed:
            cmd = self.motion.get_motion_cmd(self.current_frame)
            ghost = self.motion.get_ghost_state(self.current_frame)
            self.node.publish_cmd(cmd)
            self.node.publish_ghost(ghost)
        else:
            # Standing reference when stopped
            cmd = self.motion.get_standing_cmd()
            self.node.publish_cmd(cmd)
            # Ghost shows current frame pose (standing or paused position)
            ghost = self.motion.get_ghost_state(self.current_frame)
            self.node.publish_ghost(ghost)

        # Update UI
        if not self._slider_pressed:
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)
        self._update_label()

    def _update_label(self):
        if not self.motion:
            return
        f = self.current_frame
        n = self.motion.num_frames
        t = f / self.motion.fps
        total = n / self.motion.fps
        state = "Playing" if self.playing else "Stopped"
        self.lbl_status.setText(f"{state}  |  Frame: {f} / {n}  |  Time: {t:.1f}s / {total:.1f}s")


# ── Main ────────────────────────────────────────────────────────────

def main():
    import signal

    rclpy.init()
    node = MotionNode()

    app = QApplication(sys.argv)
    win = MotionGuiWindow(node)
    win.show()

    # Let SIGINT/SIGTERM cleanly quit the Qt loop
    signal.signal(signal.SIGINT, lambda *_: win.shutdown())
    signal.signal(signal.SIGTERM, lambda *_: win.shutdown())
    # Ensure Python signal handlers run even when Qt blocks
    sig_timer = QTimer()
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start(200)

    ret = app.exec_()

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    sys.exit(ret)


if __name__ == "__main__":
    main()
