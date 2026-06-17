#!/usr/bin/env python3
"""Publish q_ref motion REFERENCE (33D) from a motion pkl to /p73/motion_cmd @ 50Hz.

★ DELIVERY = walker_vision sparse publisher 와 동일 (검증됨, 끊김/ghost-rewind 없음):
  - time_acc += policy_dt 누적 → frame = int(time_acc*fps) % num_frames (loop)
  - CC-mode 게이팅: /p73/taskCommand task_mode 5~9 활성 시 time_acc=0 후 playback,
    그 전엔 standing reference publish. (motion_time 보간 방식은 폐기 — 그게 버그 원인이었음.)
  - /p73/ghost_state (20D) 동일.
오직 PAYLOAD 만 다름: 25D sparse → 33D q_ref.

Output topic: /p73/motion_cmd  (std_msgs/Float64MultiArray, 33D, 50Hz)
    [0:13]   q_ref          (13, ALL_JOINT_NAMES order, rad)
    [13:26]  qd_ref         (13, ALL_JOINT_NAMES order, rad/s)  -- centered finite-diff
    [26:29]  ref_root_pos   (3, m)                              -- motion root position
    [29:33]  ref_root_quat  (4, wxyz)                           -- motion root orientation
anchor error(obs[32:41])는 cc.cpp 가 q_virtual_ 로 계산 (publisher 는 robot-state 무관).

Usage:
    python3 motion_cmd_publisher.py [--pkl PATH] [--loop] [--start FRAME]
"""

import argparse
import os
import pickle

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# PKL dof_pos 순서 == ALL_JOINT_NAMES (motion_lib.WALKER_JOINT_ORDER 와 동일, net reorder=identity 검증).
ALL_JOINT_NAMES = [
    "L_HipRoll_Joint", "L_HipPitch_Joint", "L_HipYaw_Joint",
    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    "R_HipRoll_Joint", "R_HipPitch_Joint", "R_HipYaw_Joint",
    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
    "WaistYaw_Joint",
]


class MotionRefPublisher(Node):
    def __init__(self, pkl_path, loop=True, start_frame=0):
        super().__init__("motion_cmd_publisher")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.fps = int(data["fps"])
        self.root_pos = np.array(data["root_pos"], dtype=np.float64)        # (T,3)
        root_rot_raw = np.array(data["root_rot"], dtype=np.float64)         # (T,4) xyzw
        self.root_rot = np.zeros_like(root_rot_raw)                         # (T,4) wxyz
        self.root_rot[:, 0] = root_rot_raw[:, 3]   # w
        self.root_rot[:, 1] = root_rot_raw[:, 0]   # x
        self.root_rot[:, 2] = root_rot_raw[:, 1]   # y
        self.root_rot[:, 3] = root_rot_raw[:, 2]   # z
        self.dof_pos = np.array(data["dof_pos"], dtype=np.float64)          # (T,13) ALL order
        self.num_frames = len(self.root_pos)
        self.loop = loop

        # qd_ref: centered finite-diff (matches motion_lib._compute_velocity, dt=1/fps).
        dt = 1.0 / self.fps
        self.dof_vel = np.zeros_like(self.dof_pos)
        if self.num_frames > 2:
            self.dof_vel[1:-1] = (self.dof_pos[2:] - self.dof_pos[:-2]) / (2.0 * dt)
        if self.num_frames > 1:
            self.dof_vel[0] = (self.dof_pos[1] - self.dof_pos[0]) / dt
            self.dof_vel[-1] = (self.dof_pos[-1] - self.dof_pos[-2]) / dt

        # Standing reference (before CC mode): frame-0 pose, zero joint vel (in-distribution).
        self.standing_ref = self._make_ref(0, zero_vel=True)

        self.pub = self.create_publisher(Float64MultiArray, "/p73/motion_cmd", 10)
        self.ghost_pub = self.create_publisher(Float64MultiArray, "/p73/ghost_state", 10)

        # CC-mode trigger (identical to walker_vision).
        try:
            from p73_msgs.msg import TaskCmd
            self.task_sub = self.create_subscription(TaskCmd, "/p73/taskCommand", self._task_cb, 10)
            self.get_logger().info("Subscribed to /p73/taskCommand (p73_msgs/TaskCmd)")
        except ImportError:
            from std_msgs.msg import UInt32
            self.task_sub = self.create_subscription(UInt32, "/p73/taskCommand", self._task_uint_cb, 10)
            self.get_logger().warn("p73_msgs not found, using UInt32 fallback for /p73/taskCommand")

        self.policy_dt = 0.02
        self.timer = self.create_timer(self.policy_dt, self.timer_callback)
        self.cc_active = False
        self.time_acc = float(start_frame) / self.fps

        self.get_logger().info(
            f"q_ref motion_ref publisher (33D): {pkl_path} ({self.num_frames} frames @ {self.fps}fps, "
            f"duration={self.num_frames / self.fps:.1f}s, loop={loop})")
        self.get_logger().info("Waiting for CC mode (task_mode 5~9) to start motion playback...")

    def _make_ref(self, frame, zero_vel=False):
        ref = np.zeros(33)
        ref[0:13] = self.dof_pos[frame]
        ref[13:26] = 0.0 if zero_vel else self.dof_vel[frame]
        ref[26:29] = self.root_pos[frame]
        ref[29:33] = self.root_rot[frame]
        return ref

    def _set_cc(self, mode):
        if 5 <= mode < 10 and not self.cc_active:
            self.cc_active = True
            self.time_acc = 0.0
            self.get_logger().info(f"CC mode {mode} activated — motion playback starting from frame 0")
        elif (mode < 5 or mode >= 10) and self.cc_active:
            self.cc_active = False
            self.get_logger().info(f"CC mode deactivated (task_mode={mode})")

    def _task_cb(self, msg):
        self._set_cc(msg.task_mode)

    def _task_uint_cb(self, msg):
        self._set_cc(msg.data)

    def timer_callback(self):
        if not self.cc_active:
            # Before CC mode: publish standing reference, no ghost.
            msg = Float64MultiArray()
            msg.data = self.standing_ref.tolist()
            self.pub.publish(msg)
            return

        # Advance time (accumulated, NOT wall-clock) and compute integer frame.
        self.time_acc += self.policy_dt
        frame_float = self.time_acc * self.fps
        frame = int(frame_float) % self.num_frames if self.loop else min(int(frame_float), self.num_frames - 1)

        ref = self._make_ref(frame)
        msg = Float64MultiArray()
        msg.data = ref.tolist()
        self.pub.publish(msg)

        # Ghost (20D absolute pose for MuJoCo ghost viz).
        ghost = np.zeros(20)
        ghost[0:3] = self.root_pos[frame]
        ghost[3:7] = self.root_rot[frame]
        ghost[7:20] = self.dof_pos[frame, :13]
        gmsg = Float64MultiArray()
        gmsg.data = ghost.tolist()
        self.ghost_pub.publish(gmsg)


def main():
    parser = argparse.ArgumentParser(description="Publish q_ref motion reference (33D)")
    _d = os.path.join(os.path.dirname(__file__), "..", "motion_data", "p73_walk_short_full.pkl")
    if not os.path.isfile(_d):
        _d = ("/home/piene/isaaclab5.2/isaaclab_walker_motion/source/isaaclab_walker_motion/"
              "isaaclab_walker_motion/assets/data/p73_walker/motion_data/p73_walk_short_full.pkl")
    parser.add_argument("--pkl", type=str, default=_d)
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--start", type=int, default=0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = MotionRefPublisher(args.pkl, loop=args.loop, start_frame=args.start)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
