#!/usr/bin/env python3
"""Publish motion reference command from pkl motion data to /p73/motion_cmd.

Reads a motion pkl file and publishes the 19D motion command at 50Hz.
This mirrors Isaac Lab's calc_current_motion_command_proprio().

Usage:
    python3 motion_cmd_publisher.py [--pkl PATH] [--loop] [--start FRAME]

Motion command layout (19D):
    [0:2]  root_vel_local_xy (yaw-aligned body frame, m/s)
    [2]    root_pos_z (height, m)
    [3]    roll (rad)
    [4]    pitch (rad)
    [5]    root_ang_vel_z (yaw rate, rad/s)
    [6:19] ref_dof_pos (13 joints, rad)
"""

import argparse
import pickle
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def quat_rotate_inverse(q_wxyz, v):
    """Rotate vector by inverse of quaternion (wxyz convention)."""
    w, x, y, z = q_wxyz
    # q_inv for unit quaternion = conjugate
    # v_rotated = q_inv * v * q
    t = 2.0 * np.cross(np.array([-x, -y, -z]), v)
    return v + w * t + np.cross(np.array([-x, -y, -z]), t)


def extract_yaw_quat(q_wxyz):
    """Extract yaw-only quaternion from full quaternion (wxyz)."""
    w, x, y, z = q_wxyz
    # yaw = atan2(2*(wz+xy), 1-2*(yy+zz))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Return yaw-only quaternion (wxyz)
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def euler_from_quat(q_wxyz):
    """Euler angles (roll, pitch, yaw) from quaternion (wxyz)."""
    w, x, y, z = q_wxyz
    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def compute_motion_cmd(root_pos, root_rot_wxyz, root_vel_w, root_ang_vel_w, dof_pos):
    """Compute 19D motion command matching Isaac Lab calc_current_motion_command_proprio.

    Args:
        root_pos: (3,) root position
        root_rot_wxyz: (4,) root quaternion (wxyz)
        root_vel_w: (3,) root linear velocity in world frame
        root_ang_vel_w: (3,) root angular velocity in world frame
        dof_pos: (13,) joint positions
    Returns:
        (19,) motion command
    """
    # Yaw-only quaternion for frame alignment
    yaw_quat = extract_yaw_quat(root_rot_wxyz)
    # Root velocity in yaw-aligned local frame
    root_vel_local = quat_rotate_inverse(yaw_quat, root_vel_w)

    # Euler angles from full quaternion
    roll, pitch, yaw = euler_from_quat(root_rot_wxyz)

    # Yaw angular velocity (world z-axis component)
    yaw_rate = root_ang_vel_w[2]

    cmd = np.zeros(19)
    cmd[0] = root_vel_local[0]  # vx local
    cmd[1] = root_vel_local[1]  # vy local
    cmd[2] = root_pos[2]        # root_z (height)
    cmd[3] = roll
    cmd[4] = pitch
    cmd[5] = yaw_rate
    cmd[6:19] = dof_pos[:13]
    return cmd


class MotionCmdPublisher(Node):
    def __init__(self, pkl_path, loop=True, start_frame=0):
        super().__init__("motion_cmd_publisher")

        # Load motion data
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.fps = int(data["fps"])
        self.root_pos = np.array(data["root_pos"], dtype=np.float64)
        root_rot_raw = np.array(data["root_rot"], dtype=np.float64)
        # pkl stores root_rot as xyzw (despite body_quat_convention="wxyz" metadata).
        # Isaac Lab's MotionCfg uses quat_convention="xyzw" and converts to wxyz internally.
        # We do the same: xyzw → wxyz
        self.root_rot = np.zeros_like(root_rot_raw)
        self.root_rot[:, 0] = root_rot_raw[:, 3]  # w
        self.root_rot[:, 1] = root_rot_raw[:, 0]  # x
        self.root_rot[:, 2] = root_rot_raw[:, 1]  # y
        self.root_rot[:, 3] = root_rot_raw[:, 2]  # z
        self.dof_pos = np.array(data["dof_pos"], dtype=np.float64)
        self.num_frames = len(self.root_pos)
        self.loop = loop

        # Compute velocities via finite differences (at motion fps)
        dt_motion = 1.0 / self.fps
        self.root_vel = np.zeros_like(self.root_pos)
        self.root_vel[1:] = (self.root_pos[1:] - self.root_pos[:-1]) / dt_motion
        self.root_vel[0] = self.root_vel[1]

        # Root angular velocity from quaternion differences
        self.root_ang_vel = np.zeros((self.num_frames, 3))
        for i in range(1, self.num_frames):
            # Simple finite difference on euler angles
            r0, p0, y0 = euler_from_quat(self.root_rot[i - 1])
            r1, p1, y1 = euler_from_quat(self.root_rot[i])
            self.root_ang_vel[i] = np.array([r1 - r0, p1 - p0, y1 - y0]) / dt_motion
        self.root_ang_vel[0] = self.root_ang_vel[1]

        # Publisher
        self.pub = self.create_publisher(Float64MultiArray, "/p73/motion_cmd", 10)

        # Timer at 50Hz (policy rate)
        self.policy_dt = 0.02  # 50Hz
        self.timer = self.create_timer(self.policy_dt, self.timer_callback)

        self.frame_idx = start_frame
        self.time_acc = 0.0

        self.get_logger().info(
            f"Motion publisher: {pkl_path} ({self.num_frames} frames @ {self.fps}fps, "
            f"duration={self.num_frames/self.fps:.1f}s, loop={loop})"
        )

    def timer_callback(self):
        # Advance time and compute current motion frame (interpolation at policy rate)
        self.time_acc += self.policy_dt
        frame_float = self.time_acc * self.fps
        frame = int(frame_float) % self.num_frames if self.loop else min(int(frame_float), self.num_frames - 1)

        if not self.loop and int(frame_float) >= self.num_frames:
            # End of motion, publish last frame repeatedly
            frame = self.num_frames - 1

        cmd = compute_motion_cmd(
            self.root_pos[frame],
            self.root_rot[frame],
            self.root_vel[frame],
            self.root_ang_vel[frame],
            self.dof_pos[frame],
        )

        msg = Float64MultiArray()
        msg.data = cmd.tolist()
        self.pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="Publish motion reference command")
    # Default: look for motion data in p73_cc package first
    import os
    _default_pkl = os.path.join(os.path.dirname(__file__), "..", "motion_data", "p73_walk1_subject5_full.pkl")
    if not os.path.isfile(_default_pkl):
        _default_pkl = "/home/piene/isaaclab5.2/isaaclab_walker_motion/source/isaaclab_walker_motion/isaaclab_walker_motion/assets/data/p73_walker/motion_data/p73_walk1_subject5_full.pkl"
    parser.add_argument(
        "--pkl",
        type=str,
        default=_default_pkl,
        help="Path to motion pkl file",
    )
    parser.add_argument("--loop", action="store_true", default=True, help="Loop motion")
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    args = parser.parse_args()

    rclpy.init()
    node = MotionCmdPublisher(args.pkl, loop=args.loop, start_frame=args.start)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
