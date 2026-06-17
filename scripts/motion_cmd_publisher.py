#!/usr/bin/env python3
"""Publish pure motion REFERENCE (33D) from a motion pkl to /p73/motion_ref @ 50Hz.

This is the deploy-side counterpart of the isaaclab_walker_motion training env's
motion reference. It is bit-exact to how the env produces q_ref / qd_ref / root:

  - q_ref  = dof_pos[t]  in ALL_JOINT_NAMES order (= PKL order; net reorder = identity, verified)
  - qd_ref = centered finite-diff of dof_pos (dt = 1/fps) -> motion_lib._compute_velocity
  - root   = motion root (pelvis/base_link == root_pos/root_rot), root_rot converted xyzw->wxyz
  - fps != 50: env runs motion_time = step*step_dt (step_dt=0.02s) and INTERPOLATES between
    source frames inside motion_lib.calc_motion_frame (lerp pos/vel, slerp rot). This publisher
    replicates that interpolation exactly so the published reference matches the env per-step.

The publisher is robot-state-INDEPENDENT (no anchor). cc.cpp computes the anchor error
(obs[32:41]) from its own q_virtual_; see PLAN_qref_flow_deploy.md §2/§3/§5.

Output topic: /p73/motion_ref  (std_msgs/Float64MultiArray, 33D, 50Hz)
    [0:13]   q_ref          (13, ALL_JOINT_NAMES order, rad)
    [13:26]  qd_ref         (13, ALL_JOINT_NAMES order, rad/s)  -- centered finite-diff
    [26:29]  ref_root_pos   (3, m)                              -- motion root position
    [29:33]  ref_root_quat  (4, wxyz)                           -- motion root orientation

Also publishes /p73/ghost_state (20D absolute pose) for MuJoCo ghost visualization (harmless).

Usage:
    python3 motion_cmd_publisher.py [--pkl PATH] [--loop] [--start FRAME]

NOTE: filename kept as motion_cmd_publisher.py to avoid launch/script breakage; the
published TOPIC and payload are the new 33D /p73/motion_ref contract.
"""

import argparse
import os
import pickle

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# ---------------------------------------------------------------------------
# Joint orders (single source: isaaclab_walker_motion).
#   motion_lib.WALKER_JOINT_ORDER == mimic_env_cfg.ALL_JOINT_NAMES (verified identical),
#   so the net PKL->ALL reorder is identity. We still build it by name as a guard.
# ---------------------------------------------------------------------------
PKL_JOINT_ORDER = [  # motion_lib.WALKER_JOINT_ORDER
    "L_HipRoll_Joint", "L_HipPitch_Joint", "L_HipYaw_Joint",
    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    "R_HipRoll_Joint", "R_HipPitch_Joint", "R_HipYaw_Joint",
    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
    "WaistYaw_Joint",
]
ALL_JOINT_NAMES = [  # mimic_env_cfg.LEG_JOINT_NAMES + WAIST_JOINT_NAMES
    "L_HipRoll_Joint", "L_HipPitch_Joint", "L_HipYaw_Joint",
    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    "R_HipRoll_Joint", "R_HipPitch_Joint", "R_HipYaw_Joint",
    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
    "WaistYaw_Joint",
]

STEP_DT = 0.02  # env step_dt = decimation*sim_dt = 50Hz control rate


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz, match motion_utils.py exactly)
# ---------------------------------------------------------------------------
def xyzw_to_wxyz(q):
    """(N,4) xyzw -> (N,4) wxyz."""
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def quat_slerp(q0, q1, t):
    """Spherical linear interp, wxyz. Replicates motion_utils.quat_slerp (single pair)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    cos_half = float(np.dot(q0, q1))
    if cos_half < 0.0:
        q1 = -q1
        cos_half = -cos_half
    if cos_half > 0.9999:
        res = q0 + t * (q1 - q0)
        return res / np.linalg.norm(res)
    half = np.arccos(np.clip(cos_half, -1.0, 1.0))
    sin_half = np.sin(half)
    sin_half = sin_half if sin_half > 1e-8 else 1e-8
    ra = np.sin((1.0 - t) * half) / sin_half
    rb = np.sin(t * half) / sin_half
    return ra * q0 + rb * q1


def compute_velocity(data, dt):
    """Centered finite-diff matching motion_lib._compute_velocity (dt = 1/fps)."""
    vel = np.zeros_like(data)
    n = data.shape[0]
    if n > 2:
        vel[1:-1] = (data[2:] - data[:-2]) / (2.0 * dt)
    if n > 1:
        vel[0] = (data[1] - data[0]) / dt
        vel[-1] = (data[-1] - data[-2]) / dt
    return vel


def euler_from_quat(q_wxyz):
    w, x, y, z = q_wxyz
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Motion clip wrapper — replicates motion_lib loading + calc_motion_frame
# ---------------------------------------------------------------------------
class MotionClip:
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.fps = float(data["fps"])
        self.dt = 1.0 / self.fps
        self.root_pos = np.asarray(data["root_pos"], dtype=np.float64)        # (N,3)
        root_rot_raw = np.asarray(data["root_rot"], dtype=np.float64)         # (N,4) xyzw
        # root_rot is XYZW in the pkl (augment_motion passes it through; cross-checked vs
        # body_quat_w base_link which is wxyz). motion_lib uses quat_convention='xyzw' ->
        # xyzw_to_wxyz. We do the same.
        self.root_rot = xyzw_to_wxyz(root_rot_raw)                            # (N,4) wxyz
        dof_pos_pkl = np.asarray(data["dof_pos"], dtype=np.float64)           # (N,13) PKL order

        # Net reorder PKL -> ALL_JOINT_NAMES. Both lists identical -> identity.
        reorder = [PKL_JOINT_ORDER.index(n) for n in ALL_JOINT_NAMES]
        self.reorder_is_identity = (reorder == list(range(len(reorder))))
        self.dof_pos = dof_pos_pkl[:, reorder]                               # (N,13) ALL order

        # Precompute qd over the whole clip (then interpolate, exactly like the env:
        # motion_lib computes dof_vel at source frames, calc_motion_frame lerps it).
        self.dof_vel = compute_velocity(self.dof_pos, self.dt)               # (N,13) ALL order

        self.num_frames = self.root_pos.shape[0]
        # motion_lib._motion_lengths = (num_frames - 1) * dt
        self.motion_len = max((self.num_frames - 1) * self.dt, 1e-6)

    def frame_at_time(self, motion_time, loop):
        """Replicate motion_lib.calc_motion_frame for a single env at given motion_time.

        Returns (q_ref(13,), qd_ref(13,), root_pos(3,), root_quat(4, wxyz)).
        """
        if loop:
            t = motion_time % self.motion_len
        else:
            t = min(motion_time, self.motion_len)
        phase = np.clip(t / self.motion_len, 0.0, 1.0)
        f = phase * (self.num_frames - 1)
        i0 = int(f)  # == .long() floor for f >= 0
        i0 = min(i0, self.num_frames - 1)
        i1 = min(i0 + 1, self.num_frames - 1)
        blend = f - i0

        q_ref = (1.0 - blend) * self.dof_pos[i0] + blend * self.dof_pos[i1]
        qd_ref = (1.0 - blend) * self.dof_vel[i0] + blend * self.dof_vel[i1]
        root_pos = (1.0 - blend) * self.root_pos[i0] + blend * self.root_pos[i1]
        root_quat = quat_slerp(self.root_rot[i0], self.root_rot[i1], blend)
        return q_ref, qd_ref, root_pos, root_quat


class MotionRefPublisher(Node):
    def __init__(self, pkl_path, loop=True, start_frame=0):
        super().__init__("motion_cmd_publisher")

        self.clip = MotionClip(pkl_path)
        self.loop = loop
        # start_frame -> motion_time offset (frame at source fps)
        self.start_time = start_frame * self.clip.dt
        self.time_acc = 0.0

        # q_default standing fallback (frame 0 reference), published before CC mode.
        q0, qd0, rp0, rq0 = self.clip.frame_at_time(0.0, loop=False)
        self.standing_ref = np.zeros(33)
        self.standing_ref[0:13] = q0
        self.standing_ref[13:26] = 0.0  # qd_ref = 0 at standstill
        self.standing_ref[26:29] = rp0
        self.standing_ref[29:33] = rq0

        self.pub = self.create_publisher(Float64MultiArray, "/p73/motion_ref", 10)
        self.ghost_pub = self.create_publisher(Float64MultiArray, "/p73/ghost_state", 10)

        # CC mode trigger (task_mode 5~9) — same gating as before.
        try:
            from p73_msgs.msg import TaskCmd
            self.task_sub = self.create_subscription(TaskCmd, "/p73/taskCommand", self._task_cb, 10)
            self.get_logger().info("Subscribed to /p73/taskCommand (p73_msgs/TaskCmd)")
        except ImportError:
            from std_msgs.msg import UInt32
            self.task_sub = self.create_subscription(UInt32, "/p73/taskCommand", self._task_uint_cb, 10)
            self.get_logger().warn("p73_msgs not found, using UInt32 fallback for /p73/taskCommand")

        self.policy_dt = STEP_DT
        self.cc_active = False
        self.timer = self.create_timer(self.policy_dt, self.timer_callback)

        self.get_logger().info(
            f"Motion REF publisher: {pkl_path} ({self.clip.num_frames} frames @ {self.clip.fps}fps, "
            f"len={self.clip.motion_len:.2f}s, loop={loop}, reorder_identity={self.clip.reorder_is_identity})"
        )
        self.get_logger().info("Publishing 33D /p73/motion_ref. Waiting for CC mode (task_mode 5~9)...")

    def _activate(self, mode):
        if 5 <= mode < 10:
            if not self.cc_active:
                self.cc_active = True
                self.time_acc = 0.0
                self.get_logger().info(f"CC mode {mode} active — motion ref playback from start.")
        else:
            if self.cc_active:
                self.cc_active = False
                self.get_logger().info(f"CC mode deactivated (task_mode={mode})")

    def _task_cb(self, msg):
        self._activate(msg.task_mode)

    def _task_uint_cb(self, msg):
        self._activate(msg.data)

    def timer_callback(self):
        if not self.cc_active:
            # Standing fallback (first callback / pre-CC). cc.cpp keeps default pose.
            msg = Float64MultiArray()
            msg.data = self.standing_ref.tolist()
            self.pub.publish(msg)
            return

        self.time_acc += self.policy_dt
        motion_time = self.start_time + self.time_acc
        q_ref, qd_ref, root_pos, root_quat = self.clip.frame_at_time(motion_time, self.loop)

        ref = np.zeros(33)
        ref[0:13] = q_ref
        ref[13:26] = qd_ref
        ref[26:29] = root_pos
        ref[29:33] = root_quat
        msg = Float64MultiArray()
        msg.data = ref.tolist()
        self.pub.publish(msg)

        # Ghost (absolute pose) for MuJoCo ghost viz: pos(3) + quat wxyz(4) + dof(13).
        ghost = np.zeros(20)
        ghost[0:3] = root_pos
        ghost[3:7] = root_quat
        ghost[7:20] = q_ref
        gmsg = Float64MultiArray()
        gmsg.data = ghost.tolist()
        self.ghost_pub.publish(gmsg)


def main():
    parser = argparse.ArgumentParser(description="Publish 33D pure motion reference (/p73/motion_ref)")
    _default_pkl = os.path.join(
        os.path.dirname(__file__), "..", "motion_data", "p73_walk_short_full.pkl"
    )
    if not os.path.isfile(_default_pkl):
        _default_pkl = (
            "/home/piene/isaaclab5.2/isaaclab_walker_motion/source/isaaclab_walker_motion/"
            "isaaclab_walker_motion/assets/data/p73_walker/motion_data/p73_walk_short_full.pkl"
        )
    parser.add_argument("--pkl", type=str, default=_default_pkl, help="Path to motion pkl file")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop motion")
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--start", type=int, default=0, help="Start frame (source-fps frame index)")
    args = parser.parse_args()

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
