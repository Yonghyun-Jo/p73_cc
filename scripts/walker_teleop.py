#!/usr/bin/env python3
"""
Unified Walker Teleop — THE ONLY way to command the robot.

Publishes geometry_msgs/Twist to /p73/cmd_vel.
Subscribes to /joy (sensor_msgs/Joy) for joystick input.

Modes:
  [KEY] Keyboard mode (default):
    w/s : vx +/- 0.1   a/d : vy +/- 0.1   q/e : wz +/- 0.1
    space : zero all
  [JOY] Joystick mode:
    Left stick  → vx, vy    Right stick → wz
    A : push_event   X : viz_toggle   Start : estop
    RB/Back : scale up/down   Y : reset scale + clear estop

Mode switching:
  j         → JOY mode
  k         → KEY mode
  Y button  → KEY mode (from joystick)
  Ctrl+C    → quit

Usage:
  walker-teleop   (alias for bash ~/ros2_ws/src/p73_cc/scripts/walker_teleop.sh)
"""

from __future__ import annotations

import os
import subprocess
import sys
import select
import threading
import tty
import termios
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------
def apply_deadzone(v: float, dz: float) -> float:
    """Zero out values below deadzone, rescale the rest to [0, 1]."""
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0.0 else -1.0
    return sign * (abs(v) - dz) / max(1.0 - dz, 1e-6)


def apply_expo(v: float, e: float) -> float:
    """Blend linear + cubic for soft center, responsive edges."""
    if e <= 0.0:
        return v
    return (1.0 - e) * v + e * (v * v * v)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def get_key(settings, timeout: float = 0.05) -> str:
    """Read a single key with timeout (works over SSH)."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


# ---------------------------------------------------------------------------
# Joystick config (env vars allow wrapper to override per platform)
# Defaults = Ubuntu/Linux (joy_8bitdo_ultimate2_linux.yaml)
# ---------------------------------------------------------------------------
class JoyConfig:
    axis_vx:    int  = int(os.environ.get("JOY_AXIS_VX", "1"))
    axis_vy:    int  = int(os.environ.get("JOY_AXIS_VY", "0"))
    axis_wz:    int  = int(os.environ.get("JOY_AXIS_WZ", "3"))
    axis_height: int = int(os.environ.get("JOY_AXIS_HEIGHT", "4"))  # right stick Y
    invert_vx:  bool = os.environ.get("JOY_INVERT_VX", "0") == "1"
    invert_vy:  bool = os.environ.get("JOY_INVERT_VY", "0") == "1"
    invert_wz:  bool = os.environ.get("JOY_INVERT_WZ", "0") == "1"
    invert_height: bool = os.environ.get("JOY_INVERT_HEIGHT", "0") == "1"
    max_vx:     float = 1.0
    max_vy:     float = 0.5
    max_wz:     float = 1.0
    deadzone:   float = 0.10
    expo:       float = 0.30
    timeout_s:  float = 0.5
    require_deadman: bool = os.environ.get("JOY_REQUIRE_DEADMAN", "1") == "1"
    height_rate: float = 0.5  # m/s at full stick deflection

    # Buttons (8BitDo Ultimate 2, XInput)
    BTN_PUSH      = 0   # A
    BTN_VIZ       = 2   # X
    BTN_RESET     = 3   # Y  → also switches to KEY mode
    BTN_DEADMAN   = 4   # LB
    BTN_SCALE_UP  = 5   # RB
    BTN_SCALE_DN  = 6   # Back/Select
    BTN_ESTOP     = 7   # Start

    SCALE_STEP    = 0.10
    SCALE_MIN     = 0.20
    SCALE_MAX     = 1.00


# ---------------------------------------------------------------------------
# Keyboard config
# ---------------------------------------------------------------------------
KEY_STEP  = 0.1
KEY_MAX_V = 1.0
KEY_MAX_WZ = 0.6
HEIGHT_STEP = 0.05
HEIGHT_MIN = 0.55
HEIGHT_MAX = 0.89
ROLL_STEP = 0.05
ROLL_MIN = -0.25
ROLL_MAX = 0.25
PITCH_STEP = 0.1
PITCH_MIN = -0.5
PITCH_MAX = 0.5
PUB_HZ    = 50.0

KEY_BINDINGS = {
    'w': ('vx', +KEY_STEP),  's': ('vx', -KEY_STEP),
    'a': ('vy', +KEY_STEP),  'd': ('vy', -KEY_STEP),
    'q': ('wz', +KEY_STEP),  'e': ('wz', -KEY_STEP),
    'r': ('h', +HEIGHT_STEP),  'f': ('h', -HEIGHT_STEP),
    't': ('roll', +ROLL_STEP),  'g': ('roll', -ROLL_STEP),
    'y': ('pitch', +PITCH_STEP),  'h': ('pitch', -PITCH_STEP),
}

BANNER = """
=== Walker Teleop (Crouch + Pose) ===
  [KEY] w/s:fwd/back  a/d:left/right  q/e:rotate
        r/f:height  t/g:roll  y/h:pitch  space:stop
  [JOY] left stick:vx,vy  right stick X:wz  Y:height
  Mode: j→JOY  k→KEY  Y button→KEY    Ctrl+C: quit
======================================
"""


# ---------------------------------------------------------------------------
# Unified Teleop Node
# ---------------------------------------------------------------------------
class WalkerTeleop(Node):

    def __init__(self):
        super().__init__('walker_teleop')
        self.cfg = JoyConfig()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=10,
        )

        # Publishers
        self.cmd_pub  = self.create_publisher(Twist, '/p73/cmd_vel', qos)
        self.push_pub = self.create_publisher(Empty, '/p73/push_event', qos)
        self.viz_pub  = self.create_publisher(Empty, '/p73/viz_toggle', qos)

        # Subscriber
        self.create_subscription(Joy, '/joy', self._on_joy, qos)

        # --- State ---
        self.mode = "KEY"

        # Keyboard
        self.key_vx = 0.0
        self.key_vy = 0.0
        self.key_wz = 0.0

        # Pose commands (shared across modes)
        self.height = HEIGHT_MAX  # default standing
        self.roll = 0.0
        self.pitch = 0.0

        # Joystick
        self.joy_vx = 0.0
        self.joy_vy = 0.0
        self.joy_wz = 0.0
        self.joy_height_rate = 0.0  # right stick Y → height change rate
        self.joy_alive = False
        self.deadman_held = not self.cfg.require_deadman
        self.estop_active = False
        self.scale = 1.0
        self._last_joy_time = 0.0

        # Button edge detection (previous state)
        self._prev_btn = {}

    # -- Joy callback (runs in spin thread) ---------------------------------
    def _on_joy(self, msg: Joy):
        cfg = self.cfg
        self._last_joy_time = time.monotonic()

        def ax(i: int) -> float:
            return float(msg.axes[i]) if 0 <= i < len(msg.axes) else 0.0

        def bt(i: int, default: int = 0) -> int:
            return int(msg.buttons[i]) if 0 <= i < len(msg.buttons) else default

        def pressed(idx: int) -> bool:
            """True on rising edge only."""
            cur = bt(idx)
            prev = self._prev_btn.get(idx, 0)
            self._prev_btn[idx] = cur
            return cur == 1 and prev == 0

        # Axes → velocity
        raw_vx = ax(cfg.axis_vx) * (-1.0 if cfg.invert_vx else 1.0)
        raw_vy = ax(cfg.axis_vy) * (-1.0 if cfg.invert_vy else 1.0)
        raw_wz = ax(cfg.axis_wz) * (-1.0 if cfg.invert_wz else 1.0)

        proc_vx = apply_expo(apply_deadzone(raw_vx, cfg.deadzone), cfg.expo)
        proc_vy = apply_expo(apply_deadzone(raw_vy, cfg.deadzone), cfg.expo)
        proc_wz = apply_expo(apply_deadzone(raw_wz, cfg.deadzone), cfg.expo)

        self.joy_vx = proc_vx * cfg.max_vx * self.scale
        self.joy_vy = proc_vy * cfg.max_vy * self.scale
        self.joy_wz = proc_wz * cfg.max_wz * self.scale

        # Right stick Y → height change rate
        raw_h = ax(cfg.axis_height) * (-1.0 if cfg.invert_height else 1.0)
        proc_h = apply_expo(apply_deadzone(raw_h, cfg.deadzone), cfg.expo)
        self.joy_height_rate = proc_h * cfg.height_rate  # m/s

        # Deadman
        self.deadman_held = (bt(cfg.BTN_DEADMAN, default=1) == 1) or (not cfg.require_deadman)

        # Buttons (rising edge only)
        if pressed(cfg.BTN_RESET):       # Y → KEY mode + reset scale
            self.mode = "KEY"
            self.key_vx = self.key_vy = self.key_wz = 0.0
            self.height = HEIGHT_MAX
            self.roll = 0.0
            self.pitch = 0.0
            self.scale = 1.0
            self.estop_active = False

        if pressed(cfg.BTN_ESTOP):
            self.estop_active = not self.estop_active

        if pressed(cfg.BTN_SCALE_UP):
            self.scale = min(cfg.SCALE_MAX, self.scale + cfg.SCALE_STEP)

        if pressed(cfg.BTN_SCALE_DN):
            self.scale = max(cfg.SCALE_MIN, self.scale - cfg.SCALE_STEP)

        if pressed(cfg.BTN_PUSH):
            self.push_pub.publish(Empty())

        if pressed(cfg.BTN_VIZ):
            self.viz_pub.publish(Empty())

    # -- Keyboard input -----------------------------------------------------
    def handle_key(self, key: str) -> bool:
        """Process one key press. Returns False on Ctrl+C."""
        if key == '\x03':
            return False
        if key == 'j':
            self.mode = "JOY"
        elif key == 'k':
            self.mode = "KEY"
            self.key_vx = self.key_vy = self.key_wz = 0.0
        elif self.mode == "KEY":
            if key == ' ':
                self.key_vx = self.key_vy = self.key_wz = 0.0
                self.height = HEIGHT_MAX
                self.roll = 0.0
                self.pitch = 0.0
            elif key.lower() in KEY_BINDINGS:
                axis, delta = KEY_BINDINGS[key.lower()]
                if axis == 'h':
                    self.height = clamp(self.height + delta, HEIGHT_MIN, HEIGHT_MAX)
                elif axis == 'roll':
                    self.roll = clamp(self.roll + delta, ROLL_MIN, ROLL_MAX)
                elif axis == 'pitch':
                    self.pitch = clamp(self.pitch + delta, PITCH_MIN, PITCH_MAX)
                elif axis == 'vx':
                    self.key_vx = clamp(self.key_vx + delta, -KEY_MAX_V, KEY_MAX_V)
                elif axis == 'vy':
                    self.key_vy = clamp(self.key_vy + delta, -KEY_MAX_V, KEY_MAX_V)
                elif axis == 'wz':
                    self.key_wz = clamp(self.key_wz + delta, -KEY_MAX_WZ, KEY_MAX_WZ)
        return True

    # -- Build & publish command --------------------------------------------
    def publish_cmd(self):
        self.joy_alive = (
            (time.monotonic() - self._last_joy_time) <= self.cfg.timeout_s
            and self._last_joy_time > 0.0
        )

        msg = Twist()

        # Update height from right stick Y (rate control)
        if self.mode == "JOY" and self.joy_alive and abs(self.joy_height_rate) > 0.001:
            self.height = clamp(
                self.height + self.joy_height_rate / PUB_HZ,
                HEIGHT_MIN, HEIGHT_MAX,
            )

        if self.estop_active:
            msg.linear.z  = self.height  # keep height even in estop
            msg.angular.x = self.roll
            msg.angular.y = self.pitch
        elif self.mode == "JOY" and self.joy_alive and self.deadman_held:
            msg.linear.x  = float(self.joy_vx)
            msg.linear.y  = float(self.joy_vy)
            msg.angular.z = float(self.joy_wz)
            msg.linear.z  = self.height
            msg.angular.x = self.roll
            msg.angular.y = self.pitch
        elif self.mode == "KEY":
            msg.linear.x  = self.key_vx
            msg.linear.y  = self.key_vy
            msg.angular.z = self.key_wz
            msg.linear.z  = self.height
            msg.angular.x = self.roll
            msg.angular.y = self.pitch

        self.cmd_pub.publish(msg)
        return msg

    # -- Status line --------------------------------------------------------
    def status_line(self, msg: Twist) -> str:
        CLR = "\033[2K\r"
        vx, vy, wz, h = msg.linear.x, msg.linear.y, msg.angular.z, self.height
        r, p = self.roll, self.pitch
        vals = f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}  h={h:.2f}  r={r:+.2f}  p={p:+.2f}"
        if self.estop_active:
            return f"{CLR}  \033[31m[ESTOP]\033[0m  {vals}"
        if self.mode == "JOY":
            dot = "\033[32m●\033[0m" if self.joy_alive else "\033[31m○\033[0m"
            scl = f"  x{self.scale:.1f}" if abs(self.scale - 1.0) > 0.01 else ""
            return f"{CLR}  \033[33m[JOY]\033[0m {dot}  {vals}{scl}"
        return f"{CLR}  \033[36m[KEY]\033[0m     {vals}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Guard: must be launched via wrapper
    if "WALKER_TELEOP_WRAPPER" not in os.environ:
        for svc in ("p73-joy-teleop@udp.service", "p73-joy-teleop@local.service"):
            try:
                r = subprocess.run(
                    ["systemctl", "is-active", "--quiet", svc],
                    capture_output=True, timeout=3,
                )
                if r.returncode == 0:
                    print(
                        f"\n[ERROR] {svc} is active.\n"
                        f"  Use:  walker-teleop\n"
                    )
                    sys.exit(1)
            except Exception:
                pass

    settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    node = WalkerTeleop()

    # Spin in background so /joy callbacks fire immediately
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print(BANNER)

    try:
        while rclpy.ok():
            key = get_key(settings, timeout=1.0 / PUB_HZ)
            if not node.handle_key(key):
                break
            msg = node.publish_cmd()
            sys.stdout.write(node.status_line(msg))
            sys.stdout.flush()
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
        print("\nStopped.")


if __name__ == '__main__':
    main()
