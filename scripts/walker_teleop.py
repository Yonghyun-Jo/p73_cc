#!/usr/bin/env python3
"""
Walker keyboard teleop — publishes geometry_msgs/Twist to /p73/cmd_vel
Ported from TOCABI tocabi_teleop.py (same key mapping).

Keys:
  w / s  : vx  +0.1 / -0.1  (forward/backward)
  a / d  : vy  +0.1 / -0.1  (left/right strafe)
  q / e  : wz  +0.1 / -0.1  (rotate left/right)
  space  : reset all to zero
  Ctrl+C : quit

Usage:
  ros2 run p73_cc walker_teleop.py
  # or directly:
  python3 walker_teleop.py
"""

import sys
import tty
import termios
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

STEP = 0.1
MAX_V = 1.0
MAX_WZ = 0.6
PUB_HZ = 20.0

vx = 0.0
vy = 0.0
wz = 0.0
lock = threading.Lock()
running = True

KEY_BINDINGS = {
    'w': ('vx', +STEP),
    's': ('vx', -STEP),
    'a': ('vy', +STEP),
    'd': ('vy', -STEP),
    'q': ('wz', +STEP),
    'e': ('wz', -STEP),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def print_status():
    sys.stdout.write(
        f"\r  vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}   "
        "[ w/s: forward/back | a/d: left/right | q/e: rotate | space: stop | Ctrl+C: quit ]  "
    )
    sys.stdout.flush()


def keyboard_thread():
    global vx, vy, wz, running
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while running:
            ch = sys.stdin.read(1)
            if ch == '\x03':  # Ctrl+C
                running = False
                break
            with lock:
                if ch == ' ':
                    vx = vy = wz = 0.0
                elif ch.lower() in KEY_BINDINGS:
                    axis, delta = KEY_BINDINGS[ch.lower()]
                    if axis == 'vx':
                        vx = clamp(vx + delta, -MAX_V, MAX_V)
                    elif axis == 'vy':
                        vy = clamp(vy + delta, -MAX_V, MAX_V)
                    elif axis == 'wz':
                        wz = clamp(wz + delta, -MAX_WZ, MAX_WZ)
                else:
                    continue
            print_status()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    global running
    rclpy.init()
    node = Node('walker_teleop')
    pub = node.create_publisher(Twist, '/p73/cmd_vel', 10)

    print("\n=== Walker Keyboard Teleop ===")
    print(f"  max_v={MAX_V}  max_wz={MAX_WZ}  step={STEP}")
    print("  w/s : forward/back  |  a/d : left/right")
    print("  q/e : rotate        |  space : stop")
    print("  Ctrl+C : quit")
    print("==============================\n")

    t = threading.Thread(target=keyboard_thread, daemon=True)
    t.start()

    try:
        while running and rclpy.ok():
            msg = Twist()
            with lock:
                msg.linear.x = vx
                msg.linear.y = vy
                msg.angular.z = wz
            pub.publish(msg)
            print_status()
            rclpy.spin_once(node, timeout_sec=1.0 / PUB_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        # Send zero on exit
        msg = Twist()
        pub.publish(msg)
        node.destroy_node()
        rclpy.shutdown()
        print("\n")


if __name__ == '__main__':
    main()
