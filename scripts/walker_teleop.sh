#!/usr/bin/env bash
# Unified Walker Teleop — THE ONLY way to command the robot.
#
# Launches:
#   1. joy_udp_receiver  (Windows joystick via UDP)
#   2. joy_node          (if local 8BitDo dongle detected)
#   3. walker_teleop.py  (unified keyboard + joystick teleop)
#
# On exit: all subprocesses killed, joy services stay STOPPED.
# The robot cannot be moved without running this script again.
#
# Usage:
#   walker-teleop
#   # alias: echo "alias walker-teleop='bash ~/ros2_ws/src/p73_cc/scripts/walker_teleop.sh'" >> ~/.bashrc

set -e

SERVICE_UDP="p73-joy-teleop@udp.service"
SERVICE_LOCAL="p73-joy-teleop@local.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TELEOP_PY="${SCRIPT_DIR}/walker_teleop.py"
UDP_RECEIVER="${SCRIPT_DIR}/joy_udp_receiver.py"

# --- env hygiene: kick out conda, force system PATH so python3 = 3.12 -----
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONPATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# --- ROS env (auto-detect distro) ------------------------------------------
ROS_FOUND=0
for d in /opt/ros/jazzy /opt/ros/humble /opt/ros/iron /opt/ros/rolling; do
    if [[ -f "$d/setup.bash" ]]; then
        # shellcheck disable=SC1091
        source "$d/setup.bash"
        ROS_FOUND=1
        break
    fi
done
if [[ "$ROS_FOUND" -eq 0 ]]; then
    echo "[walker-teleop] ERROR: no ROS2 found in /opt/ros/. Install ROS2 first."
    exit 1
fi
# shellcheck disable=SC1091
source "${HOME}/ros2_ws/install/setup.bash"

# --- stop joy systemd services (they must not run independently) -----------
for svc in "$SERVICE_UDP" "$SERVICE_LOCAL"; do
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        echo "[walker-teleop] stopping $svc..."
        if ! sudo -n systemctl stop "$svc" 2>/dev/null; then
            sudo systemctl stop "$svc"
        fi
    fi
done

# --- PIDs to track subprocesses -------------------------------------------
PIDS=()

cleanup() {
    echo
    echo "[walker-teleop] shutting down..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.2
            done
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "[walker-teleop] stopped. Robot is now idle (no cmd_vel publisher)."
}
trap cleanup EXIT INT TERM

# --- launch joy_udp_receiver (for Windows joystick bridge) ----------------
echo "[walker-teleop] starting joy_udp_receiver (UDP port 35731)..."
python3 "$UDP_RECEIVER" --ros-args \
    -p port:=35731 \
    -p bind:="0.0.0.0" \
    -p publish_rate_hz:=50.0 \
    &>/dev/null &
PIDS+=($!)

# --- launch joy_node if local 8BitDo dongle detected ----------------------
JOY_DEV=""
if [[ -e /dev/input/p73_joystick ]]; then
    JOY_DEV="/dev/input/p73_joystick"
else
    for js in /dev/input/js*; do
        [[ -e "$js" ]] || continue
        devname="$(cat "/sys/class/input/$(basename "$js")/device/name" 2>/dev/null)"
        if [[ "$devname" == *"8BitDo"* || "$devname" == *"8bitdo"* ]]; then
            JOY_DEV="$js"
            break
        fi
    done
fi

if [[ -n "$JOY_DEV" ]]; then
    echo "[walker-teleop] local joystick: $JOY_DEV"
    ros2 run joy joy_node --ros-args \
        -p device_name:="8BitDo Ultimate 2 Wireless Controller" \
        -p deadzone:=0.0 \
        -p autorepeat_rate:=0.0 \
        -p sticky_buttons:=false \
        -p coalesce_interval:=0.01 \
        &>/dev/null &
    PIDS+=($!)
else
    echo "[walker-teleop] no local dongle → UDP/Windows mode"
    export JOY_AXIS_WZ=2
    export JOY_INVERT_VX=1
    export JOY_INVERT_VY=1
    export JOY_INVERT_WZ=1
    export JOY_REQUIRE_DEADMAN=0
fi

sleep 0.5

# --- run unified teleop ---------------------------------------------------
export WALKER_TELEOP_WRAPPER=1
python3 "$TELEOP_PY"
