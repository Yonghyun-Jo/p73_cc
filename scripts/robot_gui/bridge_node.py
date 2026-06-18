"""bridge_node.py — ROS2 ↔ 웹 브리지 (rclpy + stdlib http.server). ROS2 env에서 실행.

웹은 이산 이벤트(intent)만 보내고, 실시간 모션 스트림(50Hz)은 이 브리지 안에서만 발행한다.
순수 로직은 transport/fsm/descriptor(테스트됨). 이 파일은 rclpy + HTTP 배선 + ClipLoader.

실행 (ROS2 sourced — 추가 pip 설치 없음):
    python3 -m robot_gui.bridge_node --robot p73_walker --host 0.0.0.0
    # 또는: python3 <p73_cc>/scripts/robot_gui/bridge_node.py --robot p73_walker
HTTP는 stdlib 만 사용 → PEP 668 시스템 python에서 그대로 실행.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# scripts/ 를 path 에 → robot_gui.* 및 sibling motion_cmd_publisher import 가능
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from robot_gui.descriptor import RobotDescriptor, load_descriptor  # noqa: E402
from robot_gui.fsm import ControlFSM  # noqa: E402
from robot_gui.transport import MotionTransport  # noqa: E402

_MSG_REGISTRY: dict[str, tuple[str, str]] = {
    "std_msgs/String": ("std_msgs.msg", "String"),
    "std_msgs/Empty": ("std_msgs.msg", "Empty"),
    "std_msgs/UInt32": ("std_msgs.msg", "UInt32"),
    "std_msgs/Float32": ("std_msgs.msg", "Float32"),
    "std_msgs/Float64MultiArray": ("std_msgs.msg", "Float64MultiArray"),
    "std_msgs/Int8MultiArray": ("std_msgs.msg", "Int8MultiArray"),
    "sensor_msgs/JointState": ("sensor_msgs.msg", "JointState"),
    "geometry_msgs/Pose": ("geometry_msgs.msg", "Pose"),
    "geometry_msgs/Twist": ("geometry_msgs.msg", "Twist"),
    "p73_msgs/TaskCmd": ("p73_msgs.msg", "TaskCmd"),
    "p73_msgs/PosCmd": ("p73_msgs.msg", "PosCmd"),
    "p73_msgs/IKTaskCmd": ("p73_msgs.msg", "IKTaskCmd"),
}

STATE_TIMEOUT_S = 1.5


def resolve_msg_class(type_str: str):
    if type_str not in _MSG_REGISTRY:
        raise KeyError(f"미등록 메시지 타입: {type_str} (_MSG_REGISTRY에 추가)")
    mod_name, cls = _MSG_REGISTRY[type_str]
    return getattr(importlib.import_module(mod_name), cls)


class ClipLoader:
    """pkl 클립 → num_frames/fps + 프레임별 33D + standing.

    sibling motion_cmd_publisher.MotionClip 재사용 → 33D 포맷 단일 소스.
    """
    def __init__(self, path: str):
        from motion_cmd_publisher import MotionClip  # scripts/ 에 있음
        self.clip = MotionClip(path)
        self.fps = float(getattr(self.clip, "fps", 50.0))
        self.num_frames = int(getattr(self.clip, "num_frames",
                                      getattr(self.clip, "n_frames", 0)))

    def frame_33d(self, idx: int) -> list[float]:
        q_ref, qd_ref, root_pos, root_quat = self.clip.frame_at_time(idx / self.fps, loop=True)
        return [*q_ref, *qd_ref, *root_pos, *root_quat]

    def standing_33d(self) -> list[float]:
        q_ref, _qd, root_pos, root_quat = self.clip.frame_at_time(0.0, loop=False)
        return [*q_ref, *([0.0] * len(q_ref)), *root_pos, *root_quat]


class RobotBridge:
    def __init__(self, descriptor: RobotDescriptor, clip_loader_cls: type = ClipLoader):
        self.d = descriptor
        rate = descriptor.motion.rate_hz if descriptor.motion else 50
        self.transport = MotionTransport(fps=rate)
        self.fsm = ControlFSM()
        self.lock = threading.RLock()
        self._clip_loader_cls = clip_loader_cls
        self.clip = None
        self.clip_name: str | None = None
        self.state_cache: dict[str, Any] = {}
        self.joint_buffer: deque = deque(maxlen=600)
        self.status_tail: deque = deque(maxlen=50)
        self.sim_time: float | None = None
        self._last_sim_time: float | None = None
        self._last_state_t: float = 0.0
        self.base_quat = None        # base(pelvis) 자세 [x,y,z,w] — 3D 뷰가 기울기 표시에 사용
        self.node = None
        self._pubs: dict[str, Any] = {}

    def _now(self) -> float:
        return time.monotonic()

    def _refresh_connection(self) -> None:
        self.fsm.set_connected((self._now() - self._last_state_t) < STATE_TIMEOUT_S)

    def send_command(self, action: str, value: Any = None) -> dict:
        with self.lock:
            self._refresh_connection()
            if not self.fsm.can(action):
                raise PermissionError(f"'{action}' 불가 (상태={self.fsm.state}). 선행 단계 필요.")
            ca = self.d.control(action)
            self._publish(ca.topic, ca.type, ca.build_message(value))
            if action == "torque_on":
                self.fsm.set_torque(True)
            elif action == "torque_off":
                self.fsm.set_torque(False)
            elif action == "set_mode":
                self.fsm.set_mode_running(int(value) >= 5)
            return {"ok": True, "action": action, "topic": ca.topic, "state": self.fsm.state}

    def motion_select(self, path: str) -> dict:
        with self.lock:
            self.clip = self._clip_loader_cls(path)
            self.clip_name = path.rsplit("/", 1)[-1]
            self.transport.select(self.clip.num_frames, self.clip.fps)
            return {"ok": True, "clip": self.clip_name, "num_frames": self.clip.num_frames}

    def motion_play(self) -> dict:
        with self.lock:
            if not self.fsm.can("play"):
                raise PermissionError(f"play 불가 (상태={self.fsm.state})")
            self.transport.play()
            return self._motion_status()

    def motion_pause(self) -> dict:
        with self.lock:
            self.transport.pause(); return self._motion_status()

    def motion_stop(self) -> dict:
        with self.lock:
            self.transport.stop(); return self._motion_status()

    def motion_seek(self, frame: int) -> dict:
        with self.lock:
            self.transport.seek(frame); return self._motion_status()

    def motion_loop(self, loop: bool) -> dict:
        with self.lock:
            self.transport.set_loop(loop); return self._motion_status()

    def _motion_status(self) -> dict:
        return {"clip": self.clip_name, "playing": self.transport.playing,
                "loop": self.transport.loop, "current_frame": self.transport.current_frame,
                "num_frames": self.transport.num_frames}

    def _motion_tick(self) -> None:
        with self.lock:
            if not self.fsm.mode_running or self.clip is None:
                return
            rate = self.d.motion.rate_hz if self.d.motion else 50
            dt = 1.0 / rate
            if self.sim_time is not None and self._last_sim_time is not None:
                sd = self.sim_time - self._last_sim_time
                if sd > 1e-6:
                    dt = sd
            self._last_sim_time = self.sim_time
            self.transport.advance(dt)
            data = (self.clip.standing_33d() if self.transport.emitting_standing
                    else self.clip.frame_33d(self.transport.current_frame))
            if self.d.motion:
                self._publish(self.d.motion.topic, self.d.motion.type, {"data": data})

    def snapshot(self) -> dict:
        with self.lock:
            self._refresh_connection()
            return {
                "robot": self.d.name, "profile": self.d.profile, "branch": self.d.branch,
                "install_tree": self.d.install_tree, "fsm": self.fsm.state,
                "connected": self.fsm.connected, "torque_on": self.fsm.torque_on,
                "mode_running": self.fsm.mode_running, "motion": self._motion_status(),
                "joints": list(self.joint_buffer)[-1] if self.joint_buffer else None,
                "base_quat": self.base_quat,
                "status_tail": list(self.status_tail), "sim_time": self.sim_time,
                "can": {a: self.fsm.can(a)
                        for a in ("torque_on", "init_pose", "set_mode", "play")},
            }

    def _publish(self, topic: str, type_str: str, fields: dict) -> None:
        pub = self._pubs.get(topic)
        if pub is None:
            raise RuntimeError(f"퍼블리셔 없음: {topic} (setup_ros 먼저)")
        msg = resolve_msg_class(type_str)()
        for k, v in fields.items():
            setattr(msg, k, v)
        pub.publish(msg)

    def setup_ros(self, node) -> None:
        self.node = node
        specs = [(ca.topic, ca.type) for ca in self.d.controls.values()]
        if self.d.motion:
            specs.append((self.d.motion.topic, self.d.motion.type))
        for topic, type_str in specs:
            if topic not in self._pubs:
                self._pubs[topic] = node.create_publisher(resolve_msg_class(type_str), topic, 10)
        for name, spec in self.d.state_topics.items():
            cls = resolve_msg_class(spec["type"])
            node.create_subscription(cls, spec["topic"],
                                     lambda msg, n=name: self._on_state(n, msg), 10)
        sim_topic = self.d.viz.get("sim_time_topic", "/p73/mjcSimTime")
        try:
            from std_msgs.msg import Float32
            node.create_subscription(Float32, sim_topic,
                                     lambda m: setattr(self, "sim_time", float(m.data)), 10)
        except Exception:
            pass
        rate = self.d.motion.rate_hz if self.d.motion else 50
        node.create_timer(1.0 / rate, self._motion_tick)

    def _on_state(self, name: str, msg) -> None:
        with self.lock:
            self._last_state_t = self._now()
            if name == "status_log":
                self.status_tail.append(str(getattr(msg, "data", "")))
            elif name == "joint":
                pos = list(getattr(msg, "position", []))
                if pos:
                    self.joint_buffer.append(pos)
            elif name == "imu_pose":
                o = getattr(msg, "orientation", None)
                if o is not None:
                    self.base_quat = [o.x, o.y, o.z, o.w]   # geometry_msgs/Quaternion xyzw
            else:
                self.state_cache[name] = _simple_serialize(msg)


def _simple_serialize(msg) -> Any:
    for attr in ("data", "position"):
        if hasattr(msg, attr):
            v = getattr(msg, attr)
            return list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v
    return str(msg)


# ── HTTP 서버 (stdlib) ──
def make_handler(bridge: RobotBridge):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # CORS: 브라우저 3D 뷰어가 /state 를 직접 폴링(다른 포트=cross-origin)하도록 허용.
            # 응답 헤더일 뿐 — ROS2 제어 경로/타이밍과 무관(별도 스레드·별도 프로세스).
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _guard(self, fn) -> None:
            try:
                self._send(200, fn())
            except (PermissionError, KeyError, ValueError) as e:
                self._send(409, {"detail": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"detail": str(e)})

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return {}

        def do_GET(self):  # noqa: N802
            if self.path == "/state":
                self._guard(bridge.snapshot)
            elif self.path == "/descriptor":
                self._guard(bridge.d.to_public_dict)
            elif self.path == "/motions":
                self._guard(lambda: {"clips": [str(p) for p in bridge.d.list_motions()]})
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self):  # noqa: N802
            p = self.path
            if p == "/cmd":
                b = self._body()
                self._guard(lambda: bridge.send_command(b.get("action"), b.get("value")))
            elif p == "/motion/select":
                b = self._body()
                self._guard(lambda: bridge.motion_select(b.get("path")))
            elif p == "/motion/play":
                self._guard(bridge.motion_play)
            elif p == "/motion/pause":
                self._guard(bridge.motion_pause)
            elif p == "/motion/stop":
                self._guard(bridge.motion_stop)
            elif p.startswith("/motion/seek/"):
                self._guard(lambda: bridge.motion_seek(int(p.rsplit("/", 1)[-1])))
            elif p.startswith("/motion/loop/"):
                on = p.rsplit("/", 1)[-1].lower() in ("1", "true", "on", "yes")
                self._guard(lambda: bridge.motion_loop(on))
            else:
                self._send(404, {"detail": "not found"})

    return Handler


def serve(bridge: RobotBridge, host: str, port: int) -> None:
    ThreadingHTTPServer((host, port), make_handler(bridge)).serve_forever()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="p73_cc robot_gui ROS2↔웹 브리지")
    ap.add_argument("--robot", default="p73_walker")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    args = ap.parse_args(argv)

    descriptor = load_descriptor(args.robot, profile=args.profile)
    bridge = RobotBridge(descriptor)

    import rclpy
    rclpy.init()
    node = rclpy.create_node("robot_gui_bridge")
    bridge.setup_ros(node)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print(f"[bridge] {descriptor.name}"
          + (f"/{descriptor.profile}" if descriptor.profile else "")
          + f" → http://{args.host}:{args.port}  (branch={descriptor.branch})")
    serve(bridge, args.host, args.port)


if __name__ == "__main__":
    main()
