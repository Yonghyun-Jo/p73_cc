"""descriptor.py — 로봇 capability descriptor 로더/검증 (순수 로직).

robots/<name>.yaml 한 장이 한 로봇의 ROS2 제어 인터페이스를 선언한다. p73_cc 브랜치가
곧 연구 컨셉이므로, 이 yaml 은 현재 브랜치의 인터페이스를 운반한다(브랜치 = 인터페이스).
profile 오버레이는 한 브랜치 안에서 sub-variant 가 필요할 때만 선택적으로 쓴다.

경로에 `{PKG}` 를 쓰면 p73_cc 패키지 루트로 치환된다 → 로봇 PC clone 경로 무관.
ROS2 의존성 없음(순수 yaml/dataclass) → 단위 테스트 가능.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve()
ROBOTS_DIR = _HERE.parent / "robots"
PROFILES_DIR = _HERE.parent / "profiles"
# robot_gui → scripts → p73_cc(루트). {PKG} 치환에 사용.
PKG_ROOT = _HERE.parents[2]


class DescriptorError(ValueError):
    """robot.yaml 이 스키마를 위반할 때."""


def _expand(s: str) -> str:
    return s.replace("{PKG}", str(PKG_ROOT)) if isinstance(s, str) else s


@dataclass(frozen=True)
class ControlAction:
    topic: str
    type: str
    payload: Any = None
    field: str | None = None
    fields: dict | None = None   # 고정 다중필드 메시지 (예: PosCmd position+traj_time)

    def build_message(self, value: Any = None) -> dict:
        if self.fields is not None:
            return dict(self.fields)            # 고정 다중필드 (init_pose 등)
        if self.field is not None:
            if value is None:
                raise DescriptorError(f"action(topic={self.topic})는 value가 필요합니다")
            return {self.field: value}
        if self.type.endswith("String"):
            return {"data": self.payload}
        return {"data": self.payload} if self.payload is not None else {}


@dataclass(frozen=True)
class Mode:
    id: int
    label: str


@dataclass(frozen=True)
class MotionSource:
    topic: str
    type: str
    rate_hz: int
    dirs: list[str]
    ext: str
    layout: list[str] = field(default_factory=list)

    def list_clips(self) -> list[Path]:
        seen: dict[str, Path] = {}
        for d in self.dirs:
            p = Path(_expand(d)).expanduser()
            if not p.is_dir():
                continue
            for f in sorted(p.glob(f"*{self.ext}")):
                seen.setdefault(f.name, f)
        return [seen[k] for k in sorted(seen)]


@dataclass(frozen=True)
class RobotDescriptor:
    name: str
    display_name: str
    controls: dict[str, ControlAction]
    modes: list[Mode]
    state_topics: dict[str, dict]
    joint_names: list[str]
    motion: MotionSource | None
    viz: dict
    safety: dict
    connection: dict
    profile: str | None = None
    branch: str | None = None
    install_tree: str | None = None
    io: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    def control(self, action: str) -> ControlAction:
        if action not in self.controls:
            raise DescriptorError(
                f"'{self.name}'에 control 액션 '{action}' 없음. 가능: {sorted(self.controls)}")
        return self.controls[action]

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def list_motions(self) -> list[Path]:
        return self.motion.list_clips() if self.motion else []

    def to_public_dict(self) -> dict:
        """UI(/descriptor)로 내보낼 JSON 친화 dict — UI는 이것만 보고 렌더."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "profile": self.profile,
            "branch": self.branch,
            "install_tree": self.install_tree,
            "modes": [{"id": m.id, "label": m.label} for m in self.modes],
            "joints": list(self.joint_names),
            "joint_urdf_map": (self.raw.get("joints") or {}).get("urdf_map") or {},
            "motion": ({"rate_hz": self.motion.rate_hz, "layout": self.motion.layout}
                       if self.motion else None),
            "viz": {k: _expand(v) for k, v in (self.viz or {}).items()},
            "io": self.io,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RobotDescriptor":
        _require(d, ["name", "controls", "joints"], where="robot descriptor")

        controls = {}
        for action, spec in (d.get("controls") or {}).items():
            _require(spec, ["topic", "type"], where=f"controls.{action}")
            controls[action] = ControlAction(
                topic=spec["topic"], type=spec["type"],
                payload=spec.get("payload"), field=spec.get("field"),
                fields=spec.get("fields"))

        modes = [Mode(id=int(m["id"]), label=str(m["label"]))
                 for m in (d.get("modes") or [])]

        joints = d["joints"].get("names") if isinstance(d["joints"], dict) else d["joints"]
        if not joints:
            raise DescriptorError("joints.names 가 비어있음")

        motion = None
        if d.get("motion"):
            m = d["motion"]
            _require(m, ["topic", "type"], where="motion")
            motion = MotionSource(
                topic=m["topic"], type=m["type"], rate_hz=int(m.get("rate_hz", 50)),
                dirs=list(m.get("dirs") or []), ext=m.get("ext", ".pkl"),
                layout=list(m.get("layout") or []))

        return cls(
            name=d["name"], display_name=d.get("display_name", d["name"]),
            controls=controls, modes=modes,
            state_topics=dict(d.get("state") or {}), joint_names=list(joints),
            motion=motion, viz=dict(d.get("viz") or {}), safety=dict(d.get("safety") or {}),
            connection=dict(d.get("connection") or {}),
            profile=d.get("profile"), branch=d.get("branch"),
            install_tree=d.get("install_tree"), io=dict(d.get("io") or {}), raw=d)


def _require(d: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise DescriptorError(f"{where}: 필수 키 누락 {missing}")


# ── 디스크 로더 ──
def _load_yaml(p: Path) -> dict:
    if not p.is_file():
        raise DescriptorError(f"파일 없음: {p}")
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise DescriptorError(f"{p}: yaml 최상위가 매핑이 아님")
    return data


_OVERLAY_KEYS = ("modes", "motion", "state", "viz", "safety", "joints")


def merge_robot_profile(robot_d: dict, profile_d: dict) -> dict:
    """robot 위에 profile(브랜치 내 sub-variant) 오버레이 병합."""
    merged = dict(robot_d)
    if profile_d.get("controls"):
        c = dict(robot_d.get("controls") or {})
        c.update(profile_d["controls"])
        merged["controls"] = c
    for k in _OVERLAY_KEYS:
        if k in profile_d:
            merged[k] = profile_d[k]
    merged["profile"] = profile_d.get("name")
    merged["display_name"] = profile_d.get("display_name", robot_d.get("display_name"))
    for meta in ("branch", "install_tree", "io"):
        if meta in profile_d:
            merged[meta] = profile_d[meta]
    return merged


def load_descriptor(robot: str | Path, profile: str | None = None,
                    robots_dir: Path = ROBOTS_DIR,
                    profiles_dir: Path = PROFILES_DIR) -> RobotDescriptor:
    p = Path(robot)
    rp = p if p.suffix else robots_dir / f"{robot}.yaml"
    robot_d = _load_yaml(rp)
    if profile is None:
        return RobotDescriptor.from_dict(robot_d)
    pp = Path(profile)
    pfp = pp if pp.suffix else profiles_dir / f"{profile}.yaml"
    profile_d = _load_yaml(pfp)
    ref = profile_d.get("robot")
    if ref and ref != robot_d.get("name"):
        raise DescriptorError(
            f"profile '{profile}'는 robot '{ref}' 참조인데 '{robot_d.get('name')}'로 로드됨")
    return RobotDescriptor.from_dict(merge_robot_profile(robot_d, profile_d))


def list_robots(robots_dir: Path = ROBOTS_DIR) -> list[str]:
    if not robots_dir.is_dir():
        return []
    return sorted(p.stem for p in robots_dir.glob("*.yaml"))


def list_profiles(robot: str | None = None, profiles_dir: Path = PROFILES_DIR) -> list[str]:
    if not profiles_dir.is_dir():
        return []
    out = []
    for p in sorted(profiles_dir.glob("*.yaml")):
        if robot is None:
            out.append(p.stem)
            continue
        try:
            d = yaml.safe_load(p.read_text())
            if isinstance(d, dict) and d.get("robot") == robot:
                out.append(p.stem)
        except Exception:
            continue
    return out
