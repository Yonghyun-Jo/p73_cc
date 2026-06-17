# robot_gui — p73_cc 자체 완결 ROS2↔웹 브리지

로봇 PC는 **p73_cc만 clone**하므로 브리지 일체가 이 패키지 안에 있다.
의존성: 파이썬 stdlib + rclpy + p73_msgs + (sibling) `../motion_cmd_publisher.py`.
**추가 pip 설치 없음** (HTTP는 `http.server`). PEP 668 시스템 python에서 그대로 실행.

## 구성

```
robot_gui/
  descriptor.py     robot.yaml 로더/검증 (순수). {PKG}=p73_cc 루트 치환.
  transport.py      모션 재생 상태머신 (순수)
  fsm.py            제어 상태머신 + 버튼 게이팅 (순수)
  bridge_node.py    rclpy + http.server. ClipLoader=sibling motion_cmd_publisher.
  robots/p73_walker.yaml   현재 브랜치의 인터페이스 선언
  tests/            순수 로직 테스트 (37, ROS2 불필요)
```

## 브랜치 = 인터페이스

p73_cc 브랜치가 곧 연구 컨셉(input/output 다름). `robots/p73_walker.yaml` 이 브랜치와 함께
운반되므로, 브랜치를 checkout하면 그 브랜치의 인터페이스 선언이 자동으로 따라온다.
한 브랜치 안에서 sub-variant 가 필요하면 `profiles/<name>.yaml`(`robot:`+`branch:` 참조) 추가.

## 실행

```bash
source ~/ros2_ws/install_motion/setup.bash    # 브랜치 대응 install 트리
cd ~/ros2_ws/src/p73_cc/scripts                # robot_gui 패키지가 보이게
python3 -m robot_gui.bridge_node --robot p73_walker --host 0.0.0.0
#   → http://0.0.0.0:8600  (/state /descriptor /cmd /motion/*)
# (직접 실행도 가능: python3 robot_gui/bridge_node.py ... — sys.path 자가 보정)
```

⚠ `simulation.launch.py` 는 `motion_gui.py` 를 자동 기동해 같은 `/p73/motion_cmd` 로
발행한다. 브리지로 모션을 쏠 땐 motion_gui 가 떠 있으면 안 됨(이중 발행 충돌) →
`simulation_headless.launch.py` 사용 or motion_gui 제외.

## 테스트

```bash
cd ~/ros2_ws/src/p73_cc/scripts/robot_gui && python3 -m pytest tests/ -q
```

UI(Streamlit)는 piene_automation 쪽에 있고 이 브리지의 HTTP만 호출한다.
