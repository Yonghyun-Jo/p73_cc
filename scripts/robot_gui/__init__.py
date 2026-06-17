"""robot_gui — p73_cc 자체 완결 ROS2↔웹 브리지.

로봇 PC는 p73_cc만 clone하므로 브리지와 그 의존성(descriptor/transport/fsm/robots)이
전부 이 패키지 안에 있다. 외부 의존: stdlib + rclpy + p73_msgs + (sibling) motion_cmd_publisher.

브랜치별 인터페이스: p73_cc 브랜치 = 연구 컨셉. 따라서 robots/p73_walker.yaml 은 현재
브랜치의 인터페이스를 선언하며, 브랜치를 바꾸면 자동으로 그 브랜치의 선언이 따라온다.

실행 (ROS2 sourced, 추가 pip 설치 없음):
    python3 -m robot_gui.bridge_node --robot p73_walker --host 0.0.0.0
    # (p73_cc/scripts 를 cwd 로 두거나 bridge_node.py 가 sys.path 를 보정)

UI(Streamlit)는 piene_automation 쪽에 있으며 이 브리지의 HTTP(/state,/descriptor,/cmd,
/motion/*)만 호출한다. UI는 p73_cc 코드를 import 하지 않는다.
"""
