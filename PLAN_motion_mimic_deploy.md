# p73_cc Motion Mimic 전용 Deploy 계획서

## 목표
Walker motion mimic student policy (63D/frame × 10 history = 630D)를 MuJoCo sim-to-sim 및 real robot에서 동작하도록 p73_cc 수정.

**기존 locomotion 코드는 전부 교체** (범용 코드 X, motion-only).

---

## 1. Observation 비교

### 기존 Locomotion (47D/frame)
```
[0:3]   base_ang_vel                 3D
[3:6]   projected_gravity            3D
[6:9]   velocity_commands (vx,vy,vyaw) 3D
[9]     gait_phase_sin               1D
[10]    gait_phase_cos               1D
[11:23] motor_joint_pos (12 lower)   12D  (q - q_default)
[23:35] motor_joint_vel (12 lower)   12D  (clip ±30, /30)
[35:47] last_action (processed)      12D  (rl_action * 0.5, clip ±1.0)
```
obs function: `mdp.last_processed_action(action_name="joint_pos")` → `processed_actions = raw * 0.5`

### 새 Motion Mimic (63D/frame)
```
[0:3]   base_ang_vel                 3D
[3:6]   projected_gravity            3D
[6:25]  motion_cmd (19D)             19D  (root_vel_xy(2)+root_z(1)+roll(1)+pitch(1)+yaw_vel(1)+dof_pos(13))
[25:38] joint_pos (13 ALL)           13D  (q - q_default, WaistYaw 포함)
[38:51] joint_vel (13 ALL)           13D  (clip ±30, /30, WaistYaw 포함)
[51:63] last_action (raw)            12D  (rl_action 그대로, 스케일/클립 없음)
```
obs function: `mdp.last_action()` → `env.action_manager.action` = **raw network output**

### 핵심 차이점
| 항목 | Locomotion | Motion Mimic |
|------|-----------|-------------|
| command | vel_cmd 3D | motion_ref 19D |
| gait phase | sin+cos 2D | **삭제** |
| joint obs | 12D (lower only) | **13D (WaistYaw 포함)** |
| last_action | processed (raw×0.5) | **raw (변환 없음)** |
| frame 크기 | 47D | **63D** |

---

## 2. Action 비교

### 기존 Locomotion
```
action_scale = 0.5  (uniform)
clip = ±1.0
q_target = q_default + clip(rl_action * 0.5, ±1.0)
```

### 새 Motion Mimic
```
action_scale = per-joint (아래 참고)
clip = ±2.0  (network output 클립)
q_target = q_default + clip(rl_action, ±2.0) * per_joint_scale
         → 이후 joint limit으로 clamp
```

**Per-joint action scales** (mimic_env_cfg.py WALKER_ACTION_SCALE):
```
L_HipRoll:     0.319    R_HipRoll:     0.285
L_HipPitch:    0.626    R_HipPitch:    0.681
L_HipYaw:      0.429    R_HipYaw:      0.644
L_Knee:        1.141    R_Knee:        1.147
L_AnklePitch:  0.484    R_AnklePitch:  0.484
L_AnkleRoll:   0.231    R_AnkleRoll:   0.231
```

### 주의: `last_action` obs vs action 적용은 다른 값
- **obs에 넣는 값**: `rl_action_` 그대로 (raw network output)
- **PD target 계산**: `q_default + clip(rl_action, ±2.0) * scale[i]`
- 이 두 값이 다름! locomotion에서는 같았음 (둘 다 `rl_action * 0.5, clip ±1.0`)

---

## 3. 변경 불필요 (그대로 유지)

- **PD gains** (kp, kd): 동일
- **Torque bounds**: 동일
- **4-bar kinematics**: 동일
- **processNoise()**: 동일
- **Policy rate**: 50Hz (decimation=4, dt=0.005)
- **Joint default positions**: 동일
- **Joint position limits**: 동일
- **ONNX loading**: 거의 동일 (shape만 63*H로 변경)
- **History 관리**: 동일한 frame-major 구조

---

## 4. 수정 대상 파일 & 변경 내용

### 4.1 `include/cc.h`

#### Constants
```cpp
// 변경 전
static const int num_single_obs = 47;
// 변경 후
static const int num_single_obs = 63;
static const int num_motion_cmd = 19;
```

#### Action scale
```cpp
// 변경 전
double action_scale_ = 0.5;
// 변경 후 (per-joint)
double action_scales_[12] = {
    0.319, 0.626, 0.429, 1.141, 0.484, 0.231,  // L leg
    0.285, 0.681, 0.644, 1.147, 0.484, 0.231,  // R leg
};
```

#### Motion command
```cpp
// 추가: velocity command 대체
#include <std_msgs/msg/float64_multi_array.hpp>
std::mutex motion_cmd_mutex_;
std::array<double, 19> motion_cmd_{};
void motionCmdCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr motion_cmd_sub_;
```

#### 삭제 대상
```cpp
// 제거: locomotion 전용
double action_scale_ = 0.5;          // → action_scales_[12]로 대체
int gait_step_counter_ = 0;          // motion에서 사용 안 함
int gait_period_steps_ = 70;         // motion에서 사용 안 함
double cmd_zero_max_ = 1.0e-3;       // motion에서 사용 안 함
// vel_cmd 관련 변수/subscriber는 유지 가능 (혹시 모를 테스트용) 또는 삭제
```

### 4.2 `src/cc.cpp`

#### loadOnnX()
```cpp
// 변경: 47 → 63 체크
if (policy_obs_dim_ % num_single_obs != 0)
    throw std::runtime_error("[p73_cc] policy_obs dim must be divisible by 63.");
```

#### processObservation() — 전체 재작성
```
idx=0:  ang_vel_b (3D)
idx=3:  projected_gravity_b (3D)
idx=6:  motion_cmd_ (19D, mutex lock으로 읽기)
idx=25: joint_pos ALL 13D (q_noise_ - q_default_p73_, WaistYaw 포함)
idx=38: joint_vel ALL 13D (clip ±30, /30, WaistYaw 포함)
idx=51: last_action 12D (rl_action_ 그대로, RAW)
```

**삭제**: velocity_commands(3D), gait_sin, gait_cos, gait_step_counter 업데이트

#### feedforwardPolicy()
```cpp
// 변경 전: last_action_processed_ = clip(rl_action * 0.5, ±1.0)
// 변경 후: obs용 raw action 저장
for (int i = 0; i < num_action; i++)
    last_action_raw_(i) = rl_action_(i);  // 변수명 변경 권장
```

#### computeFast() — action → target
```cpp
// 변경 전
double dq = rl_action_(i) * action_scale_;
dq = minmax_cut(dq, -1.0, 1.0);

// 변경 후
double action_clipped = minmax_cut(rl_action_(i), -2.0, 2.0);
double dq = action_clipped * action_scales_[i];
```

#### startVelSubscriber()
```cpp
// 추가: motion_cmd subscriber
motion_cmd_sub_ = dc_.node_->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/p73/motion_cmd", 10, ...);
```

#### Debug dump / Logging
- Term names, dims를 63D 구조에 맞게 변경
- CSV header 업데이트 (cmd_vx,cmd_vy,cmd_vyaw,gait_sin,gait_cos → motion_cmd_0..18)
- joint_pos_rel 12 → 13

### 4.3 `CMakeLists.txt`
```cmake
find_package(std_msgs REQUIRED)
ament_target_dependencies(... std_msgs ...)
```

### 4.4 `scripts/export_full_policy.py`
```bash
# 사용법만 변경 (이미 --num-single-obs 파라미터 지원)
python export_full_policy.py \
  --checkpoint /path/to/model_XXXX.pt \
  --num-single-obs 63 \
  --history-length 10 \
  --output ~/ros2_ws/src/p73_cc/policy/policy.onnx
```

---

## 5. Motion Command 입력 방식

### ROS2 토픽
- 토픽: `/p73/motion_cmd`
- 타입: `std_msgs/Float64MultiArray` (19D)
- 레이아웃:
  ```
  data[0:2]   root_vel_xy (local yaw-aligned frame)
  data[2]     root_pos_z (height)
  data[3]     roll (rad)
  data[4]     pitch (rad)
  data[5]     yaw_angular_velocity (rad/s)
  data[6:19]  dof_pos (13 joints, ALL_JOINT_NAMES order)
  ```

### MuJoCo sim-to-sim 시나리오
Motion clip에서 command를 계산하여 publish하는 별도 노드가 필요.
(Isaac Lab 학습 시 `calc_current_motion_command_proprio()`와 동일한 로직)

### Real robot 시나리오
모션 플래너 또는 사전 녹화된 motion trajectory에서 publish.

---

## 6. 검증 체크리스트

- [ ] ONNX export: `--num-single-obs 63 --history-length 10` → 630D input
- [ ] MuJoCo sim: 첫 5 step obs 값이 Isaac Lab play와 일치하는지 JSONL 비교
- [ ] Action → PD target: `q_default + clip(action, ±2.0) * scale[i]` 계산 일치
- [ ] last_action obs: raw rl_action_ (no scale/clip) 확인
- [ ] WaistYaw: obs에는 포함 (13D), action에는 미포함 (12D), PD는 default 유지
- [ ] 19D motion_cmd가 0일 때 로봇이 default pose 유지하는지 확인
