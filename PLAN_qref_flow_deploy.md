# p73_cc — q_ref Flow Student Deploy Plan (FINAL)

> 대상: `isaaclab_walker_motion` flow student (q_ref command, obs 80D, flow euler).
> 학습 ckpt: `2026-06-17_00-22-29_flow_2teacher_qref/model_14000.pt` (bc_loss 0.045, time_out 83%).
> 목표: **학습값을 MuJoCo sim-to-sim → real robot에 bit-exact deploy.** 품질보다 **sim2real OOD 경계 우선.**
> branch: p73_cc `walker_motion`. anchor 메커니즘 상세 → `isaaclab_walker_motion/md_files/anchor.md`.
> ⚠ **재학습 불필요**: obs 재구조(command 26D + anchor 9D 분리)는 obs 벡터 byte-identical → 학습 정책(encoder+flow actor) 그대로 유효. ONNX export는 critic 미사용.

---

## 0. 핵심 원리 (반드시 먼저)

1. **command(q_ref+qd_ref)와 anchor를 분리.** obs[6:32]=command(publisher 순수 레퍼런스), obs[32:41]=anchor(cc.cpp가 `q_virtual_`로 계산). env가 이렇게 재구조돼 deploy 분담과 1:1 일치.
2. **anchor는 추종오차**(ref vs robot) → publisher 불가, **cc.cpp가 계산**.
3. **qd_ref는 pkl에 없음** → dof_pos **중심차분(dt=1/fps)** 으로 생성(§3). motion_lib와 동일식이어야 bit-exact.
4. **재anchor ≠ 주기적 0 리셋**(그건 OOD 톱니). 학습은 RSI 시작 정렬 후 연속 작은 오차 → MuJoCo는 재anchor 최소, real은 odometry 드리프트만 leaky 흡수(§5).

---

## 1. Observation 구조 (80D, bit-exact)
```
idx      내용                              스케일(cc.cpp)   출처
[0:3]    base_ang_vel (body, rad/s)        RAW             로봇 IMU/state
[3:6]    projected_gravity (body)          RAW             로봇 IMU/state
[6:19]   q_ref  (13, ALL_JOINT order, rad) RAW             publisher
[19:32]  qd_ref (13, ALL_JOINT order)      RAW (ONNX ×1/30) publisher (중심차분)
[32:35]  motion_root_pos_b (3, m)          RAW             cc.cpp 계산
[35:41]  motion_root_ori_b (6, Rot6D)      RAW             cc.cpp 계산
[41:54]  joint_pos_rel (13) = q - q_default RAW            로봇
[54:67]  joint_vel (13) = clip(qd,±30)/30   /30 (cc.cpp)    로봇
[67:80]  last_action (13) = rl_action raw   RAW             직전 정책출력
```
× history 10 = **800D** ONNX 입력.

### ⚠ Scaling 규칙
- ONNX(FlowExportWrapper)가 `policy_obs_scale`을 **bake** → **command(q_ref/qd_ref)·anchor는 cc.cpp가 RAW**, qd_ref ×1/30은 ONNX 내부.
- **proprio joint_vel[54:67]만 cc.cpp가 /30**(ObsTerm scale). command의 qd_ref(RAW)와 혼동 금지.

---

## 2. Anchor error 계산 (cc.cpp, **isaaclab 규약** — anchor.md §7)
```cpp
robot_pos  = q_virtual_.head<3>();                  // base pos
robot_quat = wxyz(q_virtual_.segment<4>(3));        // xyzw → wxyz
// ref_pos_a, ref_quat_a = anchor 변환 적용된 레퍼런스(§5)
// [32:35] motion_root_pos_b
e_anchor_pos = quat_rotate_inverse(robot_quat, ref_pos_a - robot_pos);   // R(robot)^T·Δ
// [35:41] motion_root_ori_b (Rot6D), isaaclab = ref ⊗ robot^-1 (WORLD)
q_rel = quat_mul(ref_quat_a, quat_conjugate(robot_quat));
//   w,x,y,z = q_rel
col0 = {1-2(y²+z²), 2(xy+wz), 2(xz-wy)};
col1 = {2(xy-wz), 1-2(x²+z²), 2(yz+wx)};
e_anchor_ori = {col0, col1};   // 6
```
- ⚠ ori는 **`quat_mul(ref, conj(robot))`** (world). mjlab(body-frame)과 다름 — 반드시 이 식.
- quat_rotate_inverse/quat_mul/quat_conjugate: isaaclab `isaaclab.utils.math` 규약(wxyz) Eigen 구현.

---

## 3. Publisher — `/p73/motion_ref` (33D 순수 레퍼런스)
`std_msgs/Float64MultiArray`, 50Hz:
```
[0:13]   q_ref      = dof_pos[t]  (ALL_JOINT_NAMES order, _motion_dof_reorder 적용)
[13:26]  qd_ref     = 중심차분(dof_pos, dt=1/fps)  ← pkl에 없으니 계산
[26:29]  ref_root_pos  (모션 프레임, m)
[29:33]  ref_root_quat (wxyz)
```
### qd_ref 중심차분 (motion_lib `_compute_velocity` 그대로)
```python
dt = 1.0/fps
qd = np.zeros_like(dof_pos)
qd[1:-1] = (dof_pos[2:] - dof_pos[:-2]) / (2*dt)
qd[0]    = (dof_pos[1]  - dof_pos[0])  / dt
qd[-1]   = (dof_pos[-1] - dof_pos[-2]) / dt
```
- 사전 계산(클립 로드 시 전체) 권장 → 매 step qd[t] 인덱싱.
- ⚠ joint 순서 = ALL_JOINT_NAMES. pkl dof_pos가 sim 순서면 reorder 후 publish.
- anchor는 **안 보냄**(cc.cpp 계산). publisher는 로봇 상태 무관 → sim/real 공통.

---

## 4. cc.cpp / cc.h 변경

### `include/cc.h`
```cpp
static const int num_single_obs = 80;
static const int num_motion_cmd = 26;            // command(q_ref13+qd_ref13)만; anchor는 별도 9
// publisher 레퍼런스
std::array<double,33> motion_ref_{};             // q_ref13+qd_ref13+ref_root_pos3+ref_root_quat4(wxyz)
std::mutex motion_ref_mutex_;
rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr motion_ref_sub_;
// anchor 상태
Eigen::Vector3d    anchor_t_{Eigen::Vector3d::Zero()};   // T_a 평행이동
Eigen::Quaterniond anchor_q_{Eigen::Quaterniond::Identity()};  // T_a 회전
bool   anchor_inited_ = false;
double anchor_leak_alpha_ = 0.0;                  // MuJoCo=0, real>0 (저주파 xy/yaw 흡수)
double anchor_xy_guard_ = 0.20;                   // 하드가드 임계(m)
void computeAnchorError(double e9[9]);            // §2
void updateAnchor();                              // §5
```
- action_scales_[13], last_action_raw_, history, PD/torque/4-bar, loadOnnX(80*H) — **walker_vision branch와 동일·불변.**

### `src/cc.cpp`
- **콜백** `/p73/motion_ref` → `motion_ref_`(mutex).
- **`updateAnchor()`**(§5): 시작 full-SE3, MuJoCo loop 재anchor, real leaky xy/yaw + 하드가드.
- **`computeAnchorError(e9)`**(§2): anchor 변환된 ref_root + `q_virtual_`.
- **`processObservation()`**:
  ```
  [0:3]   ang_vel_b
  [3:6]   projected_gravity_b
  [6:19]  motion_ref_[0:13]    (q_ref RAW)
  [19:32] motion_ref_[13:26]   (qd_ref RAW)
  [32:41] computeAnchorError() (anchor 9, RAW)
  [41:54] q_noise_ - q_default_p73_  (13)
  [54:67] clip(qd,±30)/30       (13)
  [67:80] last_action_raw_      (13)
  ```
- **standing init**: q_ref=q_default, qd_ref=0, anchor=0 (첫 콜백 전 fallback).
- 불변: feedforward(last_action_raw_=rl_action_), computeFast(q_target = q_default + clip(rl_action,±2)*action_scales_[i]).

### `CMakeLists.txt`
- std_msgs 의존(이미 있음). Eigen quat 사용(이미 있음). 추가 없음.

---

## 5. 재anchor (OOD 경계) — 수정 최종

anchor 변환 `T_a`(ref 프레임 → deploy world): `ref_pos_a = anchor_q_·ref_root_pos + anchor_t_`, `ref_quat_a = anchor_q_·ref_root_quat`.

### (a) 시작 — full SE3 (RSI 등가, 1회)
첫 유효 콜백 시: `anchor_q_,anchor_t_` = robot_pose ∘ ref_pose⁻¹ (6-DoF). → 시작 오차 ~0. `anchor_inited_=true`.

### (b) MuJoCo — 재anchor 최소 (학습 일치 기준선)
- `anchor_leak_alpha_ = 0`. 로봇 state ground-truth → e_anchor = 진짜 추종오차 = 학습과 일치.
- **모션 loop/전환 경계에서만** full-SE3 재anchor(=RSI). 단일 비루프면 시작 1회.
- → **parity 검증 기준**: cc.cpp e_anchor 분포가 Isaac play와 겹쳐야(톱니 없이).

### (c) real robot — leaky xy/yaw 드리프트 흡수
- `anchor_leak_alpha_ > 0`: 매 step `anchor_t_.xy`(+yaw)를 로봇 odometry 쪽으로 저주파 보정 → 느린 드리프트만 흡수, 빠른 추종오차는 통과(high-pass).
- z, roll, pitch는 anchor 안 함(관측가능, genuine 추종오차).
- **하드가드**: `|e_anchor_pos.xy| > anchor_xy_guard_(0.20m)` 시 1회 부드럽게 xy/yaw 재정렬(안전망, 평상시 미발동).
- α는 odometry 드리프트율에 맞춰 튜닝. MuJoCo(α=0)에서 학습 일치 확인 후 real에서 보수적으로 올림.

---

## 6. ONNX export (재학습 없이 기존 ckpt)
- isaaclab `scripts/tools/export_student_onnx.py`:
  ```bash
  conda activate p73
  python scripts/tools/export_student_onnx.py \
    --checkpoint .../2026-06-17_00-22-29_flow_2teacher_qref/model_14000.pt \
    --num-single-obs 80 --history-length 10 \
    --output ~/ros2_ws/src/p73_cc/policy/policy.onnx
  ```
- FlowExportWrapper = normalizer + policy_obs_scale + encoder + flow euler bake. **critic 미사용** → obs 재구조와 무관, 기존 ckpt 그대로.
- 입력 800D, 출력 action 13D. command qd_ref RAW(ONNX ×1/30 내부).

---

## 7. Bit-exact 검증 (Isaac ↔ cc.cpp)
1. isaaclab play(동일 모션 walk1_subject1 또는 walk_short), 매 step obs 80D + action 13D JSONL dump.
2. MuJoCo + cc.cpp 동일 모션, 동일 dump.
3. 첫 N step bit 비교(특히 [6:41]):
   - q_ref[6:19], qd_ref[19:32]: publisher 중심차분 ↔ Isaac `_ref_dof_pos/vel[reorder]`.
   - anchor[32:41]: cc.cpp computeAnchorError ↔ Isaac get_motion_root_* (동일 anchor 시점, MuJoCo α=0).
   - jvel /30, last_action raw, action_scale.
4. 불일치 우선순위: scaling(§1) / joint reorder / quat wxyz↔xyzw / **ori 규약(ref⊗robot⁻¹)** / qd_ref 차분식.

---

## 8. Real robot 고려 (q_virtual_ 신뢰도)
- ori/yaw: IMU → e_anchor_ori 신뢰. yaw 느린 드리프트 → §5(c) leaky 흡수.
- z: 다리 kinematics → e_anchor_pos.z genuine.
- x,y: `q_virtual_[0:2]` = state estimator(leg odometry) 절대 드리프트 → §5(c) leaky로 상대만.
- p73 base estimator 품질이 anchor.xy 정확도 좌우 → MuJoCo parity(α=0) 후 real α 보수적 시작.

---

## 9. 작업 순서
1. `cc.h`: num_single_obs 80 / num_motion_cmd 26 / motion_ref_ 33 / anchor 상태 / 메서드 선언.
2. `cc.cpp`: 콜백, computeAnchorError(§2), updateAnchor(§5), processObservation(§1), standing init.
3. `motion_cmd_publisher.py`: 33D 레퍼런스 + qd_ref 중심차분(§3).
4. ONNX export(§6) → policy/policy.onnx.
5. colcon build (단일 install, walker_motion branch — ros2_ws_build 규칙).
6. **MuJoCo parity**(§7, α=0): Isaac play와 bit 일치 → 기준선.
7. real: leaky α 보수적 시작 → 드리프트 관찰 튜닝, 하드가드 확인.

## 10. 체크리스트
- [ ] obs 80D, [6:32]=command(q_ref+qd_ref), [32:41]=anchor(cc.cpp), 나머지 proprio.
- [ ] qd_ref = dof_pos 중심차분(dt=1/fps), ALL_JOINT order.
- [ ] command/anchor RAW, proprio jvel /30, last_action raw.
- [ ] anchor ori = isaaclab `quat_mul(ref, conj(robot))` (world), Rot6D 부호 일치.
- [ ] q_virtual_ xyzw→wxyz 변환.
- [ ] ONNX 800D, flow euler + policy_obs_scale bake, **재학습 없이 model_14000**.
- [ ] MuJoCo: α=0, 재anchor 시작1회 → Isaac play와 e_anchor 일치(톱니 없음).
- [ ] real: leaky xy/yaw만, z/roll/pitch genuine, 하드가드 미발동이 정상.
- [ ] motion_ref 0/standing fallback 시 default pose 유지.
