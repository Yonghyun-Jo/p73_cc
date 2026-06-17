#include "cc.h"
#include <cmath>
#include <iomanip>
#include <numeric>
#include <fstream>

// =====================================================================
// NOTE on joint ordering:
//
// SHM data (from MuJoCo via launch joint_names) is in MuJoCo/IsaacLab order:
//   L_HipRoll, L_HipPitch, L_HipYaw, L_Knee, L_AnklePitch, L_AnkleRoll,
//   R_HipRoll, R_HipPitch, R_HipYaw, R_Knee, R_AnklePitch, R_AnkleRoll,
//   WaistYaw
//
// This is the SAME order as IsaacLab ALL_JOINT_NAMES and MuJoCo XML actuators.
// Therefore NO permutation is needed — data flows directly.
// =====================================================================

// =====================================================================
// Constructor
// =====================================================================
CustomController::CustomController(DataContainer &dc, RobotEigenData &rd)
    :   dc_(dc), rd_(rd),
        env(ORT_LOGGING_LEVEL_WARNING, "p73_cc"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        session(nullptr)
{
    if(is_on_robot_){
        weight_dir_ = "/home/bluerobin/ros2_ws/src/p73_cc/policy/policy.onnx";
    }
    else{
        weight_dir_ = std::string(getenv("HOME")) + "/ros2_ws/src/p73_cc/policy/policy.onnx";
    }

    if (is_write_file_) {
        writeFile.open("/tmp/p73_cc_data.csv", ofstream::out);
        writeFile << fixed << setprecision(8);
    }

    loadOnnX();
    initVariable();
    startSubscribers();
}

// =====================================================================
// initVariable — ALL values in MuJoCo/IsaacLab order (Roll, Pitch, Yaw)
// =====================================================================
void CustomController::initVariable()
{
    cout << "[p73_cc] Initializing variables (MOTION MIMIC mode)" << endl;

    q_default_p73_ << 0.0, 0.18, 0.0, 0.35, -0.17, 0.0,
                       0.0, -0.18, 0.0, -0.35, 0.17, 0.0,
                       0.0;

    kp_p73_ << 1536.0, 937.5, 625.0, 570.08, 463.896, 463.788,
               1536.0, 937.5, 625.0, 570.08, 463.896, 463.788,
               576.0;

    kd_p73_ << 76.8, 37.5, 12.5, 28.504, 16.0, 5.3,
               76.8, 37.5, 12.5, 28.504, 16.0, 5.3,
               19.2;

    torque_bound_p73_ << 352.0, 220.0, 95.0, 220.0, 95.0, 95.0,
                          352.0, 220.0, 95.0, 220.0, 95.0, 95.0,
                          152.0;

    q_limit_lower_p73_ << -0.58, -1.57, -0.78, 0.0, -0.90, -0.42,
                           -0.58, -2.09, -0.78, -2.56, -0.7, -0.42,
                           -1.57;  // WaistYaw
    q_limit_upper_p73_ << 0.3, 2.09, 0.78, 2.56, 0.7, 0.42,
                           0.3, 1.57, 0.78, 0.0, 0.90, 0.42,
                           1.57;   // WaistYaw

    rl_action_.setZero();
    last_action_raw_.setZero();
    torque_rl_.setZero();

    policy_frame_.assign(num_single_obs, 0.0f);
    policy_obs_hist_term_major_.assign(policy_obs_dim_, 0.0f);
    policy_hist_initialized_ = false;

    // Initialize motion_ref_ to standing reference (not zeros) — fallback before
    // the first /p73/motion_cmd callback. zeros would be out-of-distribution.
    //   [0:13]  q_ref  = q_default
    //   [13:26] qd_ref = 0
    //   [26:29] ref_root_pos = standing (x=0, y=0, z=0.82)
    //   [29:33] ref_root_quat = identity (wxyz = 1,0,0,0)
    motion_ref_.fill(0.0);
    for (int i = 0; i < MODEL_DOF; i++)
        motion_ref_[i] = q_default_p73_(i);   // q_ref = q_default
    motion_ref_[28] = 0.82;                    // ref_root_pos.z (standing height)
    motion_ref_[29] = 1.0;                     // ref_root_quat.w (identity)

    // anchor not yet aligned to robot; computeAnchorError() returns 0 until first callback.
    anchor_t_.setZero();
    anchor_q_.setIdentity();
    anchor_inited_ = false;
}

// =====================================================================
// loadOnnX
// =====================================================================
void CustomController::loadOnnX()
{
    string cur_path = weight_dir_;
    cout << "[p73_cc] Loading network from " << cur_path << endl;

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");
    session = Ort::Session(env, cur_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    input_number = session.GetInputCount();
    output_number = session.GetOutputCount();

    input_names.resize(input_number);
    output_names.resize(output_number);
    input_names_char.resize(input_number);
    output_names_char.resize(output_number);

    for (size_t i = 0; i < input_number; i++) {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
        input_names[i] = name.get();
    }
    for (size_t i = 0; i < output_number; i++) {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
        output_names[i] = name.get();
    }

    cout << "[p73_cc] Input names: ";
    copy(input_names.begin(), input_names.end(), ostream_iterator<string>(cout, " "));
    cout << endl;
    cout << "[p73_cc] Output names: ";
    copy(output_names.begin(), output_names.end(), ostream_iterator<string>(cout, " "));
    cout << endl;

    for (size_t i = 0; i < input_names.size(); ++i) {
        input_names_char[i] = input_names[i].c_str();
        if (input_names[i] == "policy_obs_history" || input_names[i] == "obs")
            input_policy_idx_ = static_cast<int>(i);
        if (input_names[i] == "critic_obs")
            input_critic_idx_ = static_cast<int>(i);
    }
    for (size_t i = 0; i < output_names.size(); ++i) {
        output_names_char[i] = output_names[i].c_str();
        if (output_names[i] == "actions") output_actions_idx_ = static_cast<int>(i);
        if (output_names[i] == "value")   output_value_idx_ = static_cast<int>(i);
    }

    if (input_policy_idx_ < 0)
        throw std::runtime_error("[p73_cc] ONNX input 'obs' or 'policy_obs_history' not found.");
    if (output_actions_idx_ < 0)
        throw std::runtime_error("[p73_cc] ONNX output 'actions' not found.");

    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();
        cout << "[p73_cc] Input " << i << " (" << input_names[i] << ") shape: ";
        for (size_t k = 0; k < input_shape.size(); k++)
            cout << input_shape[k] << (k + 1 < input_shape.size() ? "x" : "");
        cout << endl;

        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0f);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }

    if (input_policy_idx_ >= 0) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(static_cast<size_t>(input_policy_idx_));
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto s = tensor_info.GetShape();
        if (s.size() == 2 && s[1] > 0) {
            policy_obs_dim_ = static_cast<int>(s[1]);
            if (policy_obs_dim_ % num_single_obs != 0)
                throw std::runtime_error(
                    "[p73_cc] policy_obs dim=" + std::to_string(policy_obs_dim_) +
                    " not divisible by num_single_obs=" + std::to_string(num_single_obs) + ".");
            history_length_ = policy_obs_dim_ / num_single_obs;
            cout << "[p73_cc] Inferred policy_obs_dim=" << policy_obs_dim_
                 << " (history_length=" << history_length_ << ")" << endl;
        }
    }

    cout << "[p73_cc] Network loaded successfully." << endl;
}

// =====================================================================
// processNoise — TOCABI sim2real pattern
//
// Real robot: direct sensor values for both q and q_dot
// Simulation: tiny noise on position + numerical differentiation
//
// q_noise_ and q_vel_noise_ are used by BOTH obs AND PD (consistent)
// =====================================================================
void CustomController::processNoise()
{
    noise_time_cur_ = rd_.control_time_us_ / 1e6;

    if (is_on_robot_)
    {
        // Real robot: use sensor values directly
        q_noise_ = rd_.q_;
        q_vel_noise_ = rd_.q_dot_;
    }
    else
    {
        // Simulation: add tiny noise + numerical differentiation (matching TOCABI)
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<> dis(-0.00001, 0.00001);

        for (int i = 0; i < MODEL_DOF; i++)
            q_noise_(i) = rd_.q_(i) + dis(gen);

        double dt = noise_time_cur_ - noise_time_pre_;
        if (dt > 0.0) {
            q_vel_noise_ = (q_noise_ - q_noise_pre_) / dt;
        }

        q_noise_pre_ = q_noise_;
    }

    noise_time_pre_ = noise_time_cur_;
}

// =====================================================================
// processObservation — 80D per frame (q_ref flow student policy)
//
// Layout (PLAN_qref_flow_deploy.md §1, all RAW unless noted):
//   [0:3]   base_ang_vel           3D
//   [3:6]   projected_gravity      3D
//   [6:19]  q_ref                  13D  (motion_ref_[0:13],  publisher, RAW)
//   [19:32] qd_ref                 13D  (motion_ref_[13:26], publisher, RAW; ONNX ×1/30)
//   [32:41] anchor error           9D   (computeAnchorError, RAW; pos3 + Rot6D)
//   [41:54] joint_pos_rel          13D  (ALL joints incl. WaistYaw, q - q_default)
//   [54:67] joint_vel              13D  (ALL joints, clip ±30, /30)
//   [67:80] last_action            13D  (raw network output, legs + WaistYaw)
// =====================================================================
void CustomController::processObservation()
{
    Quaterniond q;
    q.x() = rd_.q_virtual_(3);
    q.y() = rd_.q_virtual_(4);
    q.z() = rd_.q_virtual_(5);
    q.w() = rd_.q_virtual_(6);

    // MuJoCo gyro sensor outputs body-frame angular velocity directly.
    // NO rotation needed (unlike TOCABI which uses d->qvel world-frame).
    Vector3d ang_vel_b = rd_.q_dot_virtual_.segment(3, 3);

    Vector3d g_w(0.0, 0.0, -1.0);
    Vector3d projected_gravity_b = quatRotateInverse(q, g_w);

    // Update anchor (start full-SE3 / optional leaky) BEFORE computing anchor error.
    updateAnchor();
    double e9[9];
    computeAnchorError(e9);

    int idx = 0;

    // base_ang_vel (3D)
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(0));
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(1));
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(2));

    // projected_gravity (3D)
    policy_frame_[idx++] = static_cast<float>(projected_gravity_b(0));
    policy_frame_[idx++] = static_cast<float>(projected_gravity_b(1));
    policy_frame_[idx++] = static_cast<float>(projected_gravity_b(2));

    // command: q_ref(13) + qd_ref(13) from /p73/motion_cmd (RAW — ONNX bakes scales)
    {
        std::lock_guard<std::mutex> lock(motion_ref_mutex_);
        for (int i = 0; i < num_motion_cmd; i++)   // 26 = q_ref13 + qd_ref13
            policy_frame_[idx++] = static_cast<float>(motion_ref_[i]);
    }

    // anchor error (9D) — RAW: motion_root_pos_b(3) + motion_root_ori_b Rot6D(6)
    for (int i = 0; i < 9; i++)
        policy_frame_[idx++] = static_cast<float>(e9[i]);

    // joint_pos_rel (13D) — ALL joints including WaistYaw, relative to default
    for (int i = 0; i < MODEL_DOF; i++) {
        policy_frame_[idx++] = static_cast<float>(q_noise_(i) - q_default_p73_(i));
    }

    // joint_vel (13D) — ALL joints including WaistYaw, clipped & scaled
    for (int i = 0; i < MODEL_DOF; i++) {
        double v_clip = DyrosMath::minmax_cut(q_vel_noise_(i), -30.0, 30.0);
        policy_frame_[idx++] = static_cast<float>(v_clip / 30.0);
    }

    // last_action (13D) — raw network output (legs + WaistYaw)
    for (int i = 0; i < num_action; i++)
        policy_frame_[idx++] = static_cast<float>(last_action_raw_(i));

    // Frame-major history: [frame0(80D), frame1(80D), ..., frame(H-1)(80D)]
    const int H = history_length_;
    const int F = num_single_obs;

    if (!policy_hist_initialized_) {
        // Fill all H frames with the current frame
        for (int t = 0; t < H; ++t)
            std::memcpy(policy_obs_hist_term_major_.data() + t * F,
                        policy_frame_.data(), sizeof(float) * F);
        policy_hist_initialized_ = true;
    } else {
        // Shift left by one frame (drop oldest), append newest at end
        std::memmove(policy_obs_hist_term_major_.data(),
                     policy_obs_hist_term_major_.data() + F,
                     sizeof(float) * F * (H - 1));
        std::memcpy(policy_obs_hist_term_major_.data() + F * (H - 1),
                    policy_frame_.data(), sizeof(float) * F);
    }

    std::memcpy(input_states_buffer[input_policy_idx_].data(),
                policy_obs_hist_term_major_.data(), sizeof(float) * policy_obs_dim_);

    if (input_critic_idx_ >= 0) {
        std::vector<float> &critic_in = input_states_buffer[input_critic_idx_];
        Vector3d lin_vel_w = rd_.q_dot_virtual_.segment<3>(0);
        Vector3d lin_vel_b = quatRotateInverse(q, lin_vel_w);
        critic_in[0] = static_cast<float>(lin_vel_b(0));
        critic_in[1] = static_cast<float>(lin_vel_b(1));
        critic_in[2] = static_cast<float>(ang_vel_b(2));
        for (int i = 3; i < 9; i++) critic_in[i] = 0.0f;
        if (critic_in.size() >= static_cast<size_t>(9 + num_single_obs))
            std::memcpy(critic_in.data() + 9, policy_frame_.data(), sizeof(float) * num_single_obs);
    }
}

// =====================================================================
// feedforwardPolicy
// =====================================================================
void CustomController::feedforwardPolicy()
{
    // Use local variable instead of member output_tensors to avoid
    // Ort::Value destructor interfering with heap between calls
    auto local_output = session.Run(
        Ort::RunOptions{nullptr},
        input_names_char.data(), input_tensors.data(), input_number,
        output_names_char.data(), output_number);

    if (output_actions_idx_ >= 0 &&
        static_cast<size_t>(output_actions_idx_) < local_output.size() &&
        local_output[output_actions_idx_].IsTensor()) {
        const float *actions_ptr = local_output[output_actions_idx_].GetTensorMutableData<float>();
        for (int i = 0; i < num_action; i++)
            rl_action_(i) = actions_ptr[i];
    }

    if (output_value_idx_ >= 0 &&
        static_cast<size_t>(output_value_idx_) < local_output.size() &&
        local_output[output_value_idx_].IsTensor()) {
        const float *value_ptr = local_output[output_value_idx_].GetTensorMutableData<float>();
        value_ = static_cast<double>(value_ptr[0]);
    }

    // Store raw network output for observation (mdp.last_action = raw output)
    for (int i = 0; i < num_action; i++)
        last_action_raw_(i) = rl_action_(i);
    // local_output destroyed here — Ort::Value cleanup happens at function exit
}

// =====================================================================
// computeFast — uses rd_ directly, NO copyRobotData
// =====================================================================
void CustomController::computeFast()
{
    float control_time_us = rd_.control_time_us_;

    static bool init = true;
    if (init) {
        init = false;
        start_time_ = control_time_us;
        q_init_ = rd_.q_;
        torque_init_ = rd_.torque_desired;
        time_inference_pre_ = control_time_us - policy_dt_ * 1e6;
        rl_action_.setZero();
        last_action_raw_.setZero();
        policy_hist_initialized_ = false;
        std::fill(policy_obs_hist_term_major_.begin(), policy_obs_hist_term_major_.end(), 0.0f);

        // Initialize processNoise state
        q_noise_ = rd_.q_;
        q_noise_pre_ = q_noise_;
        q_vel_noise_.setZero();
        noise_time_cur_ = control_time_us / 1e6;
        noise_time_pre_ = noise_time_cur_ - 0.001;

        cout << "[p73_cc] Mode started (is_on_robot=" << is_on_robot_ << ", MOTION MIMIC)" << endl;

        processNoise();
        processObservation();
        feedforwardPolicy();
    }

    // Update noise/velocity state every tick (before policy and PD)
    processNoise();

    // Policy update at 50Hz
    static int policy_step_count = 0;
    if ((control_time_us - time_inference_pre_) / 1.0e6 >= policy_dt_) {
        processObservation();
        feedforwardPolicy();

        time_inference_pre_ = control_time_us;
        policy_step_count++;

        // Dump first N policy steps to JSONL + console
        constexpr int dump_max_steps = 25;
        if (policy_step_count <= dump_max_steps) {
            // 80D frame: 7 terms
            constexpr int dims[] = {3, 3, 13, 13, 9, 13, 13, 13};
            const char* term_names[] = {"ang_vel", "gravity", "q_ref", "qd_ref",
                                        "anchor", "joint_pos", "joint_vel", "last_action"};
            constexpr int num_terms = 8;
            int H = history_length_;

            // Extract newest frame (80D) from frame-major buffer
            const float *newest = policy_obs_hist_term_major_.data() + (H - 1) * num_single_obs;
            int fi = 0;

            // Write JSONL to /tmp/walker_mujoco_obs.jsonl
            static std::ofstream dump_file("/tmp/walker_mujoco_obs.jsonl", std::ios::out);
            if (dump_file.is_open()) {
                dump_file << std::fixed << std::setprecision(8);
                dump_file << "{\"step\":" << policy_step_count - 1;

                // full obs
                dump_file << ",\"obs_" << policy_obs_dim_ << "\":[";
                for (int i = 0; i < policy_obs_dim_; i++)
                    dump_file << policy_obs_hist_term_major_[i] << (i < policy_obs_dim_-1 ? "," : "");
                dump_file << "]";

                // actions
                dump_file << ",\"actions\":[";
                for (int i = 0; i < num_action; i++)
                    dump_file << rl_action_(i) << (i < num_action-1 ? "," : "");
                dump_file << "]";

                // per-term newest frame
                dump_file << ",\"frame_80\":{";
                fi = 0;
                for (int t = 0; t < num_terms; t++) {
                    dump_file << "\"" << term_names[t] << "\":[";
                    for (int d = 0; d < dims[t]; d++)
                        dump_file << newest[fi++] << (d < dims[t]-1 ? "," : "");
                    dump_file << "]";
                    if (t < num_terms - 1) dump_file << ",";
                }
                dump_file << "}";

                // raw state
                dump_file << ",\"raw\":{";
                dump_file << "\"quat_xyzw\":[" << rd_.q_virtual_(3) << "," << rd_.q_virtual_(4)
                          << "," << rd_.q_virtual_(5) << "," << rd_.q_virtual_(6) << "]";
                dump_file << ",\"ang_vel_body\":[" << rd_.q_dot_virtual_(3) << ","
                          << rd_.q_dot_virtual_(4) << "," << rd_.q_dot_virtual_(5) << "]";
                dump_file << ",\"joint_pos\":[";
                for (int i = 0; i < 13; i++)
                    dump_file << rd_.q_(i) << (i < 12 ? "," : "");
                dump_file << "],\"joint_vel\":[";
                for (int i = 0; i < 13; i++)
                    dump_file << rd_.q_dot_(i) << (i < 12 ? "," : "");
                dump_file << "]}";

                dump_file << "}\n";
                dump_file.flush();
            }

        }
    }

    // Action → Target Position
    // Match Isaac Lab JointPositionAction.process_actions:
    //   processed = raw * scale + q_default   (use_default_offset=True)
    //   processed = clamp(processed, clip)     (clip = ±2.0 from ActionsCfg)
    // Then clamp to physical joint limits to prevent PD from fighting
    // unreachable targets (Isaac Lab's PhysX enforces limits physically;
    // here we must enforce them explicitly to avoid sustained max torque).
    VectorQd target_pos = q_default_p73_;
    for (int i = 0; i < num_action; i++) {
        target_pos(i) = rl_action_(i) * action_scales_[i] + q_default_p73_(i);
        target_pos(i) = DyrosMath::minmax_cut(target_pos(i), -2.0, 2.0);
        target_pos(i) = DyrosMath::minmax_cut(target_pos(i), q_limit_lower_p73_(i), q_limit_upper_p73_(i));
    }

    // Position-level spline transition for first 100ms (PD ramp-in, always)
    if (control_time_us < start_time_ + 0.1e6) {
        for (int i = 0; i < MODEL_DOF; i++) {
            rd_.q_desired(i) = DyrosMath::cubic(control_time_us, start_time_, start_time_ + 0.1e6, q_init_(i), target_pos(i), 0.0, 0.0);
        }
        // During ramp-in: always use PD control for safety
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_rl_(i) = kp_p73_(i) * (rd_.q_desired(i) - q_noise_(i)) - kd_p73_(i) * q_vel_noise_(i);
        }
    } else {
        // PD mode: recompute torque every tick at 1kHz
        rd_.q_desired = target_pos;
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_rl_(i) = kp_p73_(i) * (rd_.q_desired(i) - q_noise_(i)) - kd_p73_(i) * q_vel_noise_(i);
        }
    }

    // torque_bound_p73_ is the MOTOR-side limit. The clamp must happen on the
    // motor-space torque (τ_m = J^T τ_j), not on joint-space torque.
    if (is_on_robot_) {
        VectorQd torque_motor = rd_.four_bar_Jaco_.transpose() * torque_rl_;
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_motor(i) = DyrosMath::minmax_cut(torque_motor(i), -torque_bound_p73_(i), torque_bound_p73_(i));
        }
        rd_.torque_desired = torque_motor;
    } else {
        // Evaluate J at the current joint configuration.
        VectorQd q_motor_curr;
        sim_four_bar_.Joint2MotorDesiredPos(q_noise_, q_motor_curr);
        VectorQd joint_pos_dummy, joint_vel_dummy;
        VectorQd motor_vel_zero = VectorQd::Zero();
        sim_four_bar_.Motor2JointPosVel(q_motor_curr, joint_pos_dummy, motor_vel_zero, joint_vel_dummy);
        MatrixQQd J = sim_four_bar_.getFourBarJaco();

        VectorQd torque_motor = J.transpose() * torque_rl_;
        for (int i = 0; i < MODEL_DOF; i++) {
            torque_motor(i) = DyrosMath::minmax_cut(torque_motor(i), -torque_bound_p73_(i), torque_bound_p73_(i));
        }
        rd_.torque_desired = J.transpose().inverse() * torque_motor;
    }

    // ====== Data logging (every tick, ~1kHz) ======
    static std::ofstream log_file;
    static bool log_opened = false;
    if (!log_opened) {

        std::string log_dir = std::string(getenv("HOME")) + "/ros2_ws/src/p73_cc/logs";
        if(is_on_robot_){
            log_dir = "/home/bluerobin/ros2_ws/src/p73_cc/logs";
        }
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        localtime_r(&t, &tm_buf);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%y%m%d_%H%M%S", &tm_buf);
        std::string prefix = is_on_robot_ ? "realrobot" : "mujoco";
        std::string path = log_dir + "/" + prefix + "_" + ts + ".csv";
        log_file.open(path, std::ios::out);
        log_file << std::fixed << std::setprecision(8);
        // Header
        log_file << "time";
        // IMU quaternion (xyzw)
        log_file << ",quat_x,quat_y,quat_z,quat_w";
        // Angular velocity body frame
        log_file << ",ang_vel_bx,ang_vel_by,ang_vel_bz";
        // Projected gravity body frame
        log_file << ",proj_grav_x,proj_grav_y,proj_grav_z";
        // Motion reference (33D: q_ref13 + qd_ref13 + ref_root_pos3 + ref_root_quat4)
        for (size_t i = 0; i < motion_ref_.size(); i++) log_file << ",motion_ref_" << i;
        // Joint pos measured (13 DOF, raw)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",q_raw_" << i;
        // Joint pos desired (13 DOF)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",q_des_" << i;
        // Joint pos relative to default (13 DOF)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",q_rel_" << i;
        // Joint vel measured (13 DOF)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",qdot_" << i;
        // Policy obs frame (80D)
        for (int i = 0; i < num_single_obs; i++) log_file << ",obs_" << i;
        // RL actions (12)
        for (int i = 0; i < num_action; i++) log_file << ",action_" << i;
        // Torque (joint space) — desired then measured (13 + 13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_joint_" << i;
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_meas_joint_" << i;
        // Torque (motor space) — desired then measured (13 + 13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_motor_" << i;
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_meas_motor_" << i;
        // Linear velocity world frame (for debug)
        log_file << ",lin_vel_wx,lin_vel_wy,lin_vel_wz";
        // measured: rd_.motor_voltage_ from ECAT (motor side, real robot only)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",motor_voltage_" << i;
        // Value function output
        log_file << ",value";
        log_file << "\n";
        log_opened = true;
        cout << "[p73_cc] Logging to: " << path << endl;
    }

    if (log_file.is_open()) {
        Quaterniond q_log;
        q_log.x() = rd_.q_virtual_(3);
        q_log.y() = rd_.q_virtual_(4);
        q_log.z() = rd_.q_virtual_(5);
        q_log.w() = rd_.q_virtual_(6);
        Vector3d ang_vel_log = rd_.q_dot_virtual_.segment<3>(3);
        Vector3d g_w_log(0.0, 0.0, -1.0);
        Vector3d proj_grav_log = quatRotateInverse(q_log, g_w_log);
        Vector3d lin_vel_w_log = rd_.q_dot_virtual_.segment<3>(0);

        VectorQd tau_joint = (control_time_us < start_time_ + 0.1e6) ? torque_spline_ : torque_rl_;

        log_file << control_time_us / 1e6;
        // Quaternion
        log_file << "," << q_log.x() << "," << q_log.y() << "," << q_log.z() << "," << q_log.w();
        // Ang vel body
        log_file << "," << ang_vel_log(0) << "," << ang_vel_log(1) << "," << ang_vel_log(2);
        // Projected gravity
        log_file << "," << proj_grav_log(0) << "," << proj_grav_log(1) << "," << proj_grav_log(2);
        // Motion reference (33D)
        {
            std::lock_guard<std::mutex> lock(motion_ref_mutex_);
            for (size_t i = 0; i < motion_ref_.size(); i++) log_file << "," << motion_ref_[i];
        }
        // Joint pos measured (13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.q_(i);
        // Joint pos desired (13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.q_desired(i);
        // Joint pos relative (13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << (q_noise_(i) - q_default_p73_(i));
        // Joint vel measured (13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << q_vel_noise_(i);
        // Policy frame (80D)
        for (int i = 0; i < num_single_obs; i++) log_file << "," << policy_frame_[i];
        // Actions (12)
        for (int i = 0; i < num_action; i++) log_file << "," << rl_action_(i);
        // Torque joint space — desired then measured (13 + 13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << tau_joint(i);
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.q_torque_(i);
        // Torque motor space — desired then measured (13 + 13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.torque_desired(i);
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.q_torque_motor_(i);
        // Lin vel world
        log_file << "," << lin_vel_w_log(0) << "," << lin_vel_w_log(1) << "," << lin_vel_w_log(2);
        // Motor voltage
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.motor_voltage_(i);
        // Value
        log_file << "," << value_;
        log_file << "\n";

        // Flush every 100 ticks (~10Hz) to avoid losing data on crash
        static int flush_cnt = 0;
        if (++flush_cnt % 100 == 0) log_file.flush();
    }

    // Debug console
    static int dbg = 0;
    if (dbg++ % 500 == 0) {
        Eigen::IOFormat fmt(3, 0, " ", " ");
        cout << "[cc] t=" << control_time_us/1e6
             << " act: " << rl_action_.transpose().format(fmt) << endl;
    }
}

// =====================================================================
void CustomController::computeSlow() {}

void CustomController::copyRobotData(RobotEigenData &rd_l)
{
    // DEPRECATED: memcpy on RobotEigenData corrupts std::vector members.
    // Use rd_ directly instead.
    (void)rd_l;
}

Vector3d CustomController::quatRotateInverse(const Quaterniond &q, const Vector3d &v)
{
    Vector3d q_vec = q.vec();
    double q_w = q.w();
    Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Vector3d b = 2.0 * q_w * q_vec.cross(v);
    Vector3d c = 2.0 * q_vec * q_vec.dot(v);
    return a - b + c;
}

// =====================================================================
// Anchor math — isaaclab convention (wxyz). See PLAN §2 / anchor.md §7.
//
// Helpers below use the SAME wxyz formulas as isaaclab.utils.math:
//   quat_mul(q1,q2)   = Hamilton product, both in (w,x,y,z)
//   quat_conjugate(q) = (w, -x, -y, -z)
//   quat_rotate_inverse(q,v) = R(q)^T · v  (== quatRotateInverse above)
// Eigen::Quaterniond stores (w,x,y,z) accessible via .w()/.x()/.y()/.z();
// its operator* is the Hamilton product, matching isaaclab quat_mul exactly,
// and .conjugate() gives (w,-x,-y,-z). So we use Eigen directly.
// =====================================================================

// e9 = motion_root_pos_b(3) + motion_root_ori_b Rot6D(6)
void CustomController::computeAnchorError(double e9[9])
{
    // Before the first valid motion_ref callback, anchor is not aligned → report 0.
    if (!anchor_inited_) {
        for (int i = 0; i < 9; i++) e9[i] = 0.0;
        return;
    }

    // robot root pose from q_virtual_ ([0:3]=pos, [3:7]=quat xyzw → build wxyz)
    Vector3d robot_pos = rd_.q_virtual_.head<3>();
    Quaterniond robot_quat(rd_.q_virtual_(6),   // w
                           rd_.q_virtual_(3),   // x
                           rd_.q_virtual_(4),   // y
                           rd_.q_virtual_(5));  // z
    robot_quat.normalize();

    // reference root pose from motion_ref_ ([26:29]=pos, [29:33]=quat wxyz)
    Vector3d ref_root_pos;
    Quaterniond ref_root_quat;
    {
        std::lock_guard<std::mutex> lock(motion_ref_mutex_);
        ref_root_pos = Vector3d(motion_ref_[26], motion_ref_[27], motion_ref_[28]);
        ref_root_quat = Quaterniond(motion_ref_[29],   // w
                                    motion_ref_[30],   // x
                                    motion_ref_[31],   // y
                                    motion_ref_[32]);  // z
    }
    ref_root_quat.normalize();

    // apply anchor transform: ref_pos_a = anchor_q_*ref_root_pos + anchor_t_,
    //                         ref_quat_a = anchor_q_*ref_root_quat
    Vector3d ref_pos_a  = anchor_q_ * ref_root_pos + anchor_t_;
    Quaterniond ref_quat_a = anchor_q_ * ref_root_quat;

    // (1) pos error: e_anchor_pos = quat_rotate_inverse(robot_quat, ref_pos_a - robot_pos)
    Vector3d e_pos = quatRotateInverse(robot_quat, ref_pos_a - robot_pos);
    e9[0] = e_pos(0);
    e9[1] = e_pos(1);
    e9[2] = e_pos(2);

    // (2) ori error (Rot6D): q_rel = quat_mul(ref_quat_a, quat_conjugate(robot_quat)) (WORLD)
    Quaterniond q_rel = ref_quat_a * robot_quat.conjugate();
    q_rel.normalize();
    double w = q_rel.w(), x = q_rel.x(), y = q_rel.y(), z = q_rel.z();
    // Rot6D = first two columns of R(q_rel) — EXACT formula per anchor.md §7 / PLAN §2
    // col0 = [1-2(y²+z²), 2(xy+wz), 2(xz-wy)]
    e9[3] = 1.0 - 2.0 * (y * y + z * z);
    e9[4] = 2.0 * (x * y + w * z);
    e9[5] = 2.0 * (x * z - w * y);
    // col1 = [2(xy-wz), 1-2(x²+z²), 2(yz+wx)]
    e9[6] = 2.0 * (x * y - w * z);
    e9[7] = 1.0 - 2.0 * (x * x + z * z);
    e9[8] = 2.0 * (y * z + w * x);
}

// updateAnchor — PLAN §5: start full-SE3 (a), MuJoCo no-leak (b), real leaky + guard (c)
void CustomController::updateAnchor()
{
    // robot root pose (world)
    Vector3d robot_pos = rd_.q_virtual_.head<3>();
    Quaterniond robot_quat(rd_.q_virtual_(6),   // w
                           rd_.q_virtual_(3),   // x
                           rd_.q_virtual_(4),   // y
                           rd_.q_virtual_(5));  // z
    robot_quat.normalize();

    // reference root pose (motion frame)
    Vector3d ref_root_pos;
    Quaterniond ref_root_quat;
    {
        std::lock_guard<std::mutex> lock(motion_ref_mutex_);
        ref_root_pos = Vector3d(motion_ref_[26], motion_ref_[27], motion_ref_[28]);
        ref_root_quat = Quaterniond(motion_ref_[29], motion_ref_[30],
                                    motion_ref_[31], motion_ref_[32]);
    }
    ref_root_quat.normalize();

    // (a) First valid step: full-SE3 anchor so anchored ref(t0) == robot pose(t0).
    //     anchor_q_ = robot_quat * ref_quat^-1,  anchor_t_ = robot_pos - anchor_q_*ref_pos.
    if (!anchor_inited_) {
        anchor_q_ = robot_quat * ref_root_quat.conjugate();
        anchor_q_.normalize();
        anchor_t_ = robot_pos - anchor_q_ * ref_root_pos;
        anchor_inited_ = true;
        return;
    }

    // (c) real robot leaky xy/yaw drift absorption (MuJoCo: alpha=0 → skipped entirely).
    //     Low-pass correct only the xy translation + yaw of the anchor toward the robot;
    //     z/roll/pitch untouched (genuine tracking error stays). High-frequency tracking
    //     error passes through (high-pass). Clearly marked, guarded by alpha>0.
    if (anchor_leak_alpha_ > 0.0) {
        Vector3d ref_pos_a = anchor_q_ * ref_root_pos + anchor_t_;
        // target anchor_t_.xy that would zero the *world* xy offset of anchored ref vs robot
        Vector2d delta_xy = robot_pos.head<2>() - ref_pos_a.head<2>();
        anchor_t_.head<2>() += anchor_leak_alpha_ * delta_xy;   // leak xy only

        // yaw leak: rotate anchor_q_ slightly about world-z toward robot yaw of (robot ⊗ ref^-1)
        Quaterniond q_yaw_err = robot_quat * (anchor_q_ * ref_root_quat).conjugate();
        q_yaw_err.normalize();
        // extract yaw angle (z-axis) of the residual rotation
        double yaw_err = std::atan2(2.0 * (q_yaw_err.w() * q_yaw_err.z() +
                                           q_yaw_err.x() * q_yaw_err.y()),
                                    1.0 - 2.0 * (q_yaw_err.y() * q_yaw_err.y() +
                                                 q_yaw_err.z() * q_yaw_err.z()));
        double yaw_leak = anchor_leak_alpha_ * yaw_err;
        Quaterniond dq_yaw(std::cos(0.5 * yaw_leak), 0.0, 0.0, std::sin(0.5 * yaw_leak));
        anchor_q_ = dq_yaw * anchor_q_;   // pre-multiply (world-z rotation)
        anchor_q_.normalize();

        // (c-guard) hard guard: if anchored-ref xy error (robot frame) exceeds threshold,
        //           re-align xy once (snap anchor_t_.xy). Safety net; normally inactive.
        Vector3d e_pos = quatRotateInverse(robot_quat, ref_pos_a - robot_pos);
        if (e_pos.head<2>().norm() > anchor_xy_guard_) {
            anchor_t_.head<2>() += (robot_pos.head<2>() -
                                    (anchor_q_ * ref_root_pos + anchor_t_).head<2>());
        }
    }
    // (b) MuJoCo (alpha=0): no per-step leak. Loop/transition re-anchor (= RSI full-SE3)
    //     would be triggered externally by resetting anchor_inited_=false; not done here.
}

// =====================================================================
// ROS2 Motion Command Subscriber
// =====================================================================
void CustomController::motionRefCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg->data.size() < motion_ref_.size()) {
        return;  // Silently ignore malformed messages (need 33 doubles)
    }
    std::lock_guard<std::mutex> lock(motion_ref_mutex_);
    for (size_t i = 0; i < motion_ref_.size(); i++)
        motion_ref_[i] = msg->data[i];
}

void CustomController::startSubscribers()
{
    // Use the main controller node (dc_.node_) instead of creating a separate node.
    // This shares the same DDS participant as GUI/task command subscriptions,
    // which avoids communication issues when running with sudo on real robot.
    vel_cbg_ = dc_.node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions opts;
    opts.callback_group = vel_cbg_;

    // Motion reference subscriber (33D Float64MultiArray)
    motion_ref_sub_ = dc_.node_->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/p73/motion_cmd", 10,
        std::bind(&CustomController::motionRefCallback, this, std::placeholders::_1),
        opts);

    vel_executor_.add_callback_group(vel_cbg_, dc_.node_->get_node_base_interface());

    vel_spin_running_ = true;
    vel_spin_thread_ = std::thread([this]() {
        while (vel_spin_running_ && rclcpp::ok()) {
            vel_executor_.spin_some(std::chrono::milliseconds(5));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    cout << "[p73_cc] Motion reference subscriber started on topic: /p73/motion_cmd (33D Float64MultiArray)" << endl;
    cout << "[p73_cc]   Layout: q_ref(13) + qd_ref(13) + ref_root_pos(3) + ref_root_quat_wxyz(4)" << endl;
}

void CustomController::stopSubscribers()
{
    vel_spin_running_ = false;
    if (vel_spin_thread_.joinable()) vel_spin_thread_.join();
    motion_ref_sub_.reset();
}
