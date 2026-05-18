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
// This is the SAME order as IsaacLab _LOWER_JOINT_NAMES and MuJoCo XML actuators.
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
    startVelSubscriber();
}

// =====================================================================
// initVariable — ALL values in MuJoCo/IsaacLab order (Roll, Pitch, Yaw)
// =====================================================================
void CustomController::initVariable()
{
    cout << "[p73_cc] Initializing variables" << endl;

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
                           -0.58, -2.09, -0.78, -2.15, -0.7, -0.42;
    q_limit_upper_p73_ << 0.3, 2.09, 0.78, 2.15, 0.7, 0.42,
                           0.3, 1.57, 0.78, 0.0, 0.90, 0.42;

    rl_action_.setZero();
    last_action_processed_.setZero();
    torque_rl_.setZero();

    policy_frame_.assign(num_single_obs, 0.0f);
    policy_obs_hist_term_major_.assign(policy_obs_dim_, 0.0f);
    policy_hist_initialized_ = false;
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
                throw std::runtime_error("[p73_cc] policy_obs_history dim must be divisible by 51.");
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
// processObservation — uses q_noise_/q_vel_noise_ from processNoise()
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
    // Vector3d ang_vel_b = quatRotateInverse(q, rd_.q_dot_virtual_.segment(3, 3));

    Vector3d g_w(0.0, 0.0, -1.0);
    Vector3d projected_gravity_b = quatRotateInverse(q, g_w);

    // Joint pos/vel — from processNoise() (noised in sim, direct on robot)
    VectorXd q_pos = q_noise_.head<12>();
    VectorXd q_vel = q_vel_noise_.head<12>();
    VectorXd q_pos_rel = q_pos - q_default_p73_.head<12>();

    double local_vel_x, local_vel_y, local_vel_yaw;
    double local_height, local_roll, local_pitch, local_yaw_pose;
    {
        std::lock_guard<std::mutex> lock(vel_mutex_);
        local_vel_x = target_vel_x_;
        local_vel_y = target_vel_y_;
        local_vel_yaw = target_vel_yaw_;
        local_height = target_height_;
        local_roll = target_roll_;
        local_pitch = target_pitch_;
        local_yaw_pose = target_yaw_pose_;
    }


    // Velocity command from ROS2 teleop (topic: p73/cmd_vel)
    // Usage: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=p73/cmd_vel

    double cmd_norm = std::sqrt(local_vel_x * local_vel_x +
                                local_vel_y * local_vel_y +
                                local_vel_yaw * local_vel_yaw);
    double phase = 0.0;
    if (cmd_norm > cmd_zero_max_) {
        phase = static_cast<double>(gait_step_counter_ % gait_period_steps_) /
                static_cast<double>(gait_period_steps_);
    }
    double gait_sin = std::sin(2.0 * M_PI * phase);
    double gait_cos = std::cos(2.0 * M_PI * phase);

    int idx = 0;
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(0));
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(1));
    policy_frame_[idx++] = static_cast<float>(ang_vel_b(2));
    policy_frame_[idx++] = static_cast<float>(projected_gravity_b(0));
    policy_frame_[idx++] = static_cast<float>(projected_gravity_b(1));
    policy_frame_[idx++] = static_cast<float>(projected_gravity_b(2));
    policy_frame_[idx++] = static_cast<float>(local_vel_x);
    policy_frame_[idx++] = static_cast<float>(local_vel_y);
    policy_frame_[idx++] = static_cast<float>(local_vel_yaw);
    policy_frame_[idx++] = static_cast<float>(local_height);
    policy_frame_[idx++] = static_cast<float>(local_roll);
    policy_frame_[idx++] = static_cast<float>(local_pitch);
    policy_frame_[idx++] = static_cast<float>(local_yaw_pose);
    policy_frame_[idx++] = static_cast<float>(gait_sin);
    policy_frame_[idx++] = static_cast<float>(gait_cos);
    for (int i = 0; i < 12; i++)
        policy_frame_[idx++] = static_cast<float>(q_pos_rel(i));
    for (int i = 0; i < 12; i++) {
        // Match IsaacLab ObsTerm(clip=(-30,30), scale=1/30)
        double v_clip = DyrosMath::minmax_cut(q_vel(i), -30.0, 30.0);
        policy_frame_[idx++] = static_cast<float>(v_clip / 30.0);
    }
    for (int i = 0; i < num_action; i++)
        policy_frame_[idx++] = static_cast<float>(last_action_processed_(i));

    // Frame-major history: [frame0(48D), frame1(48D), ..., frameH-1(48D)]
    // Each frame is a complete 48D observation. Oldest at front, newest at back.
    // This matches IsaacLab's P73ObservationManager layout.
    const int H = history_length_;
    const int F = num_single_obs;  // 48

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
        // critic prefix: [vel4(vx,vy,wz,vz), foot6, height1] = 11D
        critic_in[0] = static_cast<float>(lin_vel_b(0));
        critic_in[1] = static_cast<float>(lin_vel_b(1));
        critic_in[2] = static_cast<float>(ang_vel_b(2));
        critic_in[3] = static_cast<float>(lin_vel_b(2));  // vz
        for (int i = 4; i < 10; i++) critic_in[i] = 0.0f;  // foot6 placeholder
        critic_in[10] = static_cast<float>(rd_.q_virtual_(2));  // height (pos_z)
        if (critic_in.size() >= static_cast<size_t>(11 + num_single_obs))
            std::memcpy(critic_in.data() + 11, policy_frame_.data(), sizeof(float) * num_single_obs);
    }

    gait_step_counter_++;
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

    for (int i = 0; i < num_action; i++)
        last_action_processed_(i) = DyrosMath::minmax_cut(rl_action_(i) * action_scale_, -2.0, 2.0);
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
        last_action_processed_.setZero();
        gait_step_counter_ = 0;
        policy_hist_initialized_ = false;
        std::fill(policy_obs_hist_term_major_.begin(), policy_obs_hist_term_major_.end(), 0.0f);

        // Initialize processNoise state
        q_noise_ = rd_.q_;
        q_noise_pre_ = q_noise_;
        q_vel_noise_.setZero();
        noise_time_cur_ = control_time_us / 1e6;
        noise_time_pre_ = noise_time_cur_ - 0.001;

        cout << "[p73_cc] Mode started (is_on_robot=" << is_on_robot_ << ")" << endl;

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
            constexpr int dims[] = {3, 3, 3, 4, 1, 1, 12, 12, 12};
            const char* term_names[] = {"ang_vel", "gravity", "cmd", "pose_cmd", "gait_sin", "gait_cos",
                                        "joint_pos", "joint_vel", "last_action"};
            int H = history_length_;

            // Extract newest frame (48D) from frame-major buffer
            // Frame-major: newest frame is the last 48 elements
            const float *newest = policy_obs_hist_term_major_.data() + (H - 1) * num_single_obs;
            int fi = 0;

            // Write JSONL to /tmp/walker_mujoco_obs.jsonl
            static std::ofstream dump_file("/tmp/walker_mujoco_obs.jsonl", std::ios::out);
            if (dump_file.is_open()) {
                dump_file << std::fixed << std::setprecision(8);
                dump_file << "{\"step\":" << policy_step_count - 1;

                // full obs (235D)
                dump_file << ",\"obs_235\":[";
                for (int i = 0; i < policy_obs_dim_; i++)
                    dump_file << policy_obs_hist_term_major_[i] << (i < policy_obs_dim_-1 ? "," : "");
                dump_file << "]";

                // actions
                dump_file << ",\"actions\":[";
                for (int i = 0; i < num_action; i++)
                    dump_file << rl_action_(i) << (i < num_action-1 ? "," : "");
                dump_file << "]";

                // per-term newest frame
                dump_file << ",\"frame_48\":{";
                fi = 0;
                for (int t = 0; t < 9; t++) {
                    dump_file << "\"" << term_names[t] << "\":";
                    if (dims[t] == 1) {
                        dump_file << newest[fi++];
                    } else {
                        dump_file << "[";
                        for (int d = 0; d < dims[t]; d++)
                            dump_file << newest[fi++] << (d < dims[t]-1 ? "," : "");
                        dump_file << "]";
                    }
                    if (t < 8) dump_file << ",";
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

            // Console output (first 5 steps only)
            if (policy_step_count <= 5) {
                Eigen::IOFormat fmt(6, 0, ", ", ", ");
                cout << "\n=== MuJoCo STEP " << policy_step_count - 1 << " ===" << endl;
                fi = 0;
                for (int t = 0; t < 9; t++) {
                    cout << "  " << term_names[t] << ": ";
                    for (int d = 0; d < dims[t]; d++)
                        cout << newest[fi++] << " ";
                    cout << endl;
                }
                cout << "  actions: " << rl_action_.transpose().format(fmt) << endl;
            }
        }
    }

    // Action → Target Position (used by PD for q_desired logging)
    VectorQd target_pos = q_default_p73_;
    for (int i = 0; i < num_action; i++) {
        double dq = rl_action_(i) * action_scale_;
        dq = DyrosMath::minmax_cut(dq, -2.0, 2.0);
        target_pos(i) = q_default_p73_(i) + dq;
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
    // motor-space torque
    // (τ_m = J^T τ_j), not on joint-space torque.
    //
    //   Real robot: rd_.four_bar_Jaco_ is populated by state_estimator; map
    //               τ_j → τ_m, clamp τ_m, send as motor torque. state_estimator
    //               also clamps τ_m at the ECAT boundary (double safety).
    //
    //   MuJoCo   : the sim model has no 4-bar (each <motor> acts directly on its
    //               joint), and state_estimator does not populate four_bar_Jaco_
    //               in simMode. We run the 4-bar kinematics locally here: compute
    //               motor pos from current joint pos, evaluate J, then apply
    //               τ_j' = J^{-T} · clamp(J^T τ_j, ±τ_m_bound). Feeding τ_j'
    //               back to MuJoCo reproduces the real motor-limit envelope
    //               (nonlinear, configuration-dependent) at the joint level.
    if (is_on_robot_) {
        // VectorQd torque_motor = WBC::JointTorqueToMotorTorque(rd_, torque_rl_);
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
        // τ_j' = J^{-T} τ_m_clamped — joint-space torque that realizes the
        // clamped motor torque through the 4-bar.
        rd_.torque_desired = J.transpose().inverse() * torque_motor;
    }

    // // // Spline transition for first 100ms
    // if (control_time_us < start_time_ + 0.1e6) {
    //     for (int i = 0; i < MODEL_DOF; i++)
    //         torque_spline_(i) = DyrosMath::cubic(control_time_us, start_time_, start_time_ + 0.1e6, torque_init_(i), torque_rl_(i), 0.0, 0.0);
    //     rd_.torque_desired = torque_spline_;
    //     if (is_on_robot_) {
    //         rd_.torque_desired = WBC::JointTorqueToMotorTorque(rd_, torque_spline_);
    //     }
    // } else {
    //     rd_.torque_desired = torque_rl_;
    //     if (is_on_robot_) {
    //         rd_.torque_desired = WBC::JointTorqueToMotorTorque(rd_, torque_rl_);
    //     }
    // }

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
        // Angular velocity body frame (from q_dot_virtual_)
        log_file << ",ang_vel_bx,ang_vel_by,ang_vel_bz";
        // Projected gravity body frame
        log_file << ",proj_grav_x,proj_grav_y,proj_grav_z";
        // Velocity command
        log_file << ",cmd_vx,cmd_vy,cmd_vyaw";
        // Gait phase
        log_file << ",gait_sin,gait_cos";
        // Joint pos measured (13 DOF, raw) — analog of joint_position_log
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",q_raw_" << i;
        // Joint pos desired (13 DOF) — analog of joint_desired_log
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",q_des_" << i;
        // Joint pos relative to default (12)
        for (int i = 0; i < 12; i++) log_file << ",q_rel_" << i;
        // Joint vel measured (13 DOF) — analog of joint_velocity_log
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",qdot_" << i;
        // Policy obs frame (51D, what actually goes into network)
        for (int i = 0; i < num_single_obs; i++) log_file << ",obs_" << i;
        // RL actions (12)
        for (int i = 0; i < num_action; i++) log_file << ",action_" << i;
        // Torque (joint space) — analog of torque_joint_log: [desired, measured]
        // desired: PD output BEFORE 4-bar
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_joint_" << i;
        // measured: rd_.q_torque_ from ECAT current readback (joint side)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_meas_joint_" << i;
        // Torque (motor space) — analog of torque_motor_log: [desired, measured]
        // desired: AFTER 4-bar (what actually gets sent)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_motor_" << i;
        // measured: rd_.q_torque_motor_ from ECAT (motor side, real robot only)
        for (int i = 0; i < MODEL_DOF; i++) log_file << ",tau_meas_motor_" << i;
        // Linear velocity world frame (for critic/debug)
        log_file << ",lin_vel_wx,lin_vel_wy,lin_vel_wz";
        // Value function output
        log_file << ",value";
        log_file << "\n";
        log_opened = true;
        cout << "[p73_cc] Logging to: " << path << endl;
    }

    if (log_file.is_open()) {
        // Recompute quantities for logging (some already in policy_frame_)
        Quaterniond q_log;
        q_log.x() = rd_.q_virtual_(3);
        q_log.y() = rd_.q_virtual_(4);
        q_log.z() = rd_.q_virtual_(5);
        q_log.w() = rd_.q_virtual_(6);
        Vector3d ang_vel_log = rd_.q_dot_virtual_.segment<3>(3);
        Vector3d g_w_log(0.0, 0.0, -1.0);
        Vector3d proj_grav_log = quatRotateInverse(q_log, g_w_log);
        Vector3d lin_vel_w_log = rd_.q_dot_virtual_.segment<3>(0);

        // Torque before 4-bar (reconstruct from torque_rl_ or torque_spline_)
        VectorQd tau_joint = (control_time_us < start_time_ + 0.1e6) ? torque_spline_ : torque_rl_;

        double local_vx, local_vy, local_vyaw;
        {
            std::lock_guard<std::mutex> lock(vel_mutex_);
            local_vx = target_vel_x_;
            local_vy = target_vel_y_;
            local_vyaw = target_vel_yaw_;
        }

        double cmd_n = std::sqrt(local_vx*local_vx + local_vy*local_vy + local_vyaw*local_vyaw);
        double ph = 0.0;
        if (cmd_n > cmd_zero_max_)
            ph = static_cast<double>(gait_step_counter_ % gait_period_steps_) / static_cast<double>(gait_period_steps_);

        log_file << control_time_us / 1e6;
        // Quaternion
        log_file << "," << q_log.x() << "," << q_log.y() << "," << q_log.z() << "," << q_log.w();
        // Ang vel body
        log_file << "," << ang_vel_log(0) << "," << ang_vel_log(1) << "," << ang_vel_log(2);
        // Projected gravity
        log_file << "," << proj_grav_log(0) << "," << proj_grav_log(1) << "," << proj_grav_log(2);
        // Cmd vel
        log_file << "," << local_vx << "," << local_vy << "," << local_vyaw;
        // Gait
        log_file << "," << std::sin(2.0*M_PI*ph) << "," << std::cos(2.0*M_PI*ph);
        // Joint pos measured (13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.q_(i);
        // Joint pos desired (13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << rd_.q_desired(i);
        // Joint pos relative (12)
        for (int i = 0; i < 12; i++) log_file << "," << (q_noise_(i) - q_default_p73_(i));
        // Joint vel measured (13)
        for (int i = 0; i < MODEL_DOF; i++) log_file << "," << q_vel_noise_(i);
        // Policy frame (47D)
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
             << " act: " << rl_action_.transpose().format(fmt)
             << " | gait: " << gait_step_counter_ << endl;
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
// ROS2 Velocity Command Subscriber
// =====================================================================
void CustomController::velCmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(vel_mutex_);
    target_vel_x_ = msg->linear.x;
    target_vel_y_ = msg->linear.y;
    target_vel_yaw_ = msg->angular.z;
    // Height command via Twist.linear.z (0 means "keep current")
    if (std::abs(msg->linear.z) > 1e-6)
        target_height_ = msg->linear.z;
    // Pose commands: roll via angular.x, pitch via angular.y
    target_roll_ = msg->angular.x;
    target_pitch_ = msg->angular.y;
}

void CustomController::startVelSubscriber()
{
    // Use the main controller node (dc_.node_) instead of creating a separate node.
    // This shares the same DDS participant as GUI/task command subscriptions,
    // which avoids communication issues when running with sudo on real robot.
    vel_cbg_ = dc_.node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions opts;
    opts.callback_group = vel_cbg_;

    vel_sub_ = dc_.node_->create_subscription<geometry_msgs::msg::Twist>(
        "/p73/cmd_vel", 10,
        std::bind(&CustomController::velCmdCallback, this, std::placeholders::_1),
        opts);

    vel_executor_.add_callback_group(vel_cbg_, dc_.node_->get_node_base_interface());

    vel_spin_running_ = true;
    vel_spin_thread_ = std::thread([this]() {
        while (vel_spin_running_ && rclcpp::ok()) {
            vel_executor_.spin_some(std::chrono::milliseconds(5));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    cout << "[p73_cc] Velocity command subscriber started on topic: /p73/cmd_vel" << endl;
    cout << "[p73_cc] Usage: python3 ~/Walker_ws/src/p73_cc/scripts/walker_teleop.py" << endl;
}

void CustomController::stopVelSubscriber()
{
    vel_spin_running_ = false;
    if (vel_spin_thread_.joinable()) vel_spin_thread_.join();
    vel_sub_.reset();
}
