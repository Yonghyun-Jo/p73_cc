#ifndef p73_cc_H
#define p73_cc_H

#include "p73_lib/robot_data.h"
#include "p73_lib/4bar_jac_func.h"
// #include "wholebody_functions.h"
#include "onnxruntime_cxx_api.h"
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <fstream>
#include <sstream>
#include <array>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>

using namespace Eigen;
using namespace std;

class CustomController
{
public:
    CustomController(DataContainer &dc, RobotEigenData &rd);

    void computeSlow();
    void computeFast();
    void copyRobotData(RobotEigenData &rd_l);

    DataContainer &dc_;
    RobotEigenData &rd_;
    RobotEigenData rd_cc_;

    ofstream writeFile;
    bool is_write_file_ = false;

    //////////////////////// Functions ////////////////////////
    void initVariable();
    void loadOnnX();
    void processNoise();
    void processObservation();
    void feedforwardPolicy();
    Vector3d quatRotateInverse(const Quaterniond &q, const Vector3d &v);

    // Anchor math (isaaclab convention, wxyz). See PLAN_qref_flow_deploy.md §2 / anchor.md §7.
    void computeAnchorError(double e9[9]);   // §2: motion_root_pos_b(3) + motion_root_ori_b Rot6D(6)
    void updateAnchor();                     // §5: start full-SE3, optional leaky xy/yaw + hard guard

    //////////////////////// ONNX Runtime ////////////////////////
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;
    std::vector<std::vector<float>> input_states_buffer;

    int input_policy_idx_ = -1;
    int input_critic_idx_ = -1;
    int output_actions_idx_ = -1;
    int output_value_idx_ = -1;

    //////////////////////// Network Dimensions ////////////////////////
    // q_ref flow student policy (PLAN_qref_flow_deploy.md §1, 80D single frame):
    //   base_ang_vel(3) + projected_gravity(3)
    //   + q_ref(13) + qd_ref(13)          [6:32]  command (publisher, RAW)
    //   + anchor(9)                       [32:41] computed in cc.cpp from q_virtual_
    //   + joint_pos_rel(13) + joint_vel(13) + last_action(13) = 80
    static const int num_action = 13;         // legs(12) + WaistYaw(1)
    static const int num_single_obs = 80;
    static const int num_motion_cmd = 26;     // command(q_ref13 + qd_ref13) only; anchor is separate 9

    int history_length_ = 10;     // overwritten from ONNX shape
    int policy_obs_dim_ = num_single_obs * 10;

    //////////////////////// Observation Buffers ////////////////////////
    std::vector<float> policy_frame_;                   // 80D single frame
    std::vector<float> policy_obs_hist_term_major_;     // 63*H frame-major
    bool policy_hist_initialized_ = false;

    // Critic obs (if needed)
    std::vector<float> critic_obs_;

    //////////////////////// processNoise (TOCABI sim2real pattern) ////////////////////////
    // is_on_robot_: true=real robot (direct sensor), false=sim (noise + numerical diff)
    bool is_on_robot_ = false;

    // Joint state used by BOTH obs and PD (matching TOCABI)
    Matrix<double, MODEL_DOF, 1> q_noise_;       // joint position (noised in sim, direct on robot)
    Matrix<double, MODEL_DOF, 1> q_noise_pre_;   // previous step (for numerical diff)
    Matrix<double, MODEL_DOF, 1> q_vel_noise_;   // joint velocity (numerical diff in sim, direct on robot)

    double noise_time_cur_ = 0.0;
    double noise_time_pre_ = 0.0;
    bool noise_initialized_ = false;

    //////////////////////// Robot State ////////////////////////
    // Default joint positions. P73 JOINT_NAME order matches IsaacLab
    // ALL_JOINT_NAMES order (HipRoll, HipPitch, HipYaw, ..., WaistYaw).
    Matrix<double, MODEL_DOF, 1> q_default_p73_;

    // RL action (IsaacLab order, 13D: legs + WaistYaw)
    Matrix<double, num_action, 1> rl_action_;
    Matrix<double, num_action, 1> last_action_raw_;  // raw network output (for obs)

    // Custom PD gains for RL policy (self-contained in cc.cpp, not from yaml).
    // These match the stiffness/damping used during IsaacLab training.
    VectorQd kp_p73_, kd_p73_;
    VectorQd torque_bound_p73_;
    Matrix<double, num_action, 1> q_limit_lower_p73_, q_limit_upper_p73_;

    VectorQd torque_rl_;
    VectorQd torque_init_;
    VectorQd q_init_;
    VectorQd q_init_hold_;  // DEBUG: captured pose at mode entry
    VectorQd q_spline_;
    VectorQd torque_spline_;

    // 4-bar kinematics used in sim to reproduce motor-level torque clamping
    // through J^T (state_estimator does not populate rd_.four_bar_Jaco_ in simMode).
    FourBarKinematics sim_four_bar_;

    // Per-joint action scale (from WALKER_ACTION_SCALE in mimic_env_cfg.py)
    // q_target = q_default + clip(rl_action, ±2.0) * action_scales_[i]
    double action_scales_[num_action] = {
        0.319,  // L_HipRoll
        0.681,  // L_HipPitch
        0.429,  // L_HipYaw
        1.147,  // L_Knee
        0.484,  // L_AnklePitch
        0.231,  // L_AnkleRoll
        0.319,  // R_HipRoll
        0.681,  // R_HipPitch
        0.429,  // R_HipYaw
        1.147,  // R_Knee
        0.484,  // R_AnklePitch
        0.231,  // R_AnkleRoll
        0.863,  // WaistYaw
    };

    //////////////////////// Timing ////////////////////////
    float start_time_;
    float time_inference_pre_ = 0.0;
    bool cc_init_ = true;

    // Policy rate: 50Hz (decimation=4, dt=0.005 → policy_dt=0.02)
    double policy_dt_ = 0.02;

    // Motion reference (updated by ROS2 subscriber on /p73/motion_ref).
    // 33D pure reference (robot-state independent):
    //   [0:13]  q_ref       (ALL_JOINT order, rad)
    //   [13:26] qd_ref      (ALL_JOINT order, central-difference)
    //   [26:29] ref_root_pos  (motion frame, m)
    //   [29:33] ref_root_quat (wxyz)
    std::mutex motion_ref_mutex_;
    std::array<double, 33> motion_ref_{};

    //////////////////////// Anchor state (PLAN §2/§5, isaaclab wxyz) ////////////////////////
    // anchor transform T_a (ref frame -> deploy world):
    //   ref_pos_a  = anchor_q_ * ref_root_pos + anchor_t_
    //   ref_quat_a = anchor_q_ * ref_root_quat
    Eigen::Vector3d    anchor_t_{Eigen::Vector3d::Zero()};         // T_a translation
    Eigen::Quaterniond anchor_q_{Eigen::Quaterniond::Identity()}; // T_a rotation
    bool   anchor_inited_ = false;
    double anchor_leak_alpha_ = 0.0;   // MuJoCo=0 (no leak), real>0 (low-pass xy/yaw drift absorb)
    double anchor_xy_guard_ = 0.20;    // hard-guard threshold (m)

    double value_ = 0.0;

    string weight_dir_;

    //////////////////////// ROS2 Subscribers ////////////////////////
    // Uses dc_.node_ (main controller node) to share its DDS participant,
    // avoiding communication issues when running with sudo on real robot.
    rclcpp::CallbackGroup::SharedPtr vel_cbg_;
    rclcpp::executors::SingleThreadedExecutor vel_executor_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr motion_ref_sub_;
    std::thread vel_spin_thread_;
    std::atomic<bool> vel_spin_running_{false};

    void motionRefCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void startSubscribers();
    void stopSubscribers();

private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
};

#endif
