#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <cmath>

mavros_msgs::State current_state;
geometry_msgs::PoseStamped current_pose;
bool pose_received = false;

void stateCallback(const mavros_msgs::State::ConstPtr& msg)
{
    current_state = *msg;
}

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    current_pose = *msg;
    pose_received = true;
}

double getYawFromQuaternion(double x, double y, double z, double w)
{
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "boat_control");
    ros::NodeHandle nh("~");

    double linear_speed, publish_rate, lookahead_dist, kp_y;
    nh.param("linear_speed", linear_speed, 1.0);
    nh.param("publish_rate", publish_rate, 20.0);
    nh.param("lookahead_dist", lookahead_dist, 20.0);
    nh.param("kp_y", kp_y, 0.5);

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>(
        "/boat_0/mavros/state", 10, stateCallback);

    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(
        "/boat_0/mavros/local_position/pose", 10, poseCallback);

    ros::Publisher setpoint_pub = nh.advertise<mavros_msgs::PositionTarget>(
        "/boat_0/mavros/setpoint_raw/local", 10);

    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>(
        "/boat_0/mavros/cmd/arming");

    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>(
        "/boat_0/mavros/set_mode");

    ros::Rate rate(publish_rate);

    while (ros::ok() && !current_state.connected) {
        ROS_INFO_THROTTLE(1, "Waiting for MAVROS connection...");
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("MAVROS connected!");

    while (ros::ok() && !pose_received) {
        ROS_INFO_THROTTLE(1, "Waiting for pose data...");
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Pose data received. Initial position: x=%.2f, y=%.2f",
             current_pose.pose.position.x, current_pose.pose.position.y);

    // 记录初始航向：船出生时朝向 Gazebo +X，此时的 EKF yaw 就是 Gazebo X 轴在 ENU 中的角度
    double initial_yaw = getYawFromQuaternion(
        current_pose.pose.orientation.x,
        current_pose.pose.orientation.y,
        current_pose.pose.orientation.z,
        current_pose.pose.orientation.w);
    double cos_yaw0 = std::cos(initial_yaw);
    double sin_yaw0 = std::sin(initial_yaw);
    ROS_INFO("Initial yaw (Gazebo X direction in ENU): %.2f deg", initial_yaw * 180.0 / M_PI);

    mavros_msgs::PositionTarget setpoint;
    setpoint.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    // 使用速度 + yaw，忽略位置和加速度
    setpoint.type_mask = mavros_msgs::PositionTarget::IGNORE_PX |
                         mavros_msgs::PositionTarget::IGNORE_PY |
                         mavros_msgs::PositionTarget::IGNORE_PZ |
                         mavros_msgs::PositionTarget::IGNORE_AFX |
                         mavros_msgs::PositionTarget::IGNORE_AFY |
                         mavros_msgs::PositionTarget::IGNORE_AFZ |
                         mavros_msgs::PositionTarget::IGNORE_YAW_RATE;

    // 将 Gazebo X 轴方向的速度分解到 ENU 坐标系
    setpoint.velocity.x = linear_speed * cos_yaw0;
    setpoint.velocity.y = linear_speed * sin_yaw0;
    setpoint.velocity.z = 0.0;
    setpoint.yaw = initial_yaw;

    ROS_INFO("Velocity in ENU frame: vx=%.3f, vy=%.3f, yaw=%.2f deg",
             setpoint.velocity.x, setpoint.velocity.y, initial_yaw * 180.0 / M_PI);

    // PX4 要求切换 OFFBOARD 前先收到 setpoint 流
    for (int i = 0; ros::ok() && i < 100; i++) {
        setpoint_pub.publish(setpoint);
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Setpoint stream established.");

    mavros_msgs::SetMode offboard_mode;
    offboard_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    ROS_INFO("Commanding constant velocity %.2f m/s along Gazebo X axis", linear_speed);

    while (ros::ok()) {
        bool need_offboard = (current_state.mode != "OFFBOARD");
        bool need_arm = !current_state.armed;

        if ((need_offboard || need_arm) &&
            (ros::Time::now() - last_request > ros::Duration(2.0))) {

            if (need_offboard) {
                if (set_mode_client.call(offboard_mode) &&
                    offboard_mode.response.mode_sent) {
                    ROS_INFO("OFFBOARD mode requested");
                }
            }

            if (need_arm) {
                if (arming_client.call(arm_cmd) &&
                    arm_cmd.response.success) {
                    ROS_INFO("Vehicle armed!");
                } else {
                    ROS_WARN("Arming request sent (may need retry)");
                }
            }

            last_request = ros::Time::now();
        }

        // 计算当前位置在 Gazebo X 轴垂直方向的偏差（横向偏移）
        double cur_x = current_pose.pose.position.x;
        double cur_y = current_pose.pose.position.y;
        // 横向偏差：当前位置到 Gazebo X 轴线的距离（带符号）
        double lateral_error = -sin_yaw0 * cur_x + cos_yaw0 * cur_y;

        // 主速度沿 Gazebo X 轴 + 横向修正
        double vx_corr = -kp_y * lateral_error * (-sin_yaw0);
        double vy_corr = -kp_y * lateral_error * cos_yaw0;

        setpoint.velocity.x = linear_speed * cos_yaw0 + vx_corr;
        setpoint.velocity.y = linear_speed * sin_yaw0 + vy_corr;

        ROS_INFO_THROTTLE(5, "Pos: (%.2f, %.2f), lateral_err: %.3f, vel: (%.3f, %.3f)",
                          cur_x, cur_y, lateral_error,
                          setpoint.velocity.x, setpoint.velocity.y);

        setpoint_pub.publish(setpoint);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
