// #include <ros/ros.h>
// #include <geometry_msgs/Twist.h>
// #include <mavros_msgs/CommandBool.h>
// #include <mavros_msgs/SetMode.h>
// #include <mavros_msgs/State.h>
// #include <mavros_msgs/ParamSet.h>

// mavros_msgs::State current_state;

// void stateCallback(const mavros_msgs::State::ConstPtr& msg)
// {
//     current_state = *msg;
// }

// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "vel_control");
//     ros::NodeHandle nh("~");

//     double linear_x, angular_z, publish_rate;
//     nh.param("linear_x", linear_x, 1.0);
//     nh.param("angular_z", angular_z, 0.0);
//     nh.param("publish_rate", publish_rate, 20.0);

//     ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>(
//         "/boat_0/mavros/state", 10, stateCallback);

//     ros::Publisher vel_pub = nh.advertise<geometry_msgs::Twist>(
//         "/boat_0/mavros/setpoint_velocity/cmd_vel_unstamped", 10);

//     ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>(
//         "/boat_0/mavros/cmd/arming");

//     ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>(
//         "/boat_0/mavros/set_mode");

//     ros::ServiceClient param_set_client = nh.serviceClient<mavros_msgs::ParamSet>(
//         "/boat_0/mavros/param/set");

//     ros::Rate rate(publish_rate);

//     while (ros::ok() && !current_state.connected) {
//         ROS_INFO_THROTTLE(1, "Waiting for MAVROS connection...");
//         ros::spinOnce();
//         rate.sleep();
//     }
//     ROS_INFO("MAVROS connected!");

//     // 设置 COM_RCL_EXCEPT，允许无遥控器时进入 OFFBOARD
//     mavros_msgs::ParamSet rcl_param;
//     rcl_param.request.param_id = "COM_RCL_EXCEPT";
//     rcl_param.request.value.integer = 4;
//     if (param_set_client.call(rcl_param) && rcl_param.response.success) {
//         ROS_INFO("COM_RCL_EXCEPT set to 4");
//     }

//     geometry_msgs::Twist cmd;
//     cmd.linear.x = linear_x;
//     cmd.linear.y = 0.0;
//     cmd.linear.z = 0.0;
//     cmd.angular.x = 0.0;
//     cmd.angular.y = 0.0;
//     cmd.angular.z = angular_z;

//     // PX4 要求切 OFFBOARD 前先收到 setpoint 流
//     for (int i = 0; ros::ok() && i < 100; i++) {
//         vel_pub.publish(cmd);
//         ros::spinOnce();
//         rate.sleep();
//     }
//     ROS_INFO("Setpoint stream established.");

//     mavros_msgs::SetMode offboard_mode;
//     offboard_mode.request.custom_mode = "OFFBOARD";

//     mavros_msgs::CommandBool arm_cmd;
//     arm_cmd.request.value = true;

//     ros::Time last_request = ros::Time::now();

//     ROS_INFO("Velocity control: linear.x=%.2f m/s, angular.z=%.2f rad/s, rate=%.1f Hz",
//              linear_x, angular_z, publish_rate);

//     while (ros::ok()) {
//         bool need_offboard = (current_state.mode != "OFFBOARD");
//         bool need_arm = !current_state.armed;

//         if ((need_offboard || need_arm) &&
//             (ros::Time::now() - last_request > ros::Duration(2.0))) {

//             if (need_offboard) {
//                 if (set_mode_client.call(offboard_mode) &&
//                     offboard_mode.response.mode_sent) {
//                     ROS_INFO("OFFBOARD mode requested");
//                 }
//             }

//             if (need_arm) {
//                 if (arming_client.call(arm_cmd) &&
//                     arm_cmd.response.success) {
//                     ROS_INFO("Vehicle armed!");
//                 } else {
//                     ROS_WARN("Arming request sent (may need retry)");
//                 }
//             }

//             last_request = ros::Time::now();
//         }

//         vel_pub.publish(cmd);
//         ros::spinOnce();
//         rate.sleep();
//     }

//     return 0;
// }



#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <mavros_msgs/srv/param_set.hpp>

class VelControlNode : public rclcpp::Node
{
public:
    VelControlNode()
        : Node("vel_control")
    {
        // 声明参数
        this->declare_parameter<double>("linear_x", 1.0);
        this->declare_parameter<double>("angular_z", 0.0);
        this->declare_parameter<double>("publish_rate", 20.0);

        double linear_x = this->get_parameter("linear_x").as_double();
        double angular_z = this->get_parameter("angular_z").as_double();
        double publish_rate = this->get_parameter("publish_rate").as_double();

        // 创建订阅者
        state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
            "/boat_0/mavros/state", 10,
            std::bind(&VelControlNode::stateCallback, this, std::placeholders::_1));

        // 创建发布者
        vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/boat_0/mavros/setpoint_velocity/cmd_vel_unstamped", 10);

        // 创建服务客户端
        arming_client_ = this->create_client<mavros_msgs::srv::CommandBool>(
            "/boat_0/mavros/cmd/arming");
        set_mode_client_ = this->create_client<mavros_msgs::srv::SetMode>(
            "/boat_0/mavros/set_mode");
        param_set_client_ = this->create_client<mavros_msgs::srv::ParamSet>(
            "/boat_0/mavros/param/set");

        // 等待服务可用
        while (!arming_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for arming service.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for arming service...");
        }
        
        while (!set_mode_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for set_mode service.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for set_mode service...");
        }
        
        while (!param_set_client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for param_set service.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for param_set service...");
        }

        // 等待MAVROS连接
        RCLCPP_INFO(this->get_logger(), "Waiting for MAVROS connection...");
        while (rclcpp::ok() && !current_state_.connected) {
            rclcpp::spin_some(this->shared_from_this());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        RCLCPP_INFO(this->get_logger(), "MAVROS connected!");

        // 设置 COM_RCL_EXCEPT，允许无遥控器时进入 OFFBOARD
        setComRclExcept();

        // 创建命令消息
        cmd_.linear.x = linear_x;
        cmd_.linear.y = 0.0;
        cmd_.linear.z = 0.0;
        cmd_.angular.x = 0.0;
        cmd_.angular.y = 0.0;
        cmd_.angular.z = angular_z;

        // PX4 要求切 OFFBOARD 前先收到 setpoint 流
        RCLCPP_INFO(this->get_logger(), "Establishing setpoint stream...");
        auto sleep_duration = std::chrono::duration<double>(1.0 / publish_rate);
        for (int i = 0; rclcpp::ok() && i < 100; i++) {
            vel_pub_->publish(cmd_);
            rclcpp::spin_some(this->shared_from_this());
            std::this_thread::sleep_for(sleep_duration);
        }
        RCLCPP_INFO(this->get_logger(), "Setpoint stream established.");

        RCLCPP_INFO(this->get_logger(), "Velocity control: linear.x=%.2f m/s, angular.z=%.2f rad/s, rate=%.1f Hz",
                    linear_x, angular_z, publish_rate);

        // 创建定时器
        auto timer_period = std::chrono::duration<double>(1.0 / publish_rate);
        timer_ = this->create_wall_timer(
            timer_period,
            std::bind(&VelControlNode::controlLoop, this));
    }

private:
    void stateCallback(const mavros_msgs::msg::State::SharedPtr msg)
    {
        current_state_ = *msg;
    }

    void setComRclExcept()
    {
        auto request = std::make_shared<mavros_msgs::srv::ParamSet::Request>();
        request->param_id = "COM_RCL_EXCEPT";
        request->value.integer = 4;

        auto result = param_set_client_->async_send_request(request);
        // 等待结果（可选，这里简单等待）
        auto future_result = rclcpp::spin_until_future_complete(this->shared_from_this(), result);
        if (future_result == rclcpp::FutureReturnCode::SUCCESS) {
            if (result.get()->success) {
                RCLCPP_INFO(this->get_logger(), "COM_RCL_EXCEPT set to 4");
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to set COM_RCL_EXCEPT");
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "COM_RCL_EXCEPT request timeout or failed");
        }
    }

    void controlLoop()
    {
        bool need_offboard = (current_state_.mode != "OFFBOARD");
        bool need_arm = !current_state_.armed;

        if ((need_offboard || need_arm) &&
            (this->now() - last_request_ > rclcpp::Duration(2, 0))) {

            if (need_offboard) {
                auto offboard_request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
                offboard_request->custom_mode = "OFFBOARD";
                
                auto offboard_result = set_mode_client_->async_send_request(offboard_request);
                // 异步请求，不等待结果
                RCLCPP_INFO(this->get_logger(), "OFFBOARD mode requested");
            }

            if (need_arm) {
                auto arm_request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
                arm_request->value = true;
                
                auto arm_result = arming_client_->async_send_request(arm_request);
                // 异步请求，不等待结果
                RCLCPP_INFO(this->get_logger(), "Arming request sent");
            }

            last_request_ = this->now();
        }

        vel_pub_->publish(cmd_);
    }

    // ROS 2 成员变量
    rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr vel_pub_;
    rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
    rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
    rclcpp::Client<mavros_msgs::srv::ParamSet>::SharedPtr param_set_client_;
    rclcpp::TimerBase::SharedPtr timer_;

    mavros_msgs::msg::State current_state_;
    geometry_msgs::msg::Twist cmd_;
    rclcpp::Time last_request_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VelControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}