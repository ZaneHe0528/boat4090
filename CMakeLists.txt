cmake_minimum_required(VERSION 3.8)
project(vel_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)

# 添加可执行文件
add_executable(vel_control src/vel_control.cpp)
ament_target_dependencies(vel_control 
  rclcpp 
  geometry_msgs 
  mavros_msgs
)

# 安装
install(TARGETS vel_control
  DESTINATION lib/${PROJECT_NAME}
)

# 导出和包配置
ament_package()