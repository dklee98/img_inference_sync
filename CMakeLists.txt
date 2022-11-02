cmake_minimum_required(VERSION 3.0.2)
project(kt_ros)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    nav_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    message_generation
    )

add_message_files(
    FILES
    node.msg
    )

generate_messages(
    DEPENDENCIES
    std_msgs
    nav_msgs
    geometry_msgs
    sensor_msgs
   )

