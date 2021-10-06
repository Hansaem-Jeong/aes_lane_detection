cmake_minimum_required(VERSION 3.0.2)
project(lane_detection)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  rospy
  std_msgs
  sensor_msgs
  message_generation
  ros_numpy
)

find_package(OpenCV 4 REQUIRED)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
 )


catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}

)

catkin_install_python(PROGRAMS
   scripts/lane_detection_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
