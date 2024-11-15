cmake_minimum_required(VERSION 3.0.2)
project(yellow_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  realsense2_camera
)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge image_transport realsense2_camera
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(yellow_detector_node src/yellow_detector_node.cpp)
target_link_libraries(yellow_detector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
