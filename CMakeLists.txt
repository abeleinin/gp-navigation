cmake_minimum_required(VERSION 3.0.2)
project(gp_navigation)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  rospy
  std_msgs
  nav_msgs
  visualization_msgs
  geometry_msgs
  actionlib_msgs
  roslaunch
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate actions in the 'action' folder
 add_action_files(
  FILES
  PathPlanning.action
  GPPointCloud.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  nav_msgs
  sensor_msgs
  actionlib_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

catkin_package(
  LIBRARIES gp_navigation
  CATKIN_DEPENDS nav_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(DIRECTORY worlds models launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
