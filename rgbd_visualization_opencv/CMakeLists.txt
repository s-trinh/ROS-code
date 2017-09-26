cmake_minimum_required(VERSION 2.8.3)
project(rgbd_visualization_opencv)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

find_package(Boost REQUIRED COMPONENTS 
  thread
)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES beginner_tutorials
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs
)

###########
## Build ##
###########

# Set compile flags
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall ${CMAKE_CXX_FLAGS}")

set(test_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rgbd_visualization_opencv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/rgbid_visualization_opencv.cpp
)

foreach(cpp ${test_cpp})
  get_filename_component(target ${cpp} NAME_WE)
  include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
  add_executable(${target} ${cpp})
  target_link_libraries(${target} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${Boost_LIBRARIES})
endforeach()
