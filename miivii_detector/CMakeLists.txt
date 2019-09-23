cmake_minimum_required(VERSION 3.5)
project(miivii_detector)


#check the system version
set(MAKE_CMD "cat /etc/miivii_release | awk -F \"  \" '{print $1}' | tr '[A-Z]' '[a-z]'")
execute_process(COMMAND bash "-c" "${MAKE_CMD}" -E  OUTPUT_VARIABLE HDVERSION)
if (${HDVERSION} MATCHES "4.2.0")
  set(JPVERSION 420)
  message(WARN "JPVERSION is ${JPVERSION}")
else()
  set(JPVERSION 421)
  message(WARN "JPVERSION is ${JPVERSION}")
endif()

set(CMAKE_CXX_FLAGS " -Wall -std=c++11 -O0 -DJPVERSION=${JPVERSION}")

set (CMAKE_CXX_STANDARD 11)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
             image_transport
             roscpp
             rospy
             std_msgs
             std_srvs
             sensor_msgs
             cv_bridge
             autoware_msgs
             autoware_config_msgs)

find_package(OpenCV REQUIRED )
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

###################################################
## Declare things to be passed to other projects ##
###################################################

## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

catkin_package(CATKIN_DEPENDS
        cv_bridge
        image_transport
        roscpp
        sensor_msgs
        std_msgs
        autoware_msgs
        autoware_config_msgs
        )
#catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ${PROJECT_NAME}
#)

###########
## Build ##
###########
set(linux_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_path(MIIVIIYOLO_INCLUDES MiiViiYoloSDKInterface.h
          /opt/miivii/include)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${MIIVIIYOLO_INCLUDES}
  ${linux_INCLUDE_DIR}
)

link_directories( ${catkin_LIB_DIRS} /opt/miivii/lib/)
## Declare a cpp executable
set(NODE_NAME ${PROJECT_NAME}_node)
add_executable(${PROJECT_NAME}_node src/miivii_yolo_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${catkin_LIB_DIRS} miivii_accelerator
)

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

## Copy launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
