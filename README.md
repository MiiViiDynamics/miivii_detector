# MiiVii Detector

MiiVii Detector is a ros node use mivii library to detect 2d object. It need synchronzied camera input, to use batch mode.

## MiiVii Detector User Guide

### Compile
Clone the repository and build:
```
    mkdir -p ~/catkin_ws/src & cd ~/catkin_ws/src 
    git clone https://github.com/MiiViiDynamics/miivii_detector
    cd ../..
    catkin_make_isolated
    source devel_isolated/setup.bash
```

### Run
Make sure the camera topic is set correctly in launch file.
The following launch file will match the topic in miivii_gmsl_ros node.

```
  roslaunch miivii_detector perception_front.launch
```


### Visualize
Use publish_result_image param in launch file to enable publich image with detection rect drawn.
```
  <param name="publish_result_image" value="true" />
```
Use rviz to check the result image topic.

### Configuration

## Contact
For technology issue, please file bugs on github directly.
For busniess contact, you can either visit our [taobao shop](https://shop324175547.taobao.com/?spm=a230r.7195193.1997079397.2.3154636cYGG7Vj)
, or mail to bd#miivii.com