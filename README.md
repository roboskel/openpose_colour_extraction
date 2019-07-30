# Image Processing By Pose

## Description
*Image Processing By Pose* is a ROS package that uses RGB camera to detect and track humans. Capable of assigning ids to the detected people while being designed to handle total, ocasional obscurities. It is also a usefull tool for drawing skeleton lines, given the detected points by openpose.


### Files Included:
1. **skeleton_visualiser:** This is the main node. It subscribes to openpose ROS msg and to RGB image topics. Then, the algorithm filters the detected skeletons and keeps those with high possibility of detection. It uses the filtered points (spine & shoulders) to precisely mask each person and extract visual features (calculate the histograms of every person). It uses the extracted histograms to compare the detected persons in two subsequent frames. Then, a matching process is taking place to map the old ids to the new ones. Finally, the node dosen't just match the ids between two subsequent frames but it compares newly appeared histograms to dissapeared ones for reidentification.
2. **msg:** This file contains various custom messages to store the image points along the spine and the shoulders of the detected persons.

### Publishing Topics
* **/image_converter/output_video:** output image message with skeletons on it.
* **Histogram:** Histograms of people on the latest frame.
* **skeleton_points:** Shoulder and spine points on the RGB image. It contains all the points on the line of the edges given by the openpose.

### Dependencies
For the easy setup (use of rosbags) the only dependency is the 'openpose_ros_msgs' package. This can be found here: https://github.com/firephinx/openpose_ros .
All the other dependenses sould be included automatically (like openCV, for instance). 
If one wants to run the code live without the use of custom rosbags, openpose and it's ROS wrapper https://github.com/firephinx/openpose_ros should be installed.


## Usage
To run the code, follow these steps:
1. Clone the package in your src file.
2. Get some rosbags with RGB and openpose topics (available at the lab or upon request). Otherwise, make sure you have RGB camera and openpose ROS up and running.
2. cd to your workspace diractory and run catkin_make (let's assume you have sourced devel/setup.bash)
3. Run roscore
4. Run rosrun image_processing_by_pose skeleton_visualiser


## License
[IIT DEMOCRITOS](https://www.iit.demokritos.gr/)

## Version
This package was developed and tested for `ROS Melodic`.
