# Image Processing By Pose

## Description
*Image Processing By Pose* is a ROS package that uses RGB camera to detect and track humans. It can assign ids to the detected people, designed to handle total, ocasional obscurities. It is also usefull for drawing skeleton lines, given the detected points by openpose.


### Files Included:
1. **skeleton_visualiser:** This is the main node. It subscribes to openpose ROS msg and to RGB image topics. Then, the algorithm filters the detected skeletons and keeps those with high possibility of detection. It uses the filtered points (spine & shoulders) to calculate the histograms of every person. It uses the extracted histograms to compare the detected persons in two subsequent frames. Then, a matching process is taking place to map the old ids to the new ones. Finally, the node dosen't just match the ids between two subsequent frames but it compares newly appeared histograms to dissapeared ones for reidentification.

### Publishing Topics
* **/image_converter/output_video:** output image message with skeletons on it.
* **Histogram:** Histograms of people on the latest frame.
* **skeleton_points:** Shoulder and spine points on the RGB image. It contains all the points on the line of the edges given by the openpose.

### Dependencies
All the dependenses sould be included automatically (like openCV, for instance). If one wants to run the code live without the use of custom rosbags, openpose and it's ROS wrapper 'https://github.com/firephinx/openpose_ros' should be installed.


## Usage
To run the demo, follow these steps:
1. Clone the package in your src file.
2. Get some rosbags with RGB and openpose topics (available at the lab). Otherwise, make sure you have RGB camera and openpose ROS upand running.
2. cd to your workspace diractory and run catkin_make (let's assume you have sourced devel/setup.bash)
3. Run roscore
4. Run rosrun image_processing_by_pose skeleton_visualiser


## License
[IIT DEMOCRITOS](https://www.iit.demokritos.gr/)

This package was developed and tested for `ROS Melodic`.