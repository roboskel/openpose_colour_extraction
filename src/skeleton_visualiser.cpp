#include <iostream>
#include <vector>
#include <tuple>
#include <bits/stdc++.h>
#include "std_msgs/Float32.h"
#include "std_msgs/Int16.h"

#include <string>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"

#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <openpose_ros_msgs/BoundingBox.h>
#include <openpose_ros_msgs/OpenPoseHuman.h>
#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>

#include <pcl/point_cloud.h>
#include "pcl_ros/point_cloud.h"

#include "image_processing_by_pose/SkeletonLines.h"
#include "image_processing_by_pose/PointXYRGB.h"
#include "image_processing_by_pose/Skeletons.h"



/*
This node attempts to visualise the OpenPose data streamed by a bag without the need of an openpose instalation.
 */

//typedef image_processing_by_pose my_msgs;

static const std::string OPENCV_WINDOW = "Skeleton Tracker Video";
std::vector<std::tuple<int, int>> tl;

struct Color
{
  int r, g, b, a;
};

class ImageConverter
{
  //Public atributes
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher skeleton_points_pub_;

  ros::Subscriber human_list_;
  std::vector<openpose_ros_msgs::OpenPoseHuman> humans;
  std::vector<int> tracking_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8}; //Default tracking ids
  std::vector<Color> color_of_ids = {{66, 245, 135}, {245, 66, 81}, {48, 65, 194}, {136, 77, 161}, {82, 220, 227}, {227, 176, 82}, {176, 105, 55}, {173, 181, 109}, {64, 133, 88}};

public:
  ImageConverter()
      : it_(nh_)
  {
    // Subscrive to input video feed & to OpenPoseHumanList and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &ImageConverter::imageCb, this);
    human_list_ = nh_.subscribe("/openpose_ros/human_list", 1, &ImageConverter::openposeCB, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    skeleton_points_pub_ = nh_.advertise<image_processing_by_pose::Skeletons>("/skeleton_points",1);

    cv::namedWindow(OPENCV_WINDOW);

    //Define point pairs that will be connected to form a skeleton.
    tl.push_back(std::make_tuple(0, 1));
    tl.push_back(std::make_tuple(17, 15));
    tl.push_back(std::make_tuple(0, 15));
    tl.push_back(std::make_tuple(0, 16));
    tl.push_back(std::make_tuple(16, 18));
    tl.push_back(std::make_tuple(1, 2));
    tl.push_back(std::make_tuple(2, 3));
    tl.push_back(std::make_tuple(3, 4));
    tl.push_back(std::make_tuple(1, 5));
    tl.push_back(std::make_tuple(5, 6));
    tl.push_back(std::make_tuple(6, 7));
    tl.push_back(std::make_tuple(1, 8));
    tl.push_back(std::make_tuple(8, 9));
    tl.push_back(std::make_tuple(9, 10));
    tl.push_back(std::make_tuple(11, 10));
    tl.push_back(std::make_tuple(11, 24));
    tl.push_back(std::make_tuple(11, 22));
    tl.push_back(std::make_tuple(22, 23));
    tl.push_back(std::make_tuple(8, 12));
    tl.push_back(std::make_tuple(12, 13));
    tl.push_back(std::make_tuple(13, 14));
    tl.push_back(std::make_tuple(14, 21));
    tl.push_back(std::make_tuple(14, 19));
    tl.push_back(std::make_tuple(19, 20));
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void openposeCB(const openpose_ros_msgs::OpenPoseHumanList::ConstPtr &msg)
  {
    //First clear the vector for the new data to take place.
    humans.clear();
    ROS_INFO("Number of recognised people: [%d]", msg->num_humans);
    humans.reserve(msg->num_humans);
    for (const auto &person : msg->human_list)
    {
      humans.push_back(person);
    }
    return;
  }
  void imageCb(const sensor_msgs::ImageConstPtr &msg)
  {
    ROS_INFO("Just got a new image!");
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    //Create a message of type skeletons to be published.
    image_processing_by_pose::Skeletons humansVect;

    // Draw a rectangle on every discovered skeleton.
    std::vector<int>::iterator itIds = tracking_ids.begin();
    for (std::vector<openpose_ros_msgs::OpenPoseHuman>::iterator itPersona = humans.begin(); itPersona != humans.end(); ++itPersona) //Loop for every skeleton.
    {
      //Carefull! Bounding box (x,y) point is on the top left corner.
      //cv::rectangle(image, cornerPoint1, cornerPoint2, Color [, ...])
      openpose_ros_msgs::BoundingBox tempBox = itPersona->body_bounding_box;
      cv::rectangle(cv_ptr->image, cv::Point(tempBox.x, tempBox.y), cv::Point(tempBox.x + tempBox.width, tempBox.y + tempBox.height),
                    CV_RGB(color_of_ids[*itIds].r, color_of_ids[*itIds].g, color_of_ids[*itIds].b));


      //Draw the dots.
      for (int i = 0; i < (itPersona->body_key_points_with_prob).size(); ++i)
      {
        cv::circle(cv_ptr->image, cv::Point(itPersona->body_key_points_with_prob[i].x, itPersona->body_key_points_with_prob[i].y), 2,
                   CV_RGB(color_of_ids[*itIds].r, color_of_ids[*itIds].g, color_of_ids[*itIds].b));
      }

      //Connect the dots to form a skeleton.
      for (std::vector<std::tuple<int, int>>::iterator it = tl.begin(); it != tl.end(); ++it)
      {
        //Do not draw lines with zero probability.
        if (itPersona->body_key_points_with_prob[std::get<1>(*it)].prob == 0.0 || itPersona->body_key_points_with_prob[std::get<0>(*it)].prob == 0.0)
          continue;
        cv::Point edge1 = cv::Point(itPersona->body_key_points_with_prob[std::get<0>(*it)].x, itPersona->body_key_points_with_prob[std::get<0>(*it)].y);
        cv::Point edge2 = cv::Point(itPersona->body_key_points_with_prob[std::get<1>(*it)].x, itPersona->body_key_points_with_prob[std::get<1>(*it)].y);
        //Store the points of the line in a vector to publish them later.
        image_processing_by_pose::SkeletonLines bodyPart;
        
        if (  (std::get<0>(*it) == 1 || std::get<1>(*it) == 1) && (std::get<0>(*it) != 0 && std::get<1>(*it) != 0 ) ){ //We only publish the spine and the sholders.
          cv::LineIterator lineIt(cv_ptr->image, edge1, edge2, 8);

          for(int i = 0; i < lineIt.count; i++, ++lineIt) //Store the image points of the line. Convert them to pcl points.
          {
            // OLD pointcloud version: pcl::PointXYZRGB imagePoint;
            image_processing_by_pose::PointXYRGB imagePoint;
            
            cv::Vec3b color = *(const cv::Vec3b*)*lineIt;
            imagePoint.x = lineIt.pos().x; imagePoint.y = lineIt.pos().y; imagePoint.r = color.val[0]; imagePoint.g = color.val[1]; imagePoint.b = color.val[2];

            bodyPart.linePoints.push_back(imagePoint);

            //Mark the points between edge1 and edge2.
            cv::Point pt= lineIt.pos(); 
            cv::circle(cv_ptr->image, pt, 2, CV_RGB(color_of_ids[*itIds].r, color_of_ids[*itIds].g, color_of_ids[*itIds].b));
              
          }
        //Store the body part in the vector of pcls.
        humansVect.Person.push_back(bodyPart);
        }


        //Drow the line
        cv::line(cv_ptr->image, edge1, edge2, CV_RGB(color_of_ids[*itIds].r, color_of_ids[*itIds].g, color_of_ids[*itIds].b));
      }
      std::string ID("ID ");
      ID += std::to_string(*itIds);
      //Write the id each person has on the image.
      cv::putText(cv_ptr->image, ID, cv::Point(tempBox.x, tempBox.y), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(color_of_ids[*itIds].r, color_of_ids[*itIds].g, color_of_ids[*itIds].b));
      //length(tracking_ids) should not be less than length(humans)!
      ++itIds;
    }


    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(4);

    // Publish the vector of the underlying skeleton pixels of the image.
    skeleton_points_pub_.publish(humansVect);

    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "skeleton_visualiser");
  ImageConverter ic;
  ros::spin();
  return 0;
}
