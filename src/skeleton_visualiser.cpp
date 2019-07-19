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
#include <opencv2/core/types_c.h>

#include <openpose_ros_msgs/BoundingBox.h>
#include <openpose_ros_msgs/OpenPoseHuman.h>
#include <openpose_ros_msgs/OpenPoseHumanList.h>
#include <openpose_ros_msgs/PointWithProb.h>

#include "image_processing_by_pose/SkeletonLines.h"
#include "image_processing_by_pose/PointXYRGB.h"
#include "image_processing_by_pose/Skeletons.h"

#include <math.h>


#define NUM_OF_BINS 50 //Max: 256

using namespace std;

/*
This node attempts to visualise the OpenPose data streamed by a bag without the need of an openpose instalation.
 */

//typedef image_processing_by_pose my_msgs;

static const std::string OPENCV_WINDOW = "Skeleton Tracker Video";
static const std::string OPENCV_HISTOGRAM = "Histogram";
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
  image_transport::Publisher histogram_pub_;
  ros::Publisher skeleton_points_pub_;

  std::vector<std::vector<cv::Point>> humanEdges; //Store the skeleton points that we want to consider for the mask of the histogram
  ros::Subscriber human_list_;
  std::vector<openpose_ros_msgs::OpenPoseHuman> humans;
  std::vector<int> tracking_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8}; //Default tracking ids
  std::vector<Color> color_of_ids = {{66, 245, 135}, {245, 66, 81}, {48, 65, 194}, {136, 77, 161}, {82, 220, 227}, {227, 176, 82}, {176, 105, 55}, {173, 181, 109}, {64, 133, 88}};
  cv_bridge::CvImagePtr cv_ptr;
  cv_bridge::CvImage hist_msg;

public:
  ImageConverter()
      : it_(nh_)
  {
    // Subscrive to input video feed & to OpenPoseHumanList and publish output video feed
    image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &ImageConverter::imageCb, this);
    human_list_ = nh_.subscribe("/openpose_ros/human_list", 1, &ImageConverter::openposeCB, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);
    histogram_pub_ = it_.advertise("/Histogram",1);
    skeleton_points_pub_ = nh_.advertise<image_processing_by_pose::Skeletons>("/skeleton_points", 1);

    cv::namedWindow(OPENCV_WINDOW);
    cv::namedWindow(OPENCV_HISTOGRAM, CV_WINDOW_AUTOSIZE);

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
    cv::destroyWindow(OPENCV_HISTOGRAM);
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


  cv::Mat MaskCalculation(std::vector<cv::Point> &humanEdgesForMask)
  {
    //Creates a mask of one persons skeleton. This mask will be used by openCV to calculate the histogram and other color features.
    cv::Mat mask = cv::Mat::zeros(cv_ptr->image.size(), CV_8U);
    for (std::vector<cv::Point>::iterator itEdges = humanEdgesForMask.begin(); itEdges != humanEdgesForMask.end(); itEdges++)
    {
      cv::Point edge1 = *itEdges;
      ++itEdges;
      if (itEdges == humanEdgesForMask.end())
        break;
      cv::Point edge2 = *itEdges;

      //ROS_INFO("Two edges! : Point1: x[%d] y[%d]  Point2:  x[%d] y[%d]", edge1.x,edge1.y,edge2.x, edge2.y);

      cv::line(mask, edge1, edge2, CV_RGB(255, 255, 255), 1); //You may twick the thickness of the line. The more thick, the more points on the image.
    }
    return mask;
  }


  cv::Mat HSHistogramAndDraw(const cv::Mat& mask)
{
    cv::Mat hsvImage;
    cv::cvtColor(cv_ptr->image, hsvImage, CV_BGR2HSV);

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    cv::MatND histHS;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    cv::calcHist( &hsvImage, 1, channels, mask, // do not use mask
             histHS, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
    double maxVal=0;
    cv::minMaxLoc(histHS, 0, &maxVal, 0, 0);

    int scale = 10;
    cv::Mat histImg = cv::Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

    for( int h = 0; h < hbins; h++ ){
        for( int s = 0; s < sbins; s++ )
        {
            float binVal = histHS.at<float>(h, s);
            int intensity = std::round(binVal*255/maxVal);
            cv::rectangle( histImg, cv::Point(h*scale, s*scale),
                        cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                        cv::Scalar::all(intensity),
                        CV_FILLED );
        }
    }


    cv::imshow( OPENCV_HISTOGRAM, histImg );
    cv::waitKey(4);

    return histHS;
}

  cv::Mat ThreeDimensionalColorHistogram(const cv::Mat& mask)
  {
    int imgCount = 1;
    int dims = 3;
    const int sizes[] = {256, 256, 256};
    const int channels[] = {0, 1, 2};
    float rRange[] = {0, 256};
    float gRange[] = {0, 256};
    float bRange[] = {0, 256};
    const float *ranges[] = {bRange, gRange, rRange};
    cv::Mat hist3D;
    cv::calcHist(&cv_ptr->image, imgCount, channels, mask, hist3D, dims, sizes, ranges);
    
    return hist3D;
  }

  std::vector<cv::Mat> ColorHistogram(const cv::Mat& mask)
  {
    //Calculate the color histogram for the given mask on the frame (using 'global' variable cv_ptr->image).
    //Returns an array of histograms B, G, R.

    // Separate the image in 3 places ( B, G and R ).
    std::vector<cv::Mat> bgr_planes;
    cv::split(cv_ptr->image, bgr_planes);

    // Establish the number of bins.
    int histSize = NUM_OF_BINS; 

    // Set the ranges ( for (B,G,R) )
    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;
    bool accumulate = false;
    cv::Mat b_hist, g_hist, r_hist;

    // Compute the histograms:
    cv::calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);

    std::vector<cv::Mat> histograms = {b_hist,g_hist,r_hist};
    return histograms;

  }


  cv::Mat DrawHistogram3chanels(const cv::Mat &b_hist, const cv::Mat &g_hist, const cv::Mat &r_hist,int histSize = 256, int hist_w = 512, int hist_h = 400)
  {
    //Bin width
    int bin_w = std::round((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Normalize the result to [ 0, histImage.rows ].
    cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Draw for each channel
    for (int i = 1; i < histSize; i++)
    {
      cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - std::round(b_hist.at<float>(i - 1))),
               cv::Point(bin_w * (i), hist_h - std::round(b_hist.at<float>(i))),
               cv::Scalar(255, 0, 0), 2, 8, 0);
      cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - std::round(g_hist.at<float>(i - 1))),
               cv::Point(bin_w * (i), hist_h - std::round(g_hist.at<float>(i))),
               cv::Scalar(0, 255, 0), 2, 8, 0);
      cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - std::round(r_hist.at<float>(i - 1))),
               cv::Point(bin_w * (i), hist_h - std::round(r_hist.at<float>(i))),
               cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    // Display
    cv::imshow(OPENCV_HISTOGRAM, histImage);
    cv::waitKey(4);

    return histImage;
  }

  cv::Mat DrawHistogram1chanel(const cv::Mat &one_hist, int histSize = 256, int hist_w = 512, int hist_h = 400)
  {
    //Bin width
    int bin_w = std::round((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Normalize the result to [ 0, histImage.rows ].
    cv::normalize(one_hist, one_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Draw for one channel
    for (int i = 1; i < histSize; i++)
    {
      cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - std::round(one_hist.at<float>(i - 1))),
               cv::Point(bin_w * (i), hist_h - std::round(one_hist.at<float>(i))),
               cv::Scalar(255, 255, 255), 2, 8, 0);
    }

    // Display
    cv::imshow(OPENCV_HISTOGRAM, histImage);
    cv::waitKey(4);
    return histImage;
  }
  cv::Scalar MeanValue(const cv::Mat& mask){
    return  cv::mean(cv_ptr->image, mask);
  }

  void VisualFeatures()
  {
    //For every Skeleton in the frame we find a mask and pass it to functions like 3DHistogram etc. .  Then, we publish the desired histogram.
    for (std::vector<cv::Point> oneHuman : humanEdges)
    {
      cv::Mat h_b, h_g, h_r;
      const cv::Mat& mask = MaskCalculation(oneHuman);
      //const std::vector<cv::Mat>& histVect  = ColorHistogram(mask);  const cv::Mat& hist = histVect[0]; //Other options: ... = histVect[1] or ... = histVect[2]
      //const cv::Mat& hist = ThreeDimensionalColorHistogram(mask);
      //const cv::Mat& hist = HSHistogramAndDraw(mask);

      //Want to draw a histogram live? Uncomment the first line.
      //DrawHistogram3chanels(histVect[0], histVect[1], histVect[2], NUM_OF_BINS);
      //DrawHistogram1chanel(hist[0], NUM_OF_BINS);

      //Publish the desired histogram.
      // hist_msg.header   = //not set yet
      // hist_msg.encoding = //not set yet
      //hist_msg.image = hist; 
      //histogram_pub_.publish(hist_msg.toImageMsg());

    }
    return;
  }

  void imageCb(const sensor_msgs::ImageConstPtr &msg)
  {
    ROS_INFO("Just got a new image!");

    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    //Get the skeleton points and store them.
    //We need those points for the feature extraction (e.g. histogram) before we draw lines on the image. That's why we have two main loops on this loop.
    for (std::vector<openpose_ros_msgs::OpenPoseHuman>::iterator itPersona = humans.begin(); itPersona != humans.end(); ++itPersona) //Loop for every skeleton (human).
    {
      std::vector<cv::Point> oneHumanEdges;
      for (std::vector<std::tuple<int, int>>::iterator it = tl.begin(); it != tl.end(); ++it) //Iterate for all points on a single detected skeleton (human).
      {
        //Ignore points with zero probability.
        if (itPersona->body_key_points_with_prob[std::get<1>(*it)].prob == 0.0 || itPersona->body_key_points_with_prob[std::get<0>(*it)].prob == 0.0)
          continue;
        cv::Point edge1 = cv::Point(itPersona->body_key_points_with_prob[std::get<0>(*it)].x, itPersona->body_key_points_with_prob[std::get<0>(*it)].y);
        cv::Point edge2 = cv::Point(itPersona->body_key_points_with_prob[std::get<1>(*it)].x, itPersona->body_key_points_with_prob[std::get<1>(*it)].y);

        if ((std::get<0>(*it) == 1 || std::get<1>(*it) == 1) && (std::get<0>(*it) != 0 && std::get<1>(*it) != 0))
        { //We only want to store the edges on the spine and the shoulders.

          //Store the two edges of the line.
          oneHumanEdges.push_back(edge1);
          oneHumanEdges.push_back(edge2);
        }
      }
      //Store the oneHuman line edges in the vector. Edges will be used on color feature extraction (ColorHistogram).
      humanEdges.push_back(oneHumanEdges);
    }

    VisualFeatures();

    //Create a message of type skeletons to be published.
    image_processing_by_pose::Skeletons humansVect;

    // Draw a rectangle on every discovered skeleton.
    std::vector<int>::iterator itIds = tracking_ids.begin();
    for (std::vector<openpose_ros_msgs::OpenPoseHuman>::iterator itPersona = humans.begin(); itPersona != humans.end(); ++itPersona) //Loop for every skeleton (human).
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
      //-std::vector<cv::Point> oneHumanEdges;
      //Connect the dots to form a skeleton.
      for (std::vector<std::tuple<int, int>>::iterator it = tl.begin(); it != tl.end(); ++it) //Iterate for all points on a single detected skeleton (human).
      {
        //Do not draw lines with zero probability.
        if (itPersona->body_key_points_with_prob[std::get<1>(*it)].prob == 0.0 || itPersona->body_key_points_with_prob[std::get<0>(*it)].prob == 0.0)
          continue;
        cv::Point edge1 = cv::Point(itPersona->body_key_points_with_prob[std::get<0>(*it)].x, itPersona->body_key_points_with_prob[std::get<0>(*it)].y);
        cv::Point edge2 = cv::Point(itPersona->body_key_points_with_prob[std::get<1>(*it)].x, itPersona->body_key_points_with_prob[std::get<1>(*it)].y);
        //Store the points of the line in a vector to publish them later.
        image_processing_by_pose::SkeletonLines bodyPart;

        if ((std::get<0>(*it) == 1 || std::get<1>(*it) == 1) && (std::get<0>(*it) != 0 && std::get<1>(*it) != 0))
        { //We only publish the spine and the sholders.
          cv::LineIterator lineIt(cv_ptr->image, edge1, edge2, 8);

          for (int i = 0; i < lineIt.count; i++, ++lineIt) //Store the image points of the line.
          {

            image_processing_by_pose::PointXYRGB imagePoint;

            cv::Vec3b color = *(const cv::Vec3b *)*lineIt;
            //Maybe blue and red have to be swaped (not sure): imagePoint.r = color.val[0];  imagePoint.b = color.val[2];
            imagePoint.x = lineIt.pos().x;
            imagePoint.y = lineIt.pos().y;
            imagePoint.b = color.val[0];
            imagePoint.g = color.val[1];
            imagePoint.r = color.val[2];

            bodyPart.linePoints.push_back(imagePoint); //Note: we may want to push more points.

            //Mark the points between edge1 and edge2.
            //cv::Point pt= lineIt.pos();
            //cv::circle(cv_ptr->image, pt, 2, CV_RGB(color_of_ids[*itIds].r, color_of_ids[*itIds].g, color_of_ids[*itIds].b));
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
