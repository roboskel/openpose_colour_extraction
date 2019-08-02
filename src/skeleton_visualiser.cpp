#include <iostream>
#include <vector>
#include <tuple>
#include <bits/stdc++.h>

#include <string>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include "std_msgs/Float32.h"
#include "std_msgs/Int16.h"

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


//Default parameters. If a parametrers.yamal file exists, those parameters will be overwriten.
int NUM_OF_BINS = 80; //Max: 256 You may twick this value!
double THRESHOLD_SKELETON_POSIBILITY = 0.45; //Over this possibility the points will be detected.
double LOWEST_POSIBILITY_MATCHING = 0.1; //Over this possibility we match an id of the previous frame to an id of the new frame.
double LOWEST_POSIBILITY_REMATCHING = 0.1; //Over this possibility we match an old dissapeared id to a newly apeared.
int OLD_HISTOGRAM_MEMORY_SIZE = 10;    //Size of the buffer of dissapeared histograms. If this is too big the algorithm will run slow.
int LOOP_RATE = 60;  //The rate at withch the node recieves a new image. Publishing rates are also affected.
int LINE_WIDTH = 4;  //If > 1 then in the process of mask creation, adjacent pixels to the line are included.
int COMPARE_METHOD = 0;    // 0 -> Correlation
                            // 1 -> Chi-Square
                            // 2 -> Intersection
                            // 3 -> Bhattacharyya distance
                            //Carefull! For the Chi-Square and Bhattacharyya distance methods, the lower the metric, the more accurate the match.
                            //Consequently, if we use 1 or 3 methods we need to amend the code and alter the inequalities in the 
                            //Reidentification method so as to find the minimum element.
                            //If the comparing method changes, the possibility thresholds (apart from THRESHOLD_SKELETON_POSIBILITY) should change too.
int FEATURE_EXTRACTOR_METHOD = 2; // 0 -> simple 1 chanell color histogram. Usefull if we want to plot the histogram of an image.
                                  // 1 -> 3D color histogram
                                  // 2 -> 2d HS histogram
int AND_SPINE = 1; // 1-> feature extraction from shoulder and spine points.
                    // 0-> feature extraction from shoulders only.
int USE_SPACIAL_LOCALITY_TOO =0; // 1-> use boath visual features and spatial proximity for tracking.
                                // 0-> use only visual features.
int COMPARE_EVERYTHING =0; //A cornercase. When dissapeared ids are not empty but old histogram is empty and new is not empty, check if the new person(s) match
                          // to a dissapeared histogram and if they do, do not create a new id (to do so, set COMPARE_EVERYTHING to 1). Otherwise, if old histogram is empty and new is not then new ids will
                          //be created for sure. Seems to be better when set to 0.
std::string CAMERA_TOPIC = std::string("/camera/rgb/image_raw");
std::string OPENPOSE_ROS_TOPIC = std::string("/openpose_ros/human_list");
std::string OUTPUT_VIDEO_TOPIC = std::string("/image_converter/output_video");
std::string OUTPUT_HISTOGRAM_TOPIC = std::string("/Histogram");
std::string OUTPUT_SKELETON_POINTS = std::string("/skeleton_points");
std::string IDS = std::string("/id_array");



//Parameter System.
int num_of_bins;
double threshold_skeleton_posibility;
double lowest_posibility_matching;
double lowest_posibility_rematching;
int old_histogram_memory_size;
int loop_rate;
int line_width;
int compare_method;
int feature_extractor_method;
int and_spine;
int use_spacial_locality_too;
int compare_everything;
std::string camera_topic;
std::string openpose_ros_topic;
std::string output_video_topic;
std::string output_histogram_topic;
std::string output_skeleton_points;
std::string ids;


/*
This node attempts to visualise the OpenPose data streamed by a bag without the need of an openpose instalation.
 */

//typedef image_processing_by_pose my_msgs;

static const std::string OPENCV_WINDOW = "Skeleton Tracker Video";
static const std::string OPENCV_HISTOGRAM = "Histogram";
//static const std::string OPENCV_HISTOGRAM_COMPARISON_1 = "Old Histogram";
//static const std::string OPENCV_HISTOGRAM_COMPARISON_2 = "New Histogram";
static const std::string OPENCV_WINDOW_COMPARISON_1 = "Old Image";
static const std::string OPENCV_WINDOW_COMPARISON_2 = "New Image";

std::vector<std::tuple<int, int>> tl;

struct Color
{
  int r, g, b, a;
};

// int compare (const void * a, const void * b)
// {
//   return ( *(int*)a - *(int*)b );
// }

class ImageConverter
{
  //Public atributes
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  image_transport::Publisher histogram_pub_;
  ros::Publisher skeleton_points_pub_;
  ros::Publisher ids_pub_;

  std::vector<std::vector<cv::Point>> humanEdges; //Store the skeleton points that we want to consider for the mask of the histogram
  ros::Subscriber human_list_;
  std::vector<openpose_ros_msgs::OpenPoseHuman> humans;

  std::vector<Color> color_of_ids = {{66, 245, 135}, {245, 66, 81}, {48, 65, 194}, {136, 77, 161}, {82, 220, 227}, {227, 176, 82}, {176, 105, 55}, {173, 181, 109}, {64, 133, 88}, {10, 200, 80}};
  cv_bridge::CvImagePtr cv_ptr;
  cv_bridge::CvImagePtr cv_ptr_oldImage;
  cv_bridge::CvImage hist_msg;
  std::vector<cv::Mat> *oldHist = new std::vector<cv::Mat>;         //Store the previous histogram. Compare the old one with the new one and match the ids.
  std::vector<cv::Mat> *newHist = new std::vector<cv::Mat>;         //Store the new histogram. We will be comparing those.
  std::vector<cv::Mat> *dissapearedHist = new std::vector<cv::Mat>; //Stores the dissapeared histograms
  std::vector<int> dissapearedIds;                                  //Stores the dissapeared ids.
  std::vector<cv::Point> *oldHumanPlaces = new std::vector<cv::Point>;
  std::vector<cv::Point> *newHumanPlaces = new std::vector<cv::Point>;

  

public:
  ImageConverter()
      : it_(nh_)
  {

    // Subscrive to input video feed & to OpenPoseHumanList and publish output video feed
    image_sub_ = it_.subscribe(camera_topic, 1, &ImageConverter::imageCb, this);
    human_list_ = nh_.subscribe(openpose_ros_topic, 1, &ImageConverter::openposeCB, this);
    image_pub_ = it_.advertise(output_video_topic, 1);
    histogram_pub_ = it_.advertise(output_histogram_topic, 1);
    skeleton_points_pub_ = nh_.advertise<image_processing_by_pose::Skeletons>(output_skeleton_points, 1);
    ids_pub_ = nh_.advertise<std_msgs::Int32MultiArray>(ids, 1);

    //cv::namedWindow(OPENCV_WINDOW);
    //cv::namedWindow(OPENCV_HISTOGRAM, CV_WINDOW_AUTOSIZE);
    //cv::namedWindow(OPENCV_HISTOGRAM_COMPARISON_1);
    //cv::namedWindow(OPENCV_HISTOGRAM_COMPARISON_2);
    cv::namedWindow(OPENCV_WINDOW_COMPARISON_1);
    cv::namedWindow(OPENCV_WINDOW_COMPARISON_2);

    //tracking_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};

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
    //cv::destroyWindow(OPENCV_WINDOW);
    //cv::destroyWindow(OPENCV_HISTOGRAM);
    //cv::destroyWindow(OPENCV_HISTOGRAM_COMPARISON_1);
    //cv::destroyWindow(OPENCV_HISTOGRAM_COMPARISON_2);
    cv::destroyWindow(OPENCV_WINDOW_COMPARISON_1);
    cv::destroyWindow(OPENCV_WINDOW_COMPARISON_2);
  }

  void openposeCB(const openpose_ros_msgs::OpenPoseHumanList::ConstPtr &msg)
  {
    //First clear the vector for the new data to take place.
    humans.clear();
    //ROS_INFO("Number of recognised people: [%d]", msg->num_humans);
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

      cv::line(mask, edge1, edge2, CV_RGB(255, 255, 255), line_width); //You may twick the thickness of the line (last parameter). The more thick, the more points on the image.
    }
    return mask;
  }

  cv::Mat HSHistogramAndDraw(const cv::Mat &mask)
  {
    cv::Mat hsvImage;
    cv::cvtColor(cv_ptr->image, hsvImage, CV_BGR2HSV);

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = {0, 180};
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = {0, 256};
    const float *ranges[] = {hranges, sranges};
    cv::MatND histHS;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    cv::calcHist(&hsvImage, 1, channels, mask, // do not use mask
                 histHS, 2, histSize, ranges,
                 true, // the histogram is uniform
                 false);

    //normalize( histHS, histHS, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() ); //???

    double maxVal = 0;
    cv::minMaxLoc(histHS, 0, &maxVal, 0, 0);

    int scale = 10;
    cv::Mat histImg = cv::Mat::zeros(sbins * scale, hbins * 10, CV_8UC3);

    for (int h = 0; h < hbins; h++)
    {
      for (int s = 0; s < sbins; s++)
      {
        float binVal = histHS.at<float>(h, s);
        int intensity = std::round(binVal * 255 / maxVal);
        cv::rectangle(histImg, cv::Point(h * scale, s * scale),
                      cv::Point((h + 1) * scale - 1, (s + 1) * scale - 1),
                      cv::Scalar::all(intensity),
                      CV_FILLED);
      }
    }

    cv::imshow(OPENCV_HISTOGRAM, histImg);
    cv::waitKey(4);

    return histHS;
  }

  cv::Mat ThreeDimensionalColorHistogram(const cv::Mat &mask)
  {
    int imgCount = 1;
    int dims = 3;
    const int sizes[] = {num_of_bins, num_of_bins, num_of_bins};
    const int channels[] = {0, 1, 2};
    float rRange[] = {0, 256};
    float gRange[] = {0, 256};
    float bRange[] = {0, 256};
    const float *ranges[] = {bRange, gRange, rRange};
    cv::Mat hist3D;
    cv::calcHist(&cv_ptr->image, imgCount, channels, mask, hist3D, dims, sizes, ranges);

    return hist3D;
  }

  std::vector<cv::Mat> ColorHistogram(const cv::Mat &mask)
  {
    //Calculate the color histogram for the given mask on the frame (using 'global' variable cv_ptr->image).
    //Returns an array of histograms B, G, R.

    // Separate the image in 3 places ( B, G and R ).
    std::vector<cv::Mat> bgr_planes;
    cv::split(cv_ptr->image, bgr_planes);

    // Establish the number of bins.
    int histSize = num_of_bins;

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

    std::vector<cv::Mat> histograms = {b_hist, g_hist, r_hist};
    return histograms;
  }

  cv::Mat DrawHistogram3chanels(const cv::Mat &b_hist, const cv::Mat &g_hist, const cv::Mat &r_hist, int histSize = 256, int hist_w = 512, int hist_h = 400)
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
  cv::Scalar MeanValue(const cv::Mat &mask)
  {
    return cv::mean(cv_ptr->image, mask);
  }

  void VisualFeatures()
  {
    //Extract visual features from the skeleton.
    oldHist->clear();
    //A new histogram will be created by the new frame. Update the old one.
    *oldHist = *newHist;
    newHist->clear();

    //For every Skeleton in the frame we find a mask and pass it to functions like 3DHistogram etc. .  Then, we publish the desired histogram.
    for (std::vector<cv::Point> oneHuman : humanEdges)
    {
      cv::Mat h_b, h_g, h_r;
      const cv::Mat &mask = MaskCalculation(oneHuman);
      
      if(feature_extractor_method == 0){
        const std::vector<cv::Mat>& histVect  = ColorHistogram(mask);  const cv::Mat& hist = histVect[0]; //Other options: ... = histVect[1] or ... = histVect[2]
        newHist->push_back(hist);
        //Want to draw a histogram live? Uncomment the first line. Only if you use method 0.
        //DrawHistogram3chanels(histVect[0], histVect[1], histVect[2], num_of_bins);
        //DrawHistogram1chanel(hist[0], num_of_bins);
      }
      else if(feature_extractor_method == 1){
        const cv::Mat& hist = ThreeDimensionalColorHistogram(mask);
        newHist->push_back(hist);
      }
      else if(feature_extractor_method == 2){
        const cv::Mat &hist = HSHistogramAndDraw(mask);
        newHist->push_back(hist);
      }
      else{
        std::cout << "\nINVALID ARGUMENT!\n";
      }

      //Publish the desired histogram.
      // hist_msg.header   = //not set yet
      // hist_msg.encoding = //not set yet
      //hist_msg.image = hist;
      //histogram_pub_.publish(hist_msg.toImageMsg());

      //Update the 'global' variables.
      //newHist->push_back(hist);
    }
    return;
  }

  std::vector<std::vector<double>> FramesFeatureComparison(std::vector<cv::Mat> *oldHist, std::vector<cv::Mat> *newHist, std::vector<cv::Point> *oldHumanPlaces = new std::vector<cv::Point>, std::vector<cv::Point> *newHumanPlaces = new std::vector<cv::Point>)
  {
    //Always use it after VisualFeatures().

    std::vector<std::vector<double>> Scores(oldHist->size(), std::vector<double>(newHist->size()));
    //In the begining the oldHist is empty. If that is the case, return.
    //Generally, if either the old or the new histogram is empty we can not make the comparison. Consequently, we return an empty score.
    if (oldHist->empty() || newHist->empty())
      return Scores;
    printf("\n\nComparison of people's Histograms between two consecutive frames. \n\n");
    
    int comp_meth = compare_method;

    //Want to print all methods? Uncomment the next line.
    //for (int comp_meth = 0; comp_meth < 4; comp_meth++)
    {
      
      int rowCounter = 0;
      printf("\n\nTable of Scores after Histogram Comparison method %d\n", comp_meth);
      printf("old\tnew1\tnew2\tnew3\tnew4\tnew5\n");
      for (cv::Mat OldPersonHist : *oldHist)
      {
        printf("%d", rowCounter);

        int colCounter = 0;
        //std::vector<double> oneRaw;
        //Scores[rowCounter] = new double[newHist->size()];
        for (cv::Mat NewPersonHist : *newHist)
        {

          double Score = cv::compareHist(OldPersonHist, NewPersonHist, comp_meth);
          printf("\t%1.3f", Score);
          if (comp_meth == compare_method) 
            Scores[rowCounter][colCounter] = Score;
          //oneRaw.push_back(Score);
          colCounter++;
        }
        //if(compare_method==comp_meth)
        //Scores.push_back(oneRaw)
        printf("\n");
        rowCounter++;
      }
    }
    printf("\n\n");

    //Use locality.
    if(!oldHumanPlaces->empty() && !newHumanPlaces->empty()){
      if(oldHumanPlaces->size() != oldHist->size() || newHumanPlaces->size() != newHist->size()) 
        std::cout << "\nProblem: number of human histograms dose not equal the number of human points.\n";
      int rowCounter = 0;
      for(cv::Point oldPoint : *oldHumanPlaces){
        int colCounter = 0;
        for(cv::Point newPoint : *newHumanPlaces){
          double distance = cv::norm(cv::Mat(oldPoint),cv::Mat(newPoint));
          Scores[rowCounter][colCounter] += 1/(distance+1); //You may need to change this if the comparison method changes.
          printf("\t%1.3f", Scores[rowCounter][colCounter] );
          colCounter++;
        }
        printf("\n");
        rowCounter++;
      }
    }

    return Scores;
  }

  std::vector<int> &ReIdentification(std::vector<std::vector<double>> Scores)
  {
    //Create the array to be published.
    std_msgs::Int32MultiArray arrayOfIds;
    arrayOfIds.data.clear(); //Probably not required.

    static int newId =0;
    //This method recives the matching scores and pairs the human ids in two consecutive frames.
    static std::vector<int> tracking_ids;
    if (newHist->empty())
    {
      std::cout << "\n\nNEW HIST IS EMPTY\n\n";
      tracking_ids.clear();
      return tracking_ids;
    }
    if (oldHist->empty() &&(!compare_everything || dissapearedIds.empty()) )// We must add to the condition: && dissapearedIds.empty() . Generally, adding this seems more corect but dose not produce good results.
    {
       //First time of calling OR format ids. Start from the beginning. Re-initialise id counter.
      std::cout << "\n\nNEW HIST IS NOT EMPTY - OLD HIS IS EMPTY \n\n";
      //tracking_ids.reserve(newHist->size());
      //Probably add something to compare the new ids to the dissapeared ones.
      int i;
      for (i = newId; i < newHist->size(); i++)
      {
        tracking_ids.push_back(i);
      }
      newId = i; 
      return tracking_ids;
    }

    
    std::vector<std::tuple<int, int>> maxPos; //Stores the positions of the maximum elements, in decending order
    //(from the position of the globally greater element -> the position of the globally minimum element.)

    //The Scores matrix - vecotr may not be square. In this case, either a new person was detected or an old person disapears.
    //int rows = Scores.size();
    int rows = oldHist->size();
    //int columns = Scores[0].size();
    int columns = newHist->size();

    //Create an  new vector of size columns. This will store the new ids.
    std::vector<int> newIds(columns, -1);

    //if(!oldHist->empty())
    
    std::cout << "\n\nBOATH NEW AND OLD HIST ARE NOT EMPTY\n\n";
    //if rows > columns -> A person disapeared.
    //if rows < columns -> A person apeared.
    int iterations = (rows < columns) ? rows : columns;
    int minsize = iterations;
    //First we find the position of the maximum element in each row of the 2D vector.
    while (iterations--)
    {
      std::cout << "\n\nGOT IN WHILE\n\n";
      double max = -10.0;
      int maxPosRaw = 0;
      int maxPosCol = 0;

      for (unsigned int row = 0; row < rows; ++row)
      {
        for (unsigned int column = 0; column < columns; ++column)
        {
          if (max <= Scores[row][column])
          {
            max = Scores[row][column];
            maxPosRaw = row;
            maxPosCol = column;
          }
        }
      }
      std::cout << "\n\nFound Max value: " << max << "  at position: " << maxPosRaw << ", " << maxPosCol << "\n\n";

      //Delete the raw we took into acount. We need this because otherwise there is the posibility of mathcing the old id to two different
      // new ids. We want the mapping to be 1-1. Similarly, we have to erase the posibility of mapping two rows to one column (two old ids
      // into one new id).
      //Scores[maxPosRaw].clear();
      if (max < lowest_posibility_matching)
        break; //If the possibility is too low, stop matching ids. Twick this value!
      for (int j = 0; j < columns; j++)
      {
        Scores[maxPosRaw][j] = -1.0;
      }
      for (int i = 0; i < rows; i++)
      {
        //if (Scores[i].size() -1 < maxPosCol) continue;
        Scores[i][maxPosCol] = -1.0;
      }
      maxPos.push_back(std::make_tuple(maxPosRaw, maxPosCol)); //Store the pos into a vector of touples.
    }

    std::cout << "\n\n Matching positions Printing \n";
    for (std::vector<std::tuple<int, int>>::iterator i = maxPos.begin(); i != maxPos.end(); ++i)
      std::cout << " Position: " << std::get<0>(*i) << " -> " << std::get<1>(*i) << "  ID: " << tracking_ids[std::get<0>(*i)] << "\n";


    for (std::vector<std::tuple<int, int>>::iterator itMatchIds = maxPos.begin(); itMatchIds != maxPos.end(); ++itMatchIds)
    {
      int oldPositionOfId = std::get<0>(*itMatchIds);
      int newPositionOfId = std::get<1>(*itMatchIds);
      //Copy the id to the new position.
      //Maybe we shouldn't do the match when the posibility is less than a sertain threshold.
      newIds[newPositionOfId] = tracking_ids[oldPositionOfId];
    }

    //Find the new ids.
    int order = 0;
    for (std::vector<int>::iterator oldId = tracking_ids.begin(); oldId != tracking_ids.end(); ++oldId)
    {
      bool flag = true;
      for (std::vector<int>::iterator newId = newIds.begin(); newId != newIds.end(); ++newId)
      {
        if (*newId == *oldId)
        {
          flag = false;
          break;
        }
      }
      if (flag == true)
      {
        std::cout << "\nThis ID disapeared: " << *oldId << "\n";
        //Store the disapeared histogram and it's id.
        dissapearedHist->push_back((*oldHist)[order]);
        dissapearedIds.push_back(*oldId);
      }
      order++;
    }
    

    while (dissapearedIds.size() > old_histogram_memory_size)
    {
      int advancement = dissapearedIds.size() - old_histogram_memory_size;
      std::vector<int>::iterator itId1, itId2;
      itId1 = itId2 = dissapearedIds.begin();
      advance(itId2, advancement);
      std::vector<cv::Mat>::iterator itHist1, itHist2;
      itHist1 = itHist2 = dissapearedHist->begin();
      advance(itHist2, advancement);

      dissapearedIds.erase(itId1, itId2);
      dissapearedHist->erase(itHist1, itHist2);
    }

    for (int j = 0; j < columns; j++)
    {
      if (newIds[j] == -1)
      {
        //Check if this new histogram is similar to a disapeared one.
        //If it is, do not create a new id.

        std::vector<cv::Mat> *PersonsHist = new std::vector<cv::Mat>;
        PersonsHist->push_back((*newHist)[j]);
        //std::cout << "\nFound a new histogram! Position: " << j << "\n";
        if(!dissapearedHist->empty()){
          std::vector<std::vector<double>> ComparisonScores = FramesFeatureComparison(dissapearedHist,PersonsHist);
          int maxRow=0;
          float maxim = -1.0;
          for(int i=0; i<ComparisonScores.size(); i++){
            if(maxim<= ComparisonScores[i][0]){
              maxim = ComparisonScores[i][0];
              maxRow = i;
            }
          }
          if(maxim >= lowest_posibility_rematching){//Twick this value!
            std::cout <<"\nRe found a dissapeared person!\nID: " << dissapearedIds[maxRow] << "\n";
            newIds[j]=dissapearedIds[maxRow];
            //Delete the histogram and the id in case of a hit.
            dissapearedIds.erase(dissapearedIds.begin()+maxRow);
            dissapearedHist->erase(dissapearedHist->begin()+maxRow);
          }
          else{
          newIds[j]=newId++;
          std::cout << "\n\n NEW MAN!  ID= " << newId <<"\n\n";
          }
        }
        else{
        newIds[j]=newId++;
        std::cout << "\n\n NEW MAN!  ID= " << newId <<"\n\n";
        }
      }
    }

    //Are there new apeared people ? If yes, push_back as many new ids as needed (usually, it equals to the difference columns - rows (>0) ).
    //TODO: Όταν κάποιος εξαφανίζεται, δες ποιος ήταν και αποθήκευσέ τον (Θα είναι αυτός που δεν ταίριαξε με κανέναν. Επίσης, αποθήκευσε το id του. Επίσης,
    //αποθήκευσε ενα καλό ιστόγραμμα, όχι το τελευταίο κατά προτίμηση αλλά ενα με μεγάλη πιθανότητα σημείων σκελετού.)
    //Όταν εμφανιστεί ξαφνικά ένας άνθρωπος, σύγκρινε το νέο ιστόγραμμα με κάποιο από τα "χαμένα". Αν γίνει match τότε οκ, μην φτιάξεις καινούριο id.
    //Μια ιδέα είναι η αποθήκευση των ιστογραμμάτων όταν η πιθανότητα των σημείων του σκελετού είναι αυξημένη.

    //if(columns - rows > 0){
    //int numOfNewIds = columns - rows;
    //for(int i=0; i<numOfNewIds; i++){
    //We want to store the "new" histograms so as to compare them with the dissapeared ones later.
    

    /*
    

    std::vector<cv::Mat> *newPersonsHist = new std::vector<cv::Mat>;
    std::vector<int> newPersonHistToNewIdsMap;
    for (int j = 0; j < columns; j++)
    {
      if (newIds[j] == -1)
      {
        //Check if this new histogram is similar to a disapeared one.
        //If it is, do not create a new id.

        //Store the old, dissapeared histogram.
        newPersonsHist->push_back((*newHist)[j]);
        newPersonHistToNewIdsMap.push_back(j);
        std::cout << "\nFound a new histogram! Position: " << j << "\n";
        //std::vector<std::vector<double>> ComparisonScores = FramesFeatureComparison(dissapearedHist,personHist);

        //newIds[j]=newId++;
        //std::cout << "\n\n NEW MAN!  ID= " << newId <<"\n\n";
        //break;
      }
    }
    if (!newPersonsHist->empty() && !dissapearedHist->empty())
    {
      std::cout << "\nWe match dissapeared people to new.\n";
      std::vector<std::vector<double>> ComparisonScores = FramesFeatureComparison(dissapearedHist, newPersonsHist);

      int rows = ComparisonScores.size();
      int columns = ComparisonScores[0].size();
      int iterations = (rows < columns) ? rows : columns;
      std::vector<std::tuple<int, int>> oldNewMatch;
       
      while (iterations--)
      {
        double max = -10.0;
        int maxPosRaw = 0;
        int maxPosCol = 0;

        for (unsigned int row = 0; row < rows; ++row)
        {
          for (unsigned int column = 0; column < columns; ++column)
          {
            if (max <= ComparisonScores[row][column])
            {
              max = ComparisonScores[row][column];
              maxPosRaw = row;
              maxPosCol = column;
            }
          }
        }

        if (max > 0.4)
        { //Twick this value!
          oldNewMatch.push_back(std::make_tuple(maxPosRaw, maxPosCol));
          std::cout << "\n matched man!" << maxPosRaw <<" -> " << maxPosCol << "  maxValue: " << max << "\n" ;
        }
        for (int j = 0; j < columns; j++)
        {
          ComparisonScores[maxPosRaw][j] = -1.0;
        }
        for (int i = 0; i < rows; i++)
        {
          //if (Scores[i].size() -1 < maxPosCol) continue;
          ComparisonScores[i][maxPosCol] = -1.0;
        }
      }

      std::cout << "\n\n oldNewMatch pairs \n";
      for (std::vector<std::tuple<int, int>>::iterator i = oldNewMatch.begin(); i != oldNewMatch.end(); ++i)
      std::cout << " Position: " << std::get<0>(*i) << " -> " << std::get<1>(*i) << "  ID: " << dissapearedIds[std::get<0>(*i)] << "\n";


      std::vector<int> eraser;
      for (std::vector<std::tuple<int, int>>::iterator itMatchIds = oldNewMatch.begin(); itMatchIds != oldNewMatch.end(); ++itMatchIds)
      {
        int oldPositionOfId = std::get<0>(*itMatchIds);
        int newPositionOfId = std::get<1>(*itMatchIds);

        eraser.push_back(oldPositionOfId);

        newIds[newPersonHistToNewIdsMap[newPositionOfId]] = dissapearedIds[oldPositionOfId];
        
        cout <<"\nGot in the first time\n";
        //dissapearedHist->erase(dissapearedHist->begin()+oldPositionOfId);
        //dissapearedIds.erase(dissapearedIds.begin()+oldPositionOfId);
      }
      sort (eraser.begin(), eraser.end());
      //qsort (eraser, eraser.size(), sizeof(int), compare);
      //Erase from the memory vectors the once dissapeared ids that reapeared.
      int offset = 0;
      for(auto i: eraser){
        cout << "\n Eraser!!"<< "Delete: " << i - offset <<" \n";
        std::vector<int>::iterator itId;
        itId = dissapearedIds.begin();
        advance(itId, i-offset);
        std::vector<cv::Mat>::iterator itHist;
        itHist = dissapearedHist->begin();
        advance(itHist, i-offset);

        dissapearedIds.erase(itId);
        dissapearedHist->erase(itHist);
        offset++;
      }
    }
    cout << "\n Passed the eraser succesfully!! \n";
    //}
    //}
    //Are there any "-1" left? Then, create new ids.
    for (int j = 0; j < columns; j++)
    {
      if (newIds[j] == -1)
      {
        newIds[j] = newId++;
        std::cout << "\n\n NEW MAN!  ID= " << newId << "\n\n";
      }
    }

    */
    tracking_ids.clear();
    tracking_ids = newIds;
    for( auto l : tracking_ids){
      arrayOfIds.data.push_back(l);
    }
    //We publish the new array of ids.
    ids_pub_.publish(arrayOfIds);
    // for (std::vector<int>::iterator i = newIds.begin(); i != newIds.end(); ++i){
    //   tracking_ids.push_back(*i);
    // }

    return tracking_ids;
  }

  void imageCb(const sensor_msgs::ImageConstPtr &msg)
  {
    //ROS_INFO("Just got a new image!");
    if (cv_ptr != NULL)
      cv_ptr_oldImage = cv_ptr;
    //cv_ptr->image.copyTo(oldImage); //Store the old image for comparison.

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
    humanEdges.clear();

    //We keep the coordinates of main human points.
    oldHumanPlaces->clear();
    *oldHumanPlaces = *newHumanPlaces;
    newHumanPlaces->clear();

    std::vector<int> pickedHumans; //Used to know for whitch humans we drew the histogram.
    int k=-1;
    for (std::vector<openpose_ros_msgs::OpenPoseHuman>::iterator itPersona = humans.begin(); itPersona != humans.end(); ++itPersona) //Loop for every skeleton (human).
    {
      k++;
      std::vector<cv::Point> oneHumanEdges;
      bool flag = false;
      for (std::vector<std::tuple<int, int>>::iterator it = tl.begin(); it != tl.end(); ++it) //Iterate for all points on a single detected skeleton (human).
      {

        //Ignore points with low probability.
        if (itPersona->body_key_points_with_prob[std::get<1>(*it)].prob < threshold_skeleton_posibility || itPersona->body_key_points_with_prob[std::get<0>(*it)].prob < threshold_skeleton_posibility) //Twick these values!
          continue;
        cv::Point edge1 = cv::Point(itPersona->body_key_points_with_prob[std::get<0>(*it)].x, itPersona->body_key_points_with_prob[std::get<0>(*it)].y);
        cv::Point edge2 = cv::Point(itPersona->body_key_points_with_prob[std::get<1>(*it)].x, itPersona->body_key_points_with_prob[std::get<1>(*it)].y);

        if ((std::get<0>(*it) == 1 || std::get<1>(*it) == 1) && (std::get<0>(*it) != 0 && std::get<1>(*it) != 0) && ((std::get<0>(*it) != 8 && std::get<1>(*it) != 8)|| and_spine))
        { //We only want to store the edges on the spine and the shoulders.

          //Store the two edges of the line.
          oneHumanEdges.push_back(edge1);
          oneHumanEdges.push_back(edge2);

          //Store the main point of each man
          if(std::get<0>(*it) == 1 && !flag ) {newHumanPlaces->push_back(edge1); flag = true;}
          else if(std::get<1>(*it) == 1 && !flag ) {newHumanPlaces->push_back(edge2); flag = true;}

        }
        
      }


      //Store the oneHuman line edges in the vector. Edges will be used on color feature extraction (ColorHistogram).
      if(oneHumanEdges.empty()) continue; 
      humanEdges.push_back(oneHumanEdges);
      pickedHumans.push_back(k);
      
    }

    VisualFeatures();
    std::vector<std::vector<double>> Scores;
    if(use_spacial_locality_too ==0)
      Scores = FramesFeatureComparison(oldHist, newHist);
    else
      Scores = FramesFeatureComparison(oldHist, newHist, oldHumanPlaces, newHumanPlaces);

    std::cout <<"\nThis is the special locality param: " <<use_spacial_locality_too <<"  \n";
    std::vector<int> &tracking_ids = ReIdentification(Scores);

    std::cout << "\n\n Tracking ids Vector Printing \n";
    for (std::vector<int>::iterator i = tracking_ids.begin(); i != tracking_ids.end(); ++i)
      std::cout << *i << ' ';

    std::cout << "\n\n";

    //Create a message of type skeletons to be published.
    image_processing_by_pose::Skeletons humansVect;
    k=-1;
    int j=0;
    // Draw a rectangle on every discovered skeleton.
    std::vector<int>::iterator itIds = tracking_ids.begin();
    for (std::vector<openpose_ros_msgs::OpenPoseHuman>::iterator itPersona = humans.begin(); itPersona != humans.end(); ++itPersona) //Loop for every skeleton (human).
    {
      k++;
      if(pickedHumans.size()<j+1) break; //If none human is picked do not draw anything.

      if(pickedHumans[j] != k) continue; //This means that the posibility filter ruled out this person. We didnt draw a histogram and
                                          // we didnt tracked it.
      j++;
      //Carefull! Bounding box (x,y) point is on the top left corner.
      //cv::rectangle(image, cornerPoint1, cornerPoint2, Color [, ...])
      openpose_ros_msgs::BoundingBox tempBox = itPersona->body_bounding_box;
      cv::rectangle(cv_ptr->image, cv::Point(tempBox.x, tempBox.y), cv::Point(tempBox.x + tempBox.width, tempBox.y + tempBox.height),
                    CV_RGB(color_of_ids[(*itIds) % 10].r, color_of_ids[(*itIds) % 10].g, color_of_ids[(*itIds) % 10].b));

      //Draw the dots.
      for (int i = 0; i < (itPersona->body_key_points_with_prob).size(); ++i)
      {
        if (itPersona->body_key_points_with_prob[i].prob < threshold_skeleton_posibility)
          continue; //Twick this value.
        cv::circle(cv_ptr->image, cv::Point(itPersona->body_key_points_with_prob[i].x, itPersona->body_key_points_with_prob[i].y), 2,
                   CV_RGB(color_of_ids[(*itIds) % 10].r, color_of_ids[(*itIds) % 10].g, color_of_ids[(*itIds) % 10].b));
      }
      //-std::vector<cv::Point> oneHumanEdges;
      //Connect the dots to form a skeleton.
      for (std::vector<std::tuple<int, int>>::iterator it = tl.begin(); it != tl.end(); ++it) //Iterate for all points on a single detected skeleton (human).
      {
        //Do not draw lines with near zero probability.
        if (itPersona->body_key_points_with_prob[std::get<1>(*it)].prob < threshold_skeleton_posibility || itPersona->body_key_points_with_prob[std::get<0>(*it)].prob < threshold_skeleton_posibility)
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
            //cv::circle(cv_ptr->image, pt, 2, CV_RGB(color_of_ids[(*itIds)%10].r, color_of_ids[(*itIds)%10].g, color_of_ids[(*itIds)%10].b));
          }
          //Store the body part in the vector of pcls.
          humansVect.Person.push_back(bodyPart);
        }

        //Drow the line
        cv::line(cv_ptr->image, edge1, edge2, CV_RGB(color_of_ids[(*itIds) % 10].r, color_of_ids[(*itIds) % 10].g, color_of_ids[(*itIds) % 10].b));
      }

      std::string ID("ID: ");
      ID += std::to_string(*itIds);
      //Write the id each person has on the image.
      cv::putText(cv_ptr->image, ID, cv::Point(tempBox.x, tempBox.y), cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(color_of_ids[(*itIds) % 10].r, color_of_ids[(*itIds) % 10].g, color_of_ids[(*itIds) % 10].b));
      //length(tracking_ids) should not be less than length(humans)!
      ++itIds;
    }

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW_COMPARISON_2, cv_ptr->image); //New image
    if (cv_ptr_oldImage != NULL)
      cv::imshow(OPENCV_WINDOW_COMPARISON_1, cv_ptr_oldImage->image); //Old image
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
  ros::NodeHandle n;
  //Load the parameters from *.yamal file, if it exists.
  n.param("image_processing_by_pose/CAMERA_TOPIC", camera_topic, CAMERA_TOPIC);
  n.param("image_processing_by_pose/OPENPOSE_ROS_TOPIC", openpose_ros_topic, OPENPOSE_ROS_TOPIC);
  n.param("image_processing_by_pose/OUTPUT_VIDEO_TOPIC", output_video_topic, OUTPUT_VIDEO_TOPIC);
  n.param("image_processing_by_pose/OUTPUT_HISTOGRAM_TOPIC", output_histogram_topic, OUTPUT_HISTOGRAM_TOPIC);
  n.param("image_processing_by_pose/OUTPUT_SKELETON_POINTS", output_skeleton_points, OUTPUT_SKELETON_POINTS);
  n.param("image_processing_by_pose/IDS", ids, IDS);
  n.param("image_processing_by_pose/NUM_OF_BINS", num_of_bins, NUM_OF_BINS);
  n.param("image_processing_by_pose/THRESHOLD_SKELETON_POSIBILITY", threshold_skeleton_posibility, THRESHOLD_SKELETON_POSIBILITY);
  n.param("image_processing_by_pose/LOWEST_POSIBILITY_MATCHING", lowest_posibility_matching, LOWEST_POSIBILITY_MATCHING);
  n.param("image_processing_by_pose/LOWEST_POSIBILITY_REMATCHING", lowest_posibility_rematching, LOWEST_POSIBILITY_REMATCHING);
  n.param("image_processing_by_pose/OLD_HISTOGRAM_MEMORY_SIZE", old_histogram_memory_size, OLD_HISTOGRAM_MEMORY_SIZE);
  n.param("image_processing_by_pose/LOOP_RATE", loop_rate, LOOP_RATE);
  n.param("image_processing_by_pose/LINE_WIDTH", line_width, LINE_WIDTH);
  n.param("image_processing_by_pose/COMPARE_METHOD", compare_method, COMPARE_METHOD);
  n.param("image_processing_by_pose/FEATURE_EXTRACTOR_METHOD", feature_extractor_method, FEATURE_EXTRACTOR_METHOD);
  n.param("image_processing_by_pose/AND_SPINE", and_spine, AND_SPINE);
  n.param("image_processing_by_pose/USE_SPACIAL_LOCALITY_TOO", use_spacial_locality_too, USE_SPACIAL_LOCALITY_TOO);
   n.param("image_processing_by_pose/COMPARE_EVERYTHING", compare_everything, COMPARE_EVERYTHING);
  ImageConverter ic;
  //ros::spin();
  ros::Rate loop_rate(loop_rate); //Determine the fps.

  while (ros::ok())
  {
    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}
