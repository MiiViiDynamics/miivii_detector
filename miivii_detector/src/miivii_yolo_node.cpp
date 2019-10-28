#include <image_transport/subscriber_filter.h>
#include "image_transport/image_transport.h"
#include "ros/ros.h"

#include <unistd.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"

#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <unordered_map>
#include <vector>


#if APEX
  #if JPVERSION == 420
    #include "MiiViiYoloSDKInterface.h"
  #elif JPVERSION == 422
    #include "MvAlgorithm.h"
  #endif
#else
  #include "MiiViiYoloSDKInterface.h"
#endif

#include "rect_class_score.h"

#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>

#if APEX
  #if JPVERSION == 422
    #define YOLO3C_CLASS_NUM   3
    const char lables_3c[YOLO3C_CLASS_NUM][10] = {
      "person",
      "bicycle",
      "car"
    };
  #endif
#endif

#define MAX_CAMERA_SUPPORT 4
#define DEBUG_TIME 1
#define PUB_RESULT_IMAGE 1

#define USE_DEBUG 1
#if USE_DEBUG
  #define DEBUG_PRINT \
    ROS_INFO("<MIIVII-DEBUG> [%s] %d", __FUNCTION__, __LINE__);
#endif

using namespace cv;
using namespace std;
namespace enc = sensor_msgs::image_encodings;

using message_filters::sync_policies::ExactTime;

double what_time_is_it_now() {
  struct timeval time;
  if (gettimeofday(&time, NULL)) {
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

float colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1},
                      {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
double image_clone = what_time_is_it_now();
double infer;


float get_color(int c, int x, int max) {
  float ratio = ((float)x / max) * 5;
  int i = floor(ratio);
  int j = ceil(ratio);
  ratio -= i;
  float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
  return r * 255;
}

bool file_exists(const string& file) { return access(file.c_str(), 0) == 0; }

class miivii_detector {
  using ExactPolicy2 = ExactTime<sensor_msgs::Image, sensor_msgs::Image>;
  using ExactPolicy3 =
      ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image>;
  using ExactPolicy4 = ExactTime< sensor_msgs::Image, sensor_msgs::Image,
                                  sensor_msgs::Image, sensor_msgs::Image>;
  using ExactSync2 = message_filters::Synchronizer<ExactPolicy2>;
  using ExactSync3 = message_filters::Synchronizer<ExactPolicy3>;
  using ExactSync4 = message_filters::Synchronizer<ExactPolicy4>;

public:
  ros::NodeHandle nh_;
  int camera_count;

  ros::Publisher publisher_objects_[MAX_CAMERA_SUPPORT];
  image_transport::SubscriberFilter image_sub_[MAX_CAMERA_SUPPORT];
  image_transport::Publisher image_pub_[MAX_CAMERA_SUPPORT];
  std::string camera_topic_name_[MAX_CAMERA_SUPPORT];
  std::string detect_results_topics_[MAX_CAMERA_SUPPORT];

  boost::shared_ptr<ExactSync2> exact_sync2_;
  boost::shared_ptr<ExactSync3> exact_sync3_;
  boost::shared_ptr<ExactSync4> exact_sync4_;

public:

  bool publish_result_image;
  bool debug_time;           //是否在终端打印单帧处理耗时信息

  //sdk related
  int classes;
  float thresh;
  float nms;
  int cache;
  int save;
  int int8;
  int dla;
  std::vector<int> anchors;  // anchors sizes
  vector<int> shape;
  vector<char*> output_name;
  std::vector<string> coco_label;
  std::string cachemodel;
  std::string label_file;
  std::string caffemodel;
  std::string proto;
  bool bInitOk, bModelfileOK, bLabelFileOk;
#if APEX
  #if JPVERSION == 420
    MiiViiYolov3SDK* miiviiDetector;
  #elif JPVERSION == 422
    miivii::yolo::MvHumanBikeCarDetect* miiviiDetector;
  #endif
#else
  MiiViiYolov3SDK* miiviiDetector;
#endif
  std::string window_name;
  int queue_size_;

public:
  miivii_detector() : nh_("~") {
    bInitOk = false;
    caffemodel = "";
    proto = "";
    queue_size_ = 4;
    ROS_INFO("starting to miivii detect node construct function");
#if APEX
  #if JPVERSION == 420
      // init the modle, which is a must for JPVERSION 420
      nh_.param("cachemodel", cachemodel,
                std::string("/opt/miivii/models/yolo/yolov3/APEX/"
                            "yolov3_caffemodel_batch4.tensorcache"));
      ROS_INFO("cachemodel:%s", cachemodel.c_str());
      if (file_exists(cachemodel)) {
        bModelfileOK = true;
      } else {
        ROS_ERROR("mode file does not exist");
        bModelfileOK = false;
      }

      nh_.param("label_file", label_file,
                std::string("/opt/miivii/models/yolo/yolov3/yolo.labels"));
      ROS_INFO("label_file:%s", label_file.c_str());
      if (file_exists(label_file)) {
        bLabelFileOk = true;
        std::ifstream labels(label_file);
        string line;
        classes = 0;
        while (std::getline(labels, line)) {
          coco_label.push_back(string(line));
          classes++;
        }
      } else {
        ROS_ERROR("label file does not exist");
        bLabelFileOk = false;
      }
  #elif JPVERSION == 422
      // model will be inited in the SDK
      // get label from lables_3c
      for(int i = 0; i < YOLO3C_CLASS_NUM; i++) {
        coco_label.push_back(string(lables_3c[i]));
      }
      classes = YOLO3C_CLASS_NUM;
  #endif
#else
      nh_.param("cachemodel", cachemodel,
                std::string("/opt/miivii/models/yolo/yolov3/"
                            "yolov3_caffemodel_batch32.tensorcache"));
      ROS_INFO("cachemodel:%s", cachemodel.c_str());
      if (file_exists(cachemodel)) {
        bModelfileOK = true;
      } else {
        ROS_ERROR("mode file does not exist");
        bModelfileOK = false;
      }

      nh_.param("label_file", label_file,
                std::string("/opt/miivii/models/yolo/yolov3/yolo.labels"));
      ROS_INFO("label_file:%s", label_file.c_str());
      if (file_exists(label_file)) {
        bLabelFileOk = true;
        std::ifstream labels(label_file);
        string line;
        classes = 0;
        while (std::getline(labels, line)) {
          coco_label.push_back(string(line));
          classes++;
        }
      } else {
        ROS_ERROR("label file does not exist");
        bLabelFileOk = false;
      }
#endif

    // init camera related param
    nh_.param("camera_count", camera_count, 0);
    if (camera_count < 0 || camera_count > MAX_CAMERA_SUPPORT) {
      ROS_INFO("illegal camera count %d", camera_count);
    }

    nh_.param("camera1_topic", camera_topic_name_[0], std::string("camera1"));
    nh_.param("camera2_topic", camera_topic_name_[1], std::string("camera2"));
    nh_.param("camera3_topic", camera_topic_name_[2], std::string("camera3"));
    nh_.param("camera4_topic", camera_topic_name_[3], std::string("camera4"));

    nh_.param("camera1_detect_results", detect_results_topics_[0],
              std::string("camera1_objects"));
    nh_.param("camera2_detect_results", detect_results_topics_[1],
              std::string("camera2_objects"));
    nh_.param("camera3_detect_results", detect_results_topics_[2],
              std::string("camera3_objects"));
    nh_.param("camera4_detect_results", detect_results_topics_[3],
              std::string("camera4_objects"));

    for (int i = 0; i < camera_count; i++) {
      publisher_objects_[i] = nh_.advertise<autoware_msgs::DetectedObjectArray>(
          detect_results_topics_[i].c_str(), 1);
    }

    // visualize option
    nh_.param<bool>("publish_result_image", publish_result_image, false);

    // debug
    nh_.param<bool>("debug_time", debug_time, false);

    // sdk related params
    double thresh_;
    nh_.param("thresh", thresh_, 0.25);
    thresh = (float)thresh_;
    double nms_;
    nh_.param("nms", nms_, 0.45);
    nms = (float)nms_;
    nh_.param("cache", cache, 1);
    nh_.param("save", save, 0);
    nh_.param("int8", int8, 0);
    nh_.param("dla", dla, 0);

    ROS_INFO("%f %f", thresh, nms);
    anchors = { 10, 13, 16,  30,  33, 23,  30,  61,  62,
                45, 59, 119, 116, 90, 156, 198, 373, 326};
    shape = {416, 416, 3};
    output_name = {(char*)"yolo1", (char*)"yolo2", (char*)"yolo3"};
#if APEX
  #if JPVERSION == 420
      if (bModelfileOK && bLabelFileOk) {
        bInitOk = true;
        MiiViiYolov3SDKConfig config(classes, thresh, nms, cache, int8, dla,
                                    anchors, shape, output_name, camera_count, MAX_CAMERA_SUPPORT);
        miiviiDetector =
            new MiiViiYolov3SDK(cachemodel, proto, caffemodel, config);
      }
  #elif JPVERSION == 422
      miiviiDetector = new miivii::yolo::MvHumanBikeCarDetect(0.5, camera_count);
      bInitOk = true;
  #endif
#else
  if (bModelfileOK && bLabelFileOk) {
        bInitOk = true;
        MiiViiYolov3SDKConfig config(classes, thresh, nms, cache, int8, dla,
                                    anchors, shape, output_name, camera_count, MAX_CAMERA_SUPPORT);
        miiviiDetector =
            new MiiViiYolov3SDK(cachemodel, proto, caffemodel, config);
      }
#endif

    image_transport::ImageTransport it_(nh_);

    if (publish_result_image) {
      for (int i = 0; i < camera_count; i++) {
        image_pub_[i] =
            it_.advertise(std::string("camera") + std::to_string(i) +
                              std::string("_with_rect"),
                          1);
      }
    }

    for (int i = 0; i < camera_count; i++) {
      image_sub_[i].subscribe(it_, camera_topic_name_[i], 1);
    }

    // Please tell me if there is better solutions, thanks!
    switch (camera_count) {
      case 1:
        image_sub_[0].registerCallback(
            boost::bind(&miivii_detector::detectorCallback1, this, _1));
        break;
      case 2:
        exact_sync2_ = boost::make_shared<ExactSync2>(
            ExactPolicy2(queue_size_), image_sub_[0], image_sub_[1]);
        exact_sync2_->registerCallback(
            boost::bind(&miivii_detector::detectorCallback2, this, _1, _2));
        break;
      case 3:
        exact_sync3_ = boost::make_shared<ExactSync3>(
            ExactPolicy3(queue_size_), image_sub_[0], image_sub_[1],
            image_sub_[2]);
        exact_sync3_->registerCallback(
            boost::bind(&miivii_detector::detectorCallback3, this, _1, _2, _3));
        break;
      case 4:
        exact_sync4_ = boost::make_shared<ExactSync4>(
            ExactPolicy4(queue_size_), image_sub_[0], image_sub_[1],
            image_sub_[2], image_sub_[3]);
        exact_sync4_->registerCallback(boost::bind(
            &miivii_detector::detectorCallback4, this, _1, _2, _3, _4));
        break;
      default:
        break;
    }

    ROS_INFO("yolo detect node construct function done!!!");
  }

  ~miivii_detector() {
    delete miiviiDetector;
  }

  void convert_rect_to_image_obj(
      std::vector<RectClassScore<float>>& in_objects,
      autoware_msgs::DetectedObjectArray& out_message) {
    for (unsigned int i = 0; i < in_objects.size(); ++i) {
      autoware_msgs::DetectedObject obj;

      obj.x = in_objects[i].x;
      obj.y = in_objects[i].y;
      obj.width = in_objects[i].w;
      obj.height = in_objects[i].h;
      if (in_objects[i].x < 0) obj.x = 0;
      if (in_objects[i].y < 0) obj.y = 0;
      if (in_objects[i].w < 0) obj.width = 0;
      if (in_objects[i].h < 0) obj.height = 0;

      int offset = in_objects[i].class_type * 123457 % classes;
      obj.color.r = get_color(2, offset, classes);
      obj.color.g = get_color(1, offset, classes);
      obj.color.b = get_color(0, offset, classes);
      obj.color.a = 1.0f;

      obj.score = in_objects[i].score;
      obj.label = in_objects[i].GetClassString();

      out_message.objects.push_back(obj);
    }
  }

  void DrawLabel(const autoware_msgs::DetectedObjectArray &in_detected_object,
                            cv::Mat &image)
  {
    int kRectangleThickness = 2;
    // Draw rectangles for each object
    for(const autoware_msgs::DetectedObject& object : in_detected_object.objects)
    //for (int kk = 0;kk<in_detected_object.size();kk++)
    {
      cv::Point rectangle_origin(object.x, object.y);
      // label's property
      const int font_face = cv::FONT_HERSHEY_DUPLEX;
      const double font_scale = 1.0;//0.7;
      const int font_thickness = 1;
      int font_baseline = 0;
      int icon_width = 40;
      int icon_height = 40;
      std::ostringstream label_one;
      //std::ostringstream label_two;
      cv::Size label_size = cv::getTextSize("0123456789",
                                            font_face,
                                            font_scale,
                                            font_thickness,
                                            &font_baseline);
      cv::Point label_origin = cv::Point(rectangle_origin.x,
                                        rectangle_origin.y - font_baseline - kRectangleThickness*2 - icon_height);
      label_one << object.label<<"  "<<std::to_string(object.score);
      // if (object.objectID > 0)
      // {
      //     label_one << " " << std::to_string(object.objectID);
      // }
      if(label_origin.x < 0)
          label_origin.x = 0;
      if(label_origin.y < 0)
          label_origin.y = 0;
      cv::Rect text_holder_rect;
      text_holder_rect.x = label_origin.x;
      text_holder_rect.y = label_origin.y;
      text_holder_rect.width = label_size.width + icon_width;
      if (text_holder_rect.x + text_holder_rect.width > image.cols)
          text_holder_rect.width = image.cols - text_holder_rect.x - 1;
      text_holder_rect.height = label_size.height + icon_height;
      if (text_holder_rect.y + text_holder_rect.height > image.rows)
          text_holder_rect.height = image.rows - text_holder_rect.y - 1;
      cv::Mat roi = image(text_holder_rect);
      cv::Mat text_holder (roi.size(), CV_8UC3, cv::Scalar(0,0,0));
      double alpha = 0.3;
      cv::addWeighted(text_holder, alpha, roi, 1.0 - alpha, 0.0, roi);
      label_origin.x+= icon_width;
      label_origin.y+= text_holder_rect.height / 3;

      cv::putText(image,
                  label_one.str(),
                  label_origin,
                  font_face,
                  font_scale,
                  cv::Scalar(cv::Scalar(object.color.r, object.color.g, object.color.b)),//CV_RGB(255, 255, 255),
                  1,
                  CV_AA);
      label_origin.y+= text_holder_rect.height / 3;
    }
  }

  void DrawImageRect(const autoware_msgs::DetectedObjectArray &in_detected_object,
                                    cv::Mat &image)
  {
    int RectangleThickness = 2;
    for(const autoware_msgs::DetectedObject& object : in_detected_object.objects)
      {
          if (object.x >= 0
              && object.y >= 0
              && object.width > 0
              && object.height > 0)
          {
              int x2 = object.x + object.width;
              if (x2 >= image.cols)
                  x2 = image.cols - 1;
              int y2 = object.y + object.height;
              if (y2 >= image.rows)
                  y2 = image.rows - 1;

              cv::Mat image_roi = image(cv::Rect(cv::Point(object.x, object.y),  cv::Point(x2, y2)));
              cv::Mat color_fill(image_roi.size(), CV_8UC3,  cv::Scalar(object.color.r, object.color.g, object.color.b));
              double alpha = 0.15;
              cv::addWeighted(color_fill, alpha, image_roi, 1.0 - alpha , 0.0, image_roi);
              // Draw rectangle
              cv::rectangle(image,
                            cv::Point(object.x, object.y),
                            cv::Point(x2, y2),
                            cv::Scalar(cv::Scalar(object.color.r, object.color.g, object.color.b)),
                            RectangleThickness,
                            CV_AA,
                            0);
          }
      }
      // Draw object information label
      DrawLabel(in_detected_object, image);
  }

  //ugly implemtation, forgive me.
  void detectorCallback1(const sensor_msgs::ImageConstPtr& msg1) {
    vector<sensor_msgs::ImageConstPtr> list;
    list.insert(list.end(), msg1);
    detectorCallback(list);
  }

  void detectorCallback2(const sensor_msgs::ImageConstPtr& msg1,
                         const sensor_msgs::ImageConstPtr& msg2) {
    vector<sensor_msgs::ImageConstPtr> list;
    list.insert(list.end(), msg1);
    list.insert(list.end(), msg2);
    detectorCallback(list);
  }
  void detectorCallback3(const sensor_msgs::ImageConstPtr& msg1,
                         const sensor_msgs::ImageConstPtr& msg2,
                         const sensor_msgs::ImageConstPtr& msg3) {
    vector<sensor_msgs::ImageConstPtr> list;
    list.insert(list.end(), msg1);
    list.insert(list.end(), msg2);
    list.insert(list.end(), msg3);
    detectorCallback(list);
  }
  void detectorCallback4(const sensor_msgs::ImageConstPtr& msg1,
                         const sensor_msgs::ImageConstPtr& msg2,
                         const sensor_msgs::ImageConstPtr& msg3,
                         const sensor_msgs::ImageConstPtr& msg4) {
    vector<sensor_msgs::ImageConstPtr> list;
    list.insert(list.end(), msg1);
    list.insert(list.end(), msg2);
    list.insert(list.end(), msg3);
    list.insert(list.end(), msg4);
    detectorCallback(list);
  }

  void detectorCallback(vector<sensor_msgs::ImageConstPtr> image_list) {
    // input vector in batch
    vector<cv::Mat> cvimages;
    vector<vector<miivii::yolo::MvYoloResult>> results(image_list.size());

    for (auto it = begin(image_list); it != end(image_list); ++it) {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(*it, enc::BGR8);
      cvimages.push_back(cv_ptr->image.clone());
    }

    if (bInitOk) {
      double time;
      if (debug_time) time = what_time_is_it_now();
#if APEX
  #if JPVERSION == 420
        MiiViiInput* tmp = Preprocess(cvimages, shape[0], shape[1]);
        miiviiDetector->Inference(tmp, results);
        delete tmp;
  #elif JPVERSION == 422
        miiviiDetector->Inference(cvimages, results);
  #endif
#else
  MiiViiInput* tmp = Preprocess(cvimages, shape[0], shape[1]);
        miiviiDetector->Inference(tmp, results);
        delete tmp;
#endif
      if (debug_time)
        ROS_INFO("infer cost %f ms\n", (what_time_is_it_now() - time) * 1000);
    }

    for (int i = 0; i < int(image_list.size()); i++) {
      // We publish empty message if no detection.
      // This will be kind to bag record.
      std::vector<RectClassScore<float>> detections;
      RectClassScore<float> detection;
      cv_bridge::CvImagePtr cv_ptr =
          cv_bridge::toCvCopy(image_list[i], enc::BGR8);

      for (vector<miivii::yolo::MvYoloResult>::iterator iter = results[i].begin();
            iter != results[i].end(); iter++) {
        if( iter->x0> cv_ptr->image.cols || iter->y0> cv_ptr->image.rows )
          continue;
        detection.x = iter->x0;
        detection.y = iter->y0;
        detection.w = iter->x1 - iter->x0;
        detection.h = iter->y1 - iter->y0;
        if (( detection.x + detection.w) >= cv_ptr->image.cols)
          detection.w = cv_ptr->image.cols - 1-detection.x;
        if ((detection.y + detection.h) >= cv_ptr->image.rows)
          detection.h = cv_ptr->image.rows - 1-detection.y;

       // printf("<MIIVII-DEBUG> [%s] %d (W %d, H %d ) RECT(%d %d %d %d)\n", __FUNCTION__, __LINE__, cv_ptr->image.cols, cv_ptr->image.rows,iter->x0, iter->y0, iter->x1, iter->y1);
        detection.score = iter->score;
        detection.class_type = iter->image_label;

        detections.push_back(detection);
      }

      // publish result message
      autoware_msgs::DetectedObjectArray output_message;
      output_message.header = image_list[i]->header;

      output_message.header.stamp = (cv_ptr->header.stamp);

      convert_rect_to_image_obj(detections, output_message);
      publisher_objects_[i].publish(output_message);

      if (publish_result_image) {

        DrawImageRect(output_message,  cv_ptr->image);

        cv_bridge::CvImage out_msg;
        out_msg.header = image_list[i]->header;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = cv_ptr->image;

        image_pub_[i].publish(out_msg.toImageMsg());
      }
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "miivii_detector_node");
  miivii_detector ic;
  ros::spin();
  return 0;
}
