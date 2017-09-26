#include <cstdint>
#include <thread>
#include <mutex>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/pcl_base.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

//Global variables
pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_color(new pcl::PointCloud<pcl::PointXYZRGB>());
bool cancelled = false, update_pointcloud = false;

namespace {
pcl::PointCloud<pcl::PointXYZRGB>::Ptr toPclColor(const sensor_msgs::PointCloud2ConstPtr& input) {
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*input, pcl_pc2);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);

  return temp_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr toPcl(const sensor_msgs::PointCloud2ConstPtr& input) {
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*input, pcl_pc2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);

  return temp_cloud;
}

void createDepthHistogram(const cv::Mat &depth_raw, cv::Mat &depth_color) {
  depth_color.create(depth_raw.size(), CV_8UC3);

  static uint32_t histogram[0x10000];
  memset(histogram, 0, sizeof(histogram));

  for (int i = 0; i < depth_raw.rows; i++)
    for (int j = 0; j < depth_raw.cols; j++)
      ++histogram[depth_raw.ptr<ushort>(i)[j]];

  for (int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i-1]; // Build a cumulative histogram for the indices in [1,0xFFFF]

  for(int i = 0; i < depth_raw.rows; i++) {
    for (int j = 0; j < depth_raw.cols; j++) {
      uint16_t d = depth_raw.ptr<ushort>(i)[j];
      if (d) {
        int f = (int)(histogram[d] * 255 / histogram[0xFFFF]); // 0-255 based on histogram location
        depth_color.ptr<cv::Vec3b>(i)[j] = cv::Vec3b(f, 0, 255 - f);
      } else {
        depth_color.ptr<cv::Vec3b>(i)[j] = cv::Vec3b(0, 5, 20);
      }
    }
  }
}

class ViewerWorker {
public:
  explicit ViewerWorker(const bool color_mode, std::mutex &mutex) :
    m_colorMode(color_mode), m_mutex(mutex) { }

  void run() {
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pointcloud_color);
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_pointcloud_color(new pcl::PointCloud<pcl::PointXYZRGB>());

    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters ();
    viewer->setPosition(640+80, 480+80);
    viewer->setCameraPosition(0, 0, -0.25, 0, -1, 0);
    viewer->setSize(640, 480);

    bool init = true;
    bool local_update = false, local_cancelled = false;
    while (!local_cancelled) {
      {
        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);

        if (lock.owns_lock()) {
          local_update = update_pointcloud;
          update_pointcloud = false;
          local_cancelled = cancelled;

          if (m_colorMode) {
            local_pointcloud_color = pointcloud_color->makeShared();
          } else {
            local_pointcloud = pointcloud->makeShared();
          }
        }
      }

      if (local_update && !local_cancelled) {
        local_update = false;

        if (init) {
          if (m_colorMode) {
            viewer->addPointCloud<pcl::PointXYZRGB> (local_pointcloud_color, rgb, "RGB sample cloud");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "RGB sample cloud");
          } else {
            viewer->addPointCloud<pcl::PointXYZ>(local_pointcloud, "sample cloud");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
          }
          init = false;
        } else {
          if (m_colorMode) {
            viewer->updatePointCloud<pcl::PointXYZRGB>(local_pointcloud_color, rgb, "RGB sample cloud");
          } else {
            viewer->updatePointCloud<pcl::PointXYZ>(local_pointcloud, "sample cloud");
          }
        }
      }

      viewer->spinOnce(10);
    }

    std::cout << "End of point cloud display thread" << std::endl;
  }

private:
  bool m_colorMode;
  std::mutex &m_mutex;
};

class Visualization {
  ros::NodeHandle m_nh;
  unsigned int m_queue_size;

  //https://answers.ros.org/question/9705/synchronizer-and-image_transportsubscriber/
  //http://wiki.ros.org/message_filters
  message_filters::Subscriber<sensor_msgs::Image> m_image_color_sub;
  message_filters::Subscriber<sensor_msgs::Image> m_image_infrared_sub;
  message_filters::Subscriber<sensor_msgs::Image> m_image_depth_sub;
  message_filters::Subscriber<sensor_msgs::PointCloud2> m_pointcloud_sub;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> m_sync;

  cv::Mat m_I_depth;

  std::thread m_viewer_thread;
  std::mutex m_mutex;
  ViewerWorker m_viewer;

public:
  Visualization() : m_nh(), m_queue_size(1), m_image_color_sub(m_nh, "/camera/color", m_queue_size),
      m_image_infrared_sub(m_nh, "/camera/ir", m_queue_size), m_image_depth_sub(m_nh, "/camera/depth", m_queue_size),
      m_pointcloud_sub(m_nh, "/camera/pointcloud", m_queue_size),
      // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
      m_sync(MySyncPolicy(10), m_image_color_sub, m_image_infrared_sub, m_image_depth_sub, m_pointcloud_sub),
      m_I_depth(), m_viewer_thread(), m_mutex(), m_viewer(false, m_mutex)
  {
    m_sync.registerCallback(boost::bind(&Visualization::callback, this, _1, _2, _3, _4));

    m_viewer_thread = std::thread(&ViewerWorker::run, &m_viewer);

    cv::namedWindow("RGB");
    cv::namedWindow("IR");
    cv::namedWindow("Depth");
    cv::moveWindow("RGB", 10, 200);
    cv::moveWindow("IR", 10, 560);
    cv::moveWindow("Depth", 650, 200);
  }

  virtual ~Visualization() {
    m_viewer_thread.join();
    std::lock_guard<std::mutex> lock(m_mutex);
    cancelled = true;
  }

  void spin() {
    ros::Rate loop_rate(60);

    while (m_nh.ok()) {
      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  void callback(const sensor_msgs::ImageConstPtr& image_color, const sensor_msgs::ImageConstPtr& image_infrared,
                const sensor_msgs::ImageConstPtr& image_depth, const sensor_msgs::PointCloud2ConstPtr& msg_pointcloud) {
    try {
      cv::Mat frame_color = cv_bridge::toCvShare(image_color, "bgr8")->image;
      cv::imshow("RGB", frame_color);

      cv::Mat frame_infrared = cv_bridge::toCvShare(image_infrared)->image;
      if (frame_infrared.depth() == CV_8U) {
        cv::imshow("IR", frame_infrared);
      } else if (frame_infrared.depth() == CV_16U) {
        cv::Mat frame_infrared_8u(frame_infrared.size(), CV_8U);

        for (int i = 0; i < frame_infrared.rows; i++) {
          for (int j = 0; j < frame_infrared.cols; j++) {
            frame_infrared_8u.ptr<uchar>(i)[j] = (frame_infrared.ptr<ushort>(i)[j] >> 8);
          }
        }
        cv::imshow("IR", frame_infrared_8u);
      }

      cv::Mat frame_depth = cv_bridge::toCvShare(image_depth)->image;
      if (frame_depth.depth() == CV_16U) {
        createDepthHistogram(frame_depth, m_I_depth);
      } else if (frame_depth.depth() == CV_32F) {
        cv::Mat frame_depth_16u(frame_depth.size(), CV_16U);
        for (int i = 0; i < frame_depth.rows; i++) {
          for (int j = 0; j < frame_depth.cols; j++) {
            //Asus Xtion Pro Live returns a depth map in m
            frame_depth_16u.ptr<ushort>(i)[j] = frame_depth.ptr<float>(i)[j] * 1000.0;
          }
        }

        createDepthHistogram(frame_depth_16u, m_I_depth);
      }

      cv::imshow("Depth", m_I_depth);
      cv::waitKey(1);

      {
        std::lock_guard<std::mutex> lock(m_mutex);
        pointcloud = toPcl(msg_pointcloud);
        update_pointcloud = true;
      }
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }
};

} //namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "RGBID_Visualization_opencv_node");
  Visualization visualization;
  visualization.spin();

  return 0;
}

