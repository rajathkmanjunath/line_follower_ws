#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/Float32.h>

class YellowDetector {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    image_transport::Publisher mask_pub_;
    ros::Publisher diff_pub_;

    // Updated HSV range for yellow detection
    int low_H = 5, low_S = 50, low_V = 100;
    int high_H = 25, high_S = 228, high_V = 200;

public:
    YellowDetector() : it_(nh_) {
        // Subscribe to RealSense color image
        image_sub_ = it_.subscribe("/camera/color/image_raw", 1, 
            &YellowDetector::imageCallback, this);
        
        // Publishers for processed image and mask
        image_pub_ = it_.advertise("/yellow_detector/output", 1);
        mask_pub_ = it_.advertise("/yellow_detector/mask", 1);

        // New publisher for lane following difference
        diff_pub_ = nh_.advertise<std_msgs::Float32>("lf_diff", 1);

        ROS_INFO("Yellow detector node initialized");
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Convert BGR to HSV
        cv::Mat hsv_image;
        cv::cvtColor(cv_ptr->image, hsv_image, cv::COLOR_BGR2HSV);

        // Create mask for yellow colors with updated values
        cv::Mat yellow_mask;
        cv::inRange(hsv_image, 
                   cv::Scalar(low_H, low_S, low_V), 
                   cv::Scalar(high_H, high_S, high_V), 
                   yellow_mask);

        // Mask bottom half
        int height = yellow_mask.rows;
        int width = yellow_mask.cols;
        cv::Rect roi(width/4, height/2, width/2, height/2);
        cv::Mat bottom_half = yellow_mask(roi);

        // Calculate centroid of yellow pixels in bottom half
        cv::Moments m = cv::moments(bottom_half, true);
        std_msgs::Float32 diff_msg;

        if (m.m00 > 0) {  // If yellow pixels are detected
            // Calculate centroid
            double cx = m.m10 / m.m00;
            
            // Calculate difference from center
            double center_diff = cx - (width / 2.0);
            diff_msg.data = center_diff;
        } else {
            diff_msg.data = 0.0;  // No yellow pixels detected
        }

        // Publish the difference
        diff_pub_.publish(diff_msg);

        // Apply mask to original image (showing only bottom half)
        cv::Mat result = cv_ptr->image.clone();
        result(cv::Rect(width/4, 0, 3*width/4, height/2)).setTo(cv::Scalar(0,0,0));  // Black out top half
        cv::Mat bottom_mask;
        yellow_mask.copyTo(bottom_mask);
        bottom_mask(cv::Rect(width/4, 0, 3*width/4, height/2)).setTo(0);  // Black out top half of mask
        result.copyTo(result, bottom_mask);  // Apply masked bottom half

        // Publish masked image and mask
        sensor_msgs::ImagePtr output_msg = 
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", result).toImageMsg();
        image_pub_.publish(output_msg);

        sensor_msgs::ImagePtr mask_msg = 
            cv_bridge::CvImage(std_msgs::Header(), "mono8", bottom_mask).toImageMsg();
        mask_pub_.publish(mask_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "yellow_detector_node");
    YellowDetector yellow_detector;
    ros::spin();
    return 0;
} 
