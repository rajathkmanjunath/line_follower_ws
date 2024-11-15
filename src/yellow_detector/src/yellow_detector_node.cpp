#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/Float64.h>  // Change to Float64 for steering angle

class YellowDetector {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    image_transport::Publisher mask_pub_;
    ros::Publisher steer_pub_;  // Publisher for steering angle
    ros::Publisher velocity_pub_;

    // HSV range for yellow detection
    int low_H = 7, low_S = 20, low_V =70;
    int high_H = 22, high_S = 200, high_V = 220;

    // Proportional control gain
    const double Kp = 0.1;  // Adjust this gain for your system

    // Clamping function for C++11
    double clamp(double value, double min_value, double max_value) {
        return std::max(min_value, std::min(value, max_value));
    }

public:
    YellowDetector() : it_(nh_) {
        // Subscribe to RealSense color image
        image_sub_ = it_.subscribe("/camera/color/image_raw", 1, 
            &YellowDetector::imageCallback, this);
        
        // Publishers for processed image and mask
        image_pub_ = it_.advertise("/yellow_detector/output", 1);
        mask_pub_ = it_.advertise("/yellow_detector/mask", 1);
        steer_pub_ = nh_.advertise<std_msgs::Float64>("/mpc/steer_angle", 1);
        velocity_pub_ = nh_.advertise<std_msgs::Float64>("/mpc/velocity_value", 1);

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

        // Create mask for yellow colors
        cv::Mat yellow_mask;
        cv::inRange(hsv_image, 
                   cv::Scalar(low_H, low_S, low_V), 
                   cv::Scalar(high_H, high_S, high_V), 
                   yellow_mask);

        // Apply morphological operations to reduce noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(yellow_mask, yellow_mask, kernel);
        cv::dilate(yellow_mask, yellow_mask, kernel);

        // Get image dimensions
        int height = yellow_mask.rows;
        int width = yellow_mask.cols;

        // Calculate ROI dimensions to consider the full width and bottom half
        int roi_y_start = height / 2; // Start at 50% of height
        int roi_height = height - roi_y_start; // Use bottom half

        // Create ROI for full-width bottom portion
        cv::Rect roi(0, roi_y_start, width, roi_height);
        cv::Mat bottom_half = yellow_mask(roi);

        // Calculate centroid of yellow pixels in ROI
        cv::Moments m = cv::moments(bottom_half, true);
        std_msgs::Float64 steer_msg;

        if (m.m00 > 0) {  // If yellow pixels are detected
            // Calculate centroid (relative to ROI)
            double cx = m.m10 / m.m00;
            
            // Calculate difference from center of ROI
            double center_diff = cx - (422.0);
            
            // Calculate steering angle using proportional control
            double steering_angle = -Kp * center_diff;  // Negative sign to correct direction
            
            // Clip the steering angle between -30 and 30 degrees
            steer_msg.data = clamp(steering_angle, -30.0, 30.0);
        } else {
            steer_msg.data = 0.0;  // No yellow pixels detected, go straight
        }

        std_msgs::Float64 velocity_msg;
        velocity_msg.data = 12.0;

        // Publish the steering angle
        steer_pub_.publish(steer_msg);
        velocity_pub_.publish(velocity_msg);

        // Visualize the result
        cv::Mat result = cv_ptr->image.clone();
        
        // Black out everything except our ROI
        cv::Mat mask = cv::Mat::zeros(result.size(), CV_8UC1);
        cv::rectangle(mask, roi, cv::Scalar(255), -1);  // Fill ROI with white
        
        // Create final mask combining yellow detection and ROI
        cv::Mat final_mask;
        yellow_mask.copyTo(final_mask);
        final_mask.setTo(0, mask == 0);  // Zero out everything outside ROI

        // Apply mask to original image
        result.copyTo(result, final_mask);

        // Draw ROI rectangle for visualization
        cv::rectangle(result, roi, cv::Scalar(0, 255, 0), 2);  // Green rectangle

        // Publish masked image and mask
        sensor_msgs::ImagePtr output_msg = 
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", result).toImageMsg();
        image_pub_.publish(output_msg);

        sensor_msgs::ImagePtr mask_msg = 
            cv_bridge::CvImage(std_msgs::Header(), "mono8", final_mask).toImageMsg();
        mask_pub_.publish(mask_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "yellow_detector_node");
    YellowDetector yellow_detector;
    ros::spin();
    return 0;
} 
