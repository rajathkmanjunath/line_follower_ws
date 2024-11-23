#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>  // Include for trigger message
#include <queue>

class YellowDetector {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    image_transport::Publisher mask_pub_;
    ros::Publisher steer_pub_;  // Publisher for steering angle
    ros::Publisher velocity_pub_;
    ros::Subscriber trigger_sub_;  // Subscriber for the trigger topic
    double prev_stop_trigger_;
    bool line_follow_enabled_ = false;  // Flag to check if line following is enabled
    ros::Publisher state_pub_;  // Publisher for state messages
    bool yellow_line_detected_ = false;  // Flag to track yellow line detection

    // HSV range for yellow detection - now as member variables without initialization
    int low_H, low_S, low_V;
    int high_H, high_S, high_V;

    // PID control gains - now as member variables without initialization
    double Kp;  // Proportional gain
    double Ki;  // Integral gain
    double Kd;  // Derivative gain
    double dt;  // Time step for PID control
    double stop_trigger_wait_time_;

    double prev_time = 0.0;


    double previous_error_ = 0.0;  // Previous error for derivative calculation
    std::queue<double> error_queue_;  // Queue to store the last 30 errors
    const size_t max_queue_size_ = 30;  // Maximum size of the error queue
    double accumulated_error_ = 0.0;  // Running sum of the errors for integral calculation

    image_transport::Publisher blue_mask_pub_;  // New publisher for blue mask

    // Parameters for blue line detection - now as member variables without initialization
    int MIN_CLUSTER_SIZE;
    int REQUIRED_CONSECUTIVE_FRAMES;
    int blue_detection_counter_ = 0;  // Counter for consecutive blue line detections

    // New member variables for blue detection HSV range
    int blue_low_H, blue_low_S, blue_low_V;
    int blue_high_H, blue_high_S, blue_high_V;
    
    // New member variables for steering limits and center reference
    double steering_limit_min_, steering_limit_max_;
    double center_reference_;

    // Clamping function for C++11
    double clamp(double value, double min_value, double max_value) {
        return std::max(min_value, std::min(value, max_value));
    }

    // Function to check if pixels are clustered
    bool checkClustering(const cv::Mat& mask) {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask.clone(), contours, hierarchy, 
                        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Check each contour's area
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area >= MIN_CLUSTER_SIZE) {
                return true;  // Found a cluster of sufficient size
            }
        }
        return false;  // No clusters of sufficient size found
    }

public:
    YellowDetector() : it_(nh_) {
        // Get private node handle for parameters
        ros::NodeHandle private_nh("~");

        // Load HSV parameters
        private_nh.param("hsv/low_H", low_H, 10);
        private_nh.param("hsv/low_S", low_S, 100);
        private_nh.param("hsv/low_V", low_V, 100);
        private_nh.param("hsv/high_H", high_H, 20);
        private_nh.param("hsv/high_S", high_S, 255);
        private_nh.param("hsv/high_V", high_V, 255);

        // Load PID parameters
        private_nh.param("pid/Kp", Kp, 0.1);
        private_nh.param("pid/Ki", Ki, 0.003);
        private_nh.param("pid/Kd", Kd, 0.03);
        private_nh.param("pid/dt", dt, 0.01);
        private_nh.param("pid/stop_trigger_wait_time", stop_trigger_wait_time_, 2.0);

        // Load other parameters
        private_nh.param("detection/min_cluster_size", MIN_CLUSTER_SIZE, 100);
        private_nh.param("detection/required_consecutive_frames", REQUIRED_CONSECUTIVE_FRAMES, 5);

        // Load blue HSV parameters
        private_nh.param("blue_hsv/low_H", blue_low_H, 110);
        private_nh.param("blue_hsv/low_S", blue_low_S, 25);
        private_nh.param("blue_hsv/low_V", blue_low_V, 85);
        private_nh.param("blue_hsv/high_H", blue_high_H, 120);
        private_nh.param("blue_hsv/high_S", blue_high_S, 175);
        private_nh.param("blue_hsv/high_V", blue_high_V, 145);

        // Load steering limits and center reference
        private_nh.param("control/steering_limit_min", steering_limit_min_, -30.0);
        private_nh.param("control/steering_limit_max", steering_limit_max_, 30.0);
        private_nh.param("control/center_reference", center_reference_, 770.0);

        // Subscribe to RealSense color image
        image_sub_ = it_.subscribe("/camera/color/image_raw", 1, 
            &YellowDetector::imageCallback, this);
        
        // Publishers for processed image and mask
        image_pub_ = it_.advertise("/yellow_detector/output", 1);
        mask_pub_ = it_.advertise("/yellow_detector/mask", 1);
        steer_pub_ = nh_.advertise<std_msgs::Float64>("/mpc/steer_angle", 1);
        velocity_pub_ = nh_.advertise<std_msgs::Float64>("/mpc/velocity", 1);

        // Subscribe to the trigger topic
        trigger_sub_ = nh_.subscribe("/trigger_linefollow", 1, 
            &YellowDetector::triggerCallback, this);

        // Publisher for state messages
        state_pub_ = nh_.advertise<std_msgs::String>("/state", 1);

        // Add new publisher for blue mask
        blue_mask_pub_ = it_.advertise("/yellow_detector/blue_mask", 1);

        ROS_INFO("Yellow detector node initialized");
    }

    // Callback to handle trigger messages
    void triggerCallback(const std_msgs::Bool::ConstPtr& msg) {
        line_follow_enabled_ = msg->data;
    }

    // Callback to process incoming images
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        if (!line_follow_enabled_ || ros::Time::now().toSec() - prev_time <= stop_trigger_wait_time_) {
            blue_detection_counter_ = 0;  // Reset counter when disabled
            prev_time = ros::Time::now().toSec();
            return;
        }

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

        // Get image dimensions
        int height = hsv_image.rows;
        int width = hsv_image.cols;

        // Calculate ROI dimensions to consider the full width and bottom half
        int roi_y_start = height / 2; // Start at 50% of height
        int roi_height = height - roi_y_start; // Use bottom half

        // Create ROI for full-width bottom portion
        cv::Rect roi(0, roi_y_start, width, roi_height);
        cv::Mat hsv_roi = hsv_image(roi);  // Crop the image to the ROI

        // Detect horizontal blue line in the cropped ROI
        cv::Mat blue_mask;
        cv::inRange(hsv_roi, 
                   cv::Scalar(blue_low_H, blue_low_S, blue_low_V), 
                   cv::Scalar(blue_high_H, blue_high_S, blue_high_V), 
                   blue_mask);

        // Apply morphological operations to reduce noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(blue_mask, blue_mask, kernel);
        cv::dilate(blue_mask, blue_mask, kernel);

        cv::Moments blue_m = cv::moments(blue_mask, true);

        // Create full-size blue mask for visualization
        cv::Mat full_blue_mask = cv::Mat::zeros(cv_ptr->image.size(), CV_8UC1);
        blue_mask.copyTo(full_blue_mask(roi));

        // Check for clustered blue line detection
        bool blue_line_detected = false;
        if (blue_m.m00 > 0) {  // If any blue pixels are detected
            // Check if the detected pixels form a cluster
            if (checkClustering(blue_mask)) {
                blue_detection_counter_++;  // Increment counter
                
                // Check if we've had enough consecutive detections
                if (blue_detection_counter_ >= REQUIRED_CONSECUTIVE_FRAMES) {
                    blue_line_detected = true;
                }
            } else {
                blue_detection_counter_ = 0;  // Reset counter if no clusters found
            }
        } else {
            blue_detection_counter_ = 0;  // Reset counter if no blue pixels detected
        }

        // Publish blue mask for visualization
        sensor_msgs::ImagePtr blue_mask_msg = 
            cv_bridge::CvImage(std_msgs::Header(), "mono8", full_blue_mask).toImageMsg();
        blue_mask_pub_.publish(blue_mask_msg);

        // Handle blue line detection
        if (blue_line_detected) {
            std_msgs::String state_msg;
            state_msg.data = "intermediate_stop";
            state_pub_.publish(state_msg);
            yellow_line_detected_ = false;  // Reset yellow line detection flag
            return;  // Exit early if blue line is detected
        }

        // Detect yellow line in the cropped ROI
        cv::Mat yellow_mask;
        cv::inRange(hsv_roi, 
                   cv::Scalar(low_H, low_S, low_V), 
                   cv::Scalar(high_H, high_S, high_V), 
                   yellow_mask);

        // Apply morphological operations to reduce noise
        cv::erode(yellow_mask, yellow_mask, kernel);
        cv::dilate(yellow_mask, yellow_mask, kernel);

        // Calculate centroid of yellow pixels in ROI
        cv::Moments m = cv::moments(yellow_mask, true);
        std_msgs::Float64 steer_msg;

        if (m.m00 > 0) {  // If yellow pixels are detected
            if (!yellow_line_detected_) {
                std_msgs::String state_msg;
                state_msg.data = "line_detected";
                state_pub_.publish(state_msg);
                yellow_line_detected_ = true;  // Set yellow line detection flag
            }

            // Calculate centroid (relative to ROI)
            double cx = m.m10 / m.m00;
            
            // Calculate difference from center of ROI
            double center_diff = cx - center_reference_;

            // PID control calculations
            double error = center_diff;

            // Update the error queue and accumulated error
            error_queue_.push(error);
            accumulated_error_ += error;
            if (error_queue_.size() > max_queue_size_) {
                accumulated_error_ -= error_queue_.front();  // Subtract the oldest error
                error_queue_.pop();  // Remove the oldest error if the queue is full
            }

            double derivative = error - previous_error_;  // Calculate the derivative of error
            previous_error_ = error;  // Update previous error

            // Calculate steering angle using PID control
            double steering_angle = -(Kp * error + Ki * accumulated_error_ * dt + Kd * derivative);
            
            // Clip the steering angle between -30 and 30 degrees
            steer_msg.data = clamp(steering_angle, steering_limit_min_, steering_limit_max_);
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
        cv::Mat final_mask = cv::Mat::zeros(result.size(), CV_8UC1);
        yellow_mask.copyTo(final_mask(roi));  // Copy the yellow mask to the ROI in the final mask

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
