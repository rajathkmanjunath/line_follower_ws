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
    bool line_follow_enabled_ = false;  // Flag to check if line following is enabled
    ros::Publisher state_pub_;  // Publisher for state messages
    bool yellow_line_detected_ = false;  // Flag to track yellow line detection

    // HSV range for yellow detection
    int low_H = 5, low_S = 50, low_V = 100;
    int high_H = 25, high_S = 228, high_V = 200;

    // PID control gains
    const double Kp = 0.1;  // Proportional gain
    const double Ki = 0.01; // Integral gain
    const double Kd = 0.05; // Derivative gain

    double previous_error_ = 0.0;  // Previous error for derivative calculation
    std::queue<double> error_queue_;  // Queue to store the last 30 errors
    const size_t max_queue_size_ = 30;  // Maximum size of the error queue
    double accumulated_error_ = 0.0;  // Running sum of the errors for integral calculation

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

        // Subscribe to the trigger topic
        trigger_sub_ = nh_.subscribe("/trigger_linefollow", 1, 
            &YellowDetector::triggerCallback, this);

        // Publisher for state messages
        state_pub_ = nh_.advertise<std_msgs::String>("/state", 1);

        ROS_INFO("Yellow detector node initialized");
    }

    // Callback to handle trigger messages
    void triggerCallback(const std_msgs::Bool::ConstPtr& msg) {
        line_follow_enabled_ = msg->data;
    }

    // Callback to process incoming images
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        if (!line_follow_enabled_) {
            return;  // Exit if line following is not enabled
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
                   cv::Scalar(100, 150, 0), 
                   cv::Scalar(140, 255, 255), 
                   blue_mask);

        // Apply morphological operations to reduce noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::erode(blue_mask, blue_mask, kernel);
        cv::dilate(blue_mask, blue_mask, kernel);

        cv::Moments blue_m = cv::moments(blue_mask, true);

        if (blue_m.m00 > 0) {  // If blue line is detected
            std_msgs::String state_msg;
            state_msg.data = "Intermediate_stop";
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
            double center_diff = cx - (width / 2.0);

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
            double steering_angle = -(Kp * error + Ki * accumulated_error_ + Kd * derivative);
            
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
