import cv2
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt

class HSVThresholder:
    def __init__(self):
        # Create windows for trackbars
        cv2.namedWindow('HSV Thresholder')
        cv2.namedWindow('Original | Filtered')
        
        # Create trackbars for HSV lower and upper bounds
        cv2.createTrackbar('H_low', 'HSV Thresholder', 0, 179, lambda x: None)
        cv2.createTrackbar('S_low', 'HSV Thresholder', 0, 255, lambda x: None)
        cv2.createTrackbar('V_low', 'HSV Thresholder', 0, 255, lambda x: None)
        cv2.createTrackbar('H_high', 'HSV Thresholder', 179, 179, lambda x: None)
        cv2.createTrackbar('S_high', 'HSV Thresholder', 255, 255, lambda x: None)
        cv2.createTrackbar('V_high', 'HSV Thresholder', 255, 255, lambda x: None)
        
        # Set default values (typical for yellow)
        cv2.setTrackbarPos('H_low', 'HSV Thresholder', 20)
        cv2.setTrackbarPos('S_low', 'HSV Thresholder', 100)
        cv2.setTrackbarPos('V_low', 'HSV Thresholder', 100)
        cv2.setTrackbarPos('H_high', 'HSV Thresholder', 30)
        cv2.setTrackbarPos('S_high', 'HSV Thresholder', 255)
        cv2.setTrackbarPos('V_high', 'HSV Thresholder', 255)
        
        # Store the current HSV image and current mouse position
        self.current_hsv = None
        self.current_mouse_pos = None
        
        # Set up mouse callback
        cv2.setMouseCallback('Original | Filtered', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        # Update current mouse position regardless of movement
        self.current_mouse_pos = (x, y)

    def get_hsv_bounds(self):
        # Get current positions of trackbars
        h_low = cv2.getTrackbarPos('H_low', 'HSV Thresholder')
        s_low = cv2.getTrackbarPos('S_low', 'HSV Thresholder')
        v_low = cv2.getTrackbarPos('V_low', 'HSV Thresholder')
        h_high = cv2.getTrackbarPos('H_high', 'HSV Thresholder')
        s_high = cv2.getTrackbarPos('S_high', 'HSV Thresholder')
        v_high = cv2.getTrackbarPos('V_high', 'HSV Thresholder')
        
        return (h_low, s_low, v_low), (h_high, s_high, v_high)

def process_image(image, hsv_thresholder):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Store HSV image in the thresholder
    hsv_thresholder.current_hsv = hsv
    
    # Get current HSV bounds
    (h_low, s_low, v_low), (h_high, s_high, v_high) = hsv_thresholder.get_hsv_bounds()
    
    # Create mask
    lower_yellow = np.array([h_low, s_low, v_low])
    upper_yellow = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return mask, result

def main():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    # Initialize HSV thresholder
    hsv_thresholder = HSVThresholder()
    
    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            # Convert images to numpy arrays
            frame = np.asanyarray(color_frame.get_data())
            
            # Process frame
            mask, result = process_image(frame, hsv_thresholder)
            
            # Stack images horizontally for display
            display = np.hstack((frame, result))
            
            # Show results
            cv2.imshow('Original | Filtered', display)
            cv2.imshow('Mask', mask)
            
            # Get current threshold values
            (h_low, s_low, v_low), (h_high, s_high, v_high) = hsv_thresholder.get_hsv_bounds()
            
            # Create status string
            status = f"Threshold - HSV Lower: ({h_low}, {s_low}, {v_low}) Upper: ({h_high}, {s_high}, {v_high})"
            
            # Add current mouse position HSV values if within original image
            if hsv_thresholder.current_mouse_pos:
                x, y = hsv_thresholder.current_mouse_pos
                if x < frame.shape[1] and y < frame.shape[0] and x >= 0 and y >= 0:
                    hsv_value = hsv_thresholder.current_hsv[y, x]
                    status = f"HSV at ({x}, {y}): H: {hsv_value[0]}, S: {hsv_value[1]}, V: {hsv_value[2]}\n" + status
            
            # Print status (using \r to stay on same line and clear previous output)
            print(f"\r{status}", end='')
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()