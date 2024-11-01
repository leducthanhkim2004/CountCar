import depthai as dai
import cv2
import numpy as np
import os
import time

# Base folder paths for saving data
base_rgb_path = "D:/car/rgb"
base_depth_path = "D:/car/depth_map"
frame_id = 0

# Create a pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(10)  # Set FPS for RGB camera

cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setFps(10)  # Set FPS for left mono camera

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setFps(10)  # Set FPS for right mono camera

stereo = pipeline.createStereoDepth()
stereo.setOutputDepth(True)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
cam_left.out.link(stereo.left)
cam_right.out.link(stereo.right)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=8, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=8, blocking=False)

    while True:
        # Get frames
        rgb_frame = rgb_queue.get().getCvFrame()
        depth_frame = depth_queue.get().getFrame()

        # Get the current date and timestamp
        timestamp = time.strftime("%Y-%m-%d")
        

        # Define paths based on the current date
        rgb_path = os.path.join(base_rgb_path, timestamp)
        depth_path = os.path.join(base_depth_path,timestamp)

        # Create directories if they do not exist
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)

        # Save the RGB and depth frames with additional information in the filename
        cv2.imwrite(f"{rgb_path}/frame_{frame_id}.png", rgb_frame)
        cv2.imwrite(f"{depth_path}/frame_{frame_id}.png", depth_frame)

        frame_id += 1
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
