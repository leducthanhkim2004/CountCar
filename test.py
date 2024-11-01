import depthai as dai
import cv2
import numpy as np
# Initialize the pipeline
pipeline = dai.Pipeline()

# Set up left and right mono cameras for stereo depth
cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Create stereo depth node
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(200)
stereo.setOutputDepth(True)
stereo.setOutputRectified(True)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)  # For more precise depth maps

# Link cameras to stereo depth node
cam_left.out.link(stereo.left)
cam_right.out.link(stereo.right)

# Set up output for the depth map
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    # Output queue to retrieve the depth map
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        depth_frame = depth_queue.get()
        depth_image = depth_frame.getFrame()

        # Display the depth map
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_colored = cv2.applyColorMap(depth_image_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Depth Map", depth_image_colored)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
