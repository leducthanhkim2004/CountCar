#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Configuration flags
extended_disparity = False
subpixel = False
lr_check = True

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

# Configure depth
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Post-processing settings
config = depth.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
depth.initialConfig.set(config)

# Configure mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Link cameras to depth node
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

# XLink output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

# Run the pipeline
with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)

    while True:
        frame = getFrame(q)
        
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_colored = cv2.applyColorMap(frame_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imshow("Stereo_Processed", frame_colored)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
