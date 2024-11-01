import numpy as np 
import depthai as dai
import cv2
def getframe(queue):
    frame= queue.get()
    return frame.getCvFrame()
def getMonoCamera(pipeline,isLeft):
    mono =pipeline.createMonoCamera()
        #set Camera resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)# SET (640*640)
    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

if __name__ == '__main__':    
    pipeline =dai.Pipeline()# CREATE A PIPELINE 
        #SET UP LEFT AND RIGHT CAMERA
    monoLeft=getMonoCamera(pipeline,True)
    monoRight=getMonoCamera(pipeline,False)
        #create a stereo node
    stereo= pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)
        #Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
        
        #create a preview output
    disparity_out = pipeline.createXLinkOut()
    disparity_out.setStreamName("disparity")
    stereo.disparity.link(disparity_out.input)
__name__== "__main__"   
with dai.Device(pipeline) as device:
    q_disp=device.getOutputQueue(name="disparity",maxSize=1,blocking=False)
    while True:
        in_disp = q_disp.get()
        disp_frame = in_disp.getCvFrame()
        disparityMultiplier = 255 / stereo.getMaxDisparity()

            # Colormap disparity for display.
        disp_frame = (disp_frame * disparityMultiplier).astype('uint8')
        disp_frame = cv2.applyColorMap(disp_frame, cv2.COLORMAP_JET)
        cv2.imshow("Disparity Map", disp_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
