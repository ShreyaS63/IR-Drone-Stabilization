#!/usr/bin/env python
import cv2
import numpy as np
import time
import rospy
import tf
import argparse
from cv_bridge import*
from geometry_msgs.msg import PoseStamped
from math import*
from pykalman import KalmanFilter
from mavros.utils import *
import message_filters
from sensor_msgs.msg import*
from mavros import*
from std_msgs.msg import String


#intantiate CvBridge
bridge = CvBridge()
msg = PoseStamped()
pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
# #IMU data
# def callback_imu(data):
#     global data_imu 
#     data_imu = data


## pattern noise filtering
def ok(a,arr):
    for index,i in enumerate(arr):
        if((i[0]-a[0])**2+(i[1]-a[1])**2)<20:
            if(i[2]>a[2]):
                return
            else:
                arr.pop(index)
                arr.append(a)
                return
    arr.append(a)
    return
    

def marker_points(image):

    stamp = time.time()

    ## convert rgb to grayscale and blur it
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (11,11),0)   
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray_image)  

    ##threshold the image to reveal light regions in the blurred image
    thresh = cv2.threshold(gray_image,120,255,cv2.THRESH_BINARY)[1]
    ## perform a series of erosions and dilations to remove any small blobs of noise from the thresholded image
    cv2.Dilate(thresh, thresh, None, 18)  
    cv2.Erode(thresh, thresh, None, 10)
    thr = thresh.copy()

    #Create Kalman Filter
    kalman = cv2.CreateKalman(4,2,0)
    kalman_state = cv2.CreateMat(4, 1, cv2.CV_32FC1)
    kalman_process_noise = cv2.CreateMat(4, 1, cv2.CV_32FC1)
    kalman_measurement = cv2.CreateMat(2, 1, cv2.CV_32FC1)

    cp1 = []
    cp11 = []
    points = []
    
    #find the bright contours 
    (_,cnts, _) = cv2.findContours(thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print "Found %d LEDs" % (len(cnts))

        
    while cnts:
        ##find area and compute the blob centers 
        area=cv2.ContourArea(list(cnts))
        
        for c in range(len(cnts)):
            M = cv2.moments(cnts[c])
            cp1 = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            cp11.append(cp1)
        

            # set previous state for prediction
            kalman.state_pre[0,0]  = cp1
            kalman.state_pre[1,0]  = 0
           
            # set Kalman Filter
            cv2.SetIdentity(kalman.measurement_matrix, cv2.RealScalar(1))
            cv2.SetIdentity(kalman.process_noise_cov, cv2.RealScalar(1e-5))
            cv2.SetIdentity(kalman.measurement_noise_cov, cv2.RealScalar(1e-1))
            cv2.SetIdentity(kalman.error_cov_post, cv2.RealScalar(1))

            #Prediction
            kalman_prediction = cv2.KalmanPredict(kalman)
            predict_pt  = (int(kalman_prediction[0,0]),int( kalman_prediction[1,0]))
            predict_pt1.append(predict_pt)
            
            #Correction
            kalman_estimated = cv2.KalmanCorrect(kalman, kalman_measurement)
            state_pt = (kalman_estimated[0,0], kalman_estimated[1,0])

            #Measurement
            kalman_measurement[0, 0] = cp11[0]
            
    
    if len(cnts)==6: 
        #draw circle around brightest blobs        
        #for i in cnts:
            #cv2.circle(image, (i[1],i[0]), 3, (0, 0, 255), -1)
            #cv2.imshow("IMAGE", image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #return np.array(pixels,dtype = "double")
        return np.array(cp11,dtype = "double")
        
    elif len(cnts)>6:
        print ("No marker")
        return marker_points(image)
    else:
        print("No marker")
        return marker_points(image)   
    

def pose_estimation_callback(image):
    print("Received image!")
    # Use cv_bridge() to convert the ROS image to OpenCV format
    image = bridge.imgmsg_to_cv2(image, "bgr8")
    stamp =  time.time()
    #3D model points
    model_points = np.array([
                            (0.0, 0.0, 0.0),             
                            (0.0, 140.0, 0.0),       
                            (70.0, 70.0, 0.0),
                            (140.0, 0.0, 0.0),
                            (140.0, 70.0, 0.0), 
                            (140.0, 140.0, 0.0),
                            ], dtype = "double")
    size = image.shape
    
    #2D image points
    image_points = marker_points(image)
    image_points = np.array(image_points,dtype = "double")
    #print(image_points)
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    camera_matrix = np.array(
                             [[1.13449375e+03,0.00000000e+00,2.74258830e+02],
                             [0.00000000e+00,1.13658526e+03,1.88387172e+02],
                             [0, 0, 1]], dtype = "double"
                             )
       
    #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    dist_coeffs = np.array([-0.27456514, 0.58715645, -0.00300159, 0.00023316, 0.00000000])
    stamp2 = time.time()
    success, rVec, tVec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    print ('solvePnPtime:')
    print (time.time()-stamp2)
    print ('time elapsed:')
    #x-right y-down
    
    #print translation and rotation vectors
    print ('translation vector:')
    print (tVec)
    print ('rotation vector')
    print (rVec)

    #get rotation matrix using Rogrigues function
    Rmat = cv2.Rodrigues(rVec)[0] 
    R = - np.transpose(Rmat)
    #position in world coordinates
    position = R*(tVec/1000)
    
    #calculate the euler angles https://stackoverflow.com/questions/16265714/camera-pose-estimation-opencv-pnp
    rad = 180/pi
    roll = atan2(-R[2][1], R[2][2])
    pitch = asin(R[2][0])
    yaw = atan2(-R[1][0], R[0][0])

    #print position
    #print roll, pitch, yaw

    # rospy.Subscriber('mavros/imu/data', Imu, callback_imu)
    # quaternion = (
    #         data_imu.orientation.x,
    #         data_imu.orientation.y,
    #         data_imu.orientation.z,
    #         data_imu.orientation.w)
    # euler = tf.transformations.euler_from_quaternion(quaternion)
   
    worldFrame = "world"
    msg.header.seq = 0
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = worldFrame
    msg.pose.position.x = position[0][0]
    msg.pose.position.y = position[1][1]
    msg.pose.position.z = position[2][2]
    quaternion = tf.transformations.quaternion_from_euler(roll,pitch,yaw)
    msg.pose.orientation.x = quaternion[0]
    msg.pose.orientation.y = quaternion[1]
    msg.pose.orientation.z = quaternion[2]
    msg.pose.orientation.w = quaternion[3]
    msg.header.seq += 1
    msg.header.stamp = rospy.Time.now()
    #print (msg.pose)
    pub.publish(msg)  

    #     print ('\nImu data:\n')
    #     #print (data_imu.orientation)
    #     #print ('roll: ', euler[0])
    #     print ('roll: ', degrees(euler[0])
    #     #print('pitch: ', euler[1])
    #     print ('pitch: ', degrees(euler[1])
    #     print


def drone_setup():
    
    print ('hello world')
    rospy.init_node('drone_setup', anonymous=True)

    # while(1):
    rate = rospy.Rate(10)

    while( not rospy.is_shutdown() ): 
        # Subscribe to the camera image and set the appropriate callback
        rospy.Subscriber('/cv_camera/image_raw', Image, pose_estimation_callback)
        #rospy.logininfo(msg)   
        print ('publishing')  
        rate.sleep()        

if __name__ == '__main__':
    try:
        drone_setup()
    # except rospy.ROSInterruptException:
    except KeyboardInterrupt:
        pass
