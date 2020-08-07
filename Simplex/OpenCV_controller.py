#!/usr/bin/env python
"""
@package simplex_blocks
@file OpenCV_impl.py
"""
import rospy
import tf
import actionlib
from alc_msgs.msg import Steering_Angle
from sensor_msgs.msg import Image
# ********** protected region user include package begin **********#
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
from keras.models import load_model
import rospkg
# global graph
# graph = tf.get_default_graph()
from keras import backend as K
import math
import cv2
import csv
from tensorflow.python.keras.backend import set_session


# ********** protected region user include package end   **********#
class OpenCVImplementation(object):
    """
    Class to contain Developer implementation.
    """

    def __init__(self):
        """
        Definition and initialization of class attributes
        """
        self.topic_sub_color_image = rospy.get_param("~topic_sub_color_image")
        self.topic_pub_steering_angle = rospy.get_param("~topic_pub_steering_angle")
        # subscribers
        self.color_image_ = rospy.Subscriber(self.topic_sub_color_image, Image, self.callback_color_image, queue_size=1)
        self.in_color_image = Image()
        self.in_color_image_updated = bool()
        # publishers
        self.steering_angle_ = rospy.Publisher(self.topic_pub_steering_angle, Steering_Angle, queue_size=1)
        self.out_steering_angle = Steering_Angle()
        self.out_steering_angle_active = bool()

        # ********** protected region user member variables begin **********#
        # ********** protected region user member variables end   **********#

    # callback for subscriber - color_image
    def callback_color_image(self, msg):
        """
        callback at reception of message on topic color_image
        """
        self.in_color_image = msg
        self.in_color_image_updated = True

        # ********** protected region user implementation of subscriber callback for color_image begin **********#
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        laneDetection, blur, steering_OpenCV = SafetyManagerAutonomousClient.runSafetyManager(frame)
        # miny = -1
        # maxy = 1
        # #print("2")
        # img = cv2.resize(frame, (200, 66))
        # img = img / 255.
        # inputs = np.array(img)[np.newaxis]
        # with self.graph.as_default():
        #     set_session(self.sess)
        #     steer = self.model.predict(inputs, batch_size=1)
        # #steer = self.execute_sl(img)
        # steering=(float(steer)*(maxy-miny))+miny
        # steering=round(steering, 2)
        # print(steering)
        msg = Steering_Angle()
        msg.angle = steering_OpenCV
        self.steering_angle_.publish(msg)
        # ********** protected region user implementation of subscriber callback for color_image end   **********#
        pass

    def update(self, event):
        # ********** protected region user update functions begin **********#
        # ********** protected region user update functions end   **********#
        return


# ********** protected region user additional functions begin **********#
# laneDetect()
# Description: process image to determine if off track
# parameters:  image => image matrix
# returnVal    -1 => no track found
#              0  => track found
def laneDetect(image):
    # mask out all but white
    # filter noise from canny edge
    gauss_gray = cv2.GaussianBlur(image, (5, 5), 0)
    mask_white = cv2.inRange(gauss_gray, 215, 255)
    # apply canny edge
    low_threshold = 50
    high_threshold = 125
    canny_edges = cv2.Canny(mask_white, low_threshold, high_threshold)
    # mask region of interest
    mask = np.zeros((66, 200), np.uint8)
    pts = np.array([[0, 30], [0, 60], [60, 60], [60, 30]])
    cv2.drawContours(mask, np.int32([pts]), 0, 255, -1)
    pts = np.array([[199, 30], [199, 60], [139, 60], [139, 30]])
    cv2.drawContours(mask, np.int32([pts]), 0, 255, -1)
    # hough line transform
    canny_edges = cv2.bitwise_or(canny_edges, canny_edges, mask=mask)
    leftSide = canny_edges[30:60, 0:60].copy()
    rightSide = canny_edges[30:60, 139:199].copy()
    linesLeft = cv2.HoughLines(leftSide, 3, np.pi / 180, 23)
    linesRight = cv2.HoughLines(rightSide, 3, np.pi / 180, 23)
    if (linesRight is not None and linesLeft is not None):
        # print('Straight')
        return 3
    elif (linesRight is None and linesLeft is not None):
        # print('Turning Right')
        return 2
    elif (linesLeft is None and linesRight is not None):
        # print('Turning Left')
        return 1
    else:
        # print('Stop')
        return -1


# measureBlurrienss()
# Description: returns blurriness measurement (variance of laplacian)
# parameters:  frame => image matrix
# returnVal    higher value indicates less blurry image
def measureBlurriness(frame):
    gauss_gray = cv2.GaussianBlur(frame, (3, 3), 0)
    fm = cv2.Laplacian(gauss_gray, cv2.CV_64F).var()
    return fm


# run()
# Description: Returns the results from the Safety manager
# parameters: frame => image matrix
# returnVal: lane detection result, image blurriness, previous prediction cycle time
def runSafetyManager(frame):
    blur = measureBlurriness(frame)
    lanedetection = laneDetect(frame)
    # Discrete steering calculation from lanedetection values
    if (lanedetection == 3):
        steeringSS = 15
    elif (lanedetection == 2):
        steeringSS = 20
    elif (lanedetection == 1):
        steeringSS = 10
    else:
        steeringSS = 0
    return lanedetection, blur, steeringSS


# ********** protected region user additional functions end   **********#


def main():
    """
    @brief Entry point of the package.
    Instanciate the node interface containing the Developer implementation
    @return nothing
    """
    rospy.init_node("OpenCV", anonymous=False)
    node = OpenCVImplementation()
    rospy.Timer(rospy.Duration(1.0 / 10), node.update)
    rospy.spin()


if __name__ == '__main__':
    main()