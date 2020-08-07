#!/usr/bin/env python
"""
@package simplex_blocks
@file LEC_Driving_impl.py
"""
import rospy
import tf
import actionlib
from alc_msgs.msg import Steering_Angle
from sensor_msgs.msg import Image
from alc_msgs.msg import Velocity
# from alc_ros.msg import AssuranceMonitorConfidence, AssuranceMonitorConfidenceStamped
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
class LEC_DrivingImplementation(object):
    """
    Class to contain Developer implementation.
    """

    def __init__(self):
        """
        Definition and initialization of class attributes
        """
        # parameters
        self.screen = rospy.get_param("~screen", "output")
        self.deployment_folder = rospy.get_param("~DeepNNCar-Key")
        self.network_interface = None
        self.assurance_monitor_paths = [];
        self.ams = ''
        # self.am_topic_str = "//am_LEC_Control"
        # self.lec_topic_str = "//LEC_Control"
        self.pub_assurance_monitor_output = ''
        self.pub_lec_input_output = ''
        self.init_sl(self.deployment_folder)
        # self.init_am(self.deployment_folder)
        # self.init_lec_input_output_publisher()
        self.topic_sub_color_image = rospy.get_param("~topic_sub_color_image")
        self.topic_sub_current_velocity = rospy.get_param("~topic_sub_current_velocity")
        self.topic_pub_steering_angle = rospy.get_param("~topic_pub_steering_angle")
        # subscribers
        self.color_image_ = rospy.Subscriber(self.topic_sub_color_image, Image, self.callback_color_image, queue_size=1)
        self.in_color_image = Image()
        self.in_color_image_updated = bool()
        self.current_velocity_ = rospy.Subscriber(self.topic_sub_current_velocity, Velocity,
                                                  self.callback_current_velocity, queue_size=1)
        self.in_current_velocity = Velocity()
        self.in_current_velocity_updated = bool()
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
        miny = -1
        maxy = 1
        # print("2")
        img = cv2.resize(frame, (200, 66))
        img = img / 255.
        inputs = np.array(img)[np.newaxis]
        with self.graph.as_default():
            set_session(self.sess)
            steer = self.model.predict(inputs, batch_size=1)
        # steer = self.execute_sl(img)
        steering = (float(steer) * (maxy - miny)) + miny
        steering = round(steering, 2)
        # print(steering)
        msg = Steering_Angle()
        msg.angle = steering
        self.steering_angle_.publish(msg)
        # ********** protected region user implementation of subscriber callback for color_image end   **********#
        pass

    # callback for subscriber - current_velocity
    def callback_current_velocity(self, msg):
        """
        callback at reception of message on topic current_velocity
        """
        self.in_current_velocity = msg
        self.in_current_velocity_updated = True

        # ********** protected region user implementation of subscriber callback for current_velocity begin **********#
        # ********** protected region user implementation of subscriber callback for current_velocity end   **********#
        pass

    # initialize_sl
    def init_sl(self, model_folder, **kwargs):
        from alc_utils.network_interface import NetworkInterface
        self.network_interface = NetworkInterface()
        self.network_interface.load(model_folder)

    # initialize assurance monitor
    def init_am(self, model_folder):
        self.assurance_monitor_paths.append(model_folder);
        import alc_utils.assurance_monitor
        self.ams = alc_utils.assurance_monitor.MultiAssuranceMonitor()
        self.ams.load(self.assurance_monitor_paths)

        # set up publisher for assurance monitor
        if (self.ams and self.ams.assurance_monitors):
            self.pub_assured_network_output = rospy.Publisher(self.am_topic_str, AssuranceMonitorConfidenceStamped,
                                                              queue_size=1)

    # initialize assurance monitor
    def init_lec_input_output_publisher(self):
        # need one or more message types
        # self.pub_lec_input_output = rospy.Publisher(self.lec_topic_str, LEC_Input_Output_Message_Type, queue_size=1)
        pass

    def execute_sl(self, raw_input, use_batch_mode=True):
        return self.network_interface.predict(raw_input, batch_mode=use_batch_mode)

    def publish_lec_input_output(self, states, actions):
        # convert states and actions to message and publish
        # return lec_input_msg( i.e state) and lec_output_msg (i.e. action)
        pass

    def step_am(self, states, actions):
        # do this for assurance monitors
        # lec_input_msg, lec_output_msg = self.publish_lec_input_output(states, actions)

        # invoke assurance monitor with the messages
        # if (self.ams and self.ams.assurance_monitors and self.pub_assurance_monitor_output):
        #    assurance_result = self.ams.evaluate(lec_input_msg,lec_output_msg)
        #    if (assurance_result is not None):
        #        assurance_msg = AssuranceMonitorConfidenceStamped()
        #        assurance_msg.header.stamp = rospy.Time.now()
        #        for i in range(0, len(assurance_result)):
        #            confidence_msg = AssuranceMonitorConfidence()
        #            confidence_msg.type = AssuranceMonitorConfidence.TYPE_SVDD
        #            confidence_level_bounds = assurance_result[i][:3]
        #            confidence_msg.values = confidence_level_bounds
        #            assurance_msg.confs.append(confidence_msg)
        #        self.pub_assured_network_output.publish(assurance_msg)
        pass

    def update(self, event):
        # ********** protected region user update functions begin **********#
        # ********** protected region user update functions end   **********#
        return


# ********** protected region user additional functions begin **********#
# ********** protected region user additional functions end   **********#


def main():
    """
    @brief Entry point of the package.
    Instanciate the node interface containing the Developer implementation
    @return nothing
    """
    rospy.init_node("LEC_Driving", anonymous=False)
    node = LEC_DrivingImplementation()
    rospy.Timer(rospy.Duration(1.0 / 10), node.update)
    rospy.spin()


if __name__ == '__main__':
    main()