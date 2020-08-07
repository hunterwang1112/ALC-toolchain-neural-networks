#!/usr/bin/env python
"""
@package simplex_blocks
@file DeepNNCar_Connector_impl.py
"""
import rospy
import tf
import actionlib
from alc_msgs.msg import Steering_Angle
from alc_msgs.msg import Velocity
# ********** protected region user include package begin **********#
import pigpio
import RPi.GPIO as GPIO
import signal

pi = pigpio.pi()
print(pi.connected)
import time
import message_filters


# ********** protected region user include package end   **********#
class DeepNNCar_ConnectorImplementation(object):
    """
    Class to contain Developer implementation.
    """

    def __init__(self):
        """
        Definition and initialization of class attributes
        """
        self.topic_sub_LEC_steering_angle_out = rospy.get_param("~topic_sub_LEC_steering_angle_out")
        self.topic_sub_OpenCV_steering_angle_out = rospy.get_param("~topic_sub_OpenCV_steering_angle_out")
        self.topic_sub_velocity_out = rospy.get_param("~topic_sub_velocity_out")
        self.topic_pub_measured_velocity = rospy.get_param("~topic_pub_measured_velocity")
        # subscribers
        self.LEC_steering_angle_out_ = rospy.Subscriber(self.topic_sub_LEC_steering_angle_out, Steering_Angle,
                                                        self.callback_LEC_steering_angle_out, queue_size=1)
        self.in_LEC_steering_angle_out = Steering_Angle()
        self.in_LEC_steering_angle_out_updated = bool()
        self.OpenCV_steering_angle_out_ = rospy.Subscriber(self.topic_sub_OpenCV_steering_angle_out, Steering_Angle,
                                                           self.callback_OpenCV_steering_angle_out, queue_size=1)
        self.in_OpenCV_steering_angle_out = Steering_Angle()
        self.in_OpenCV_steering_angle_out_updated = bool()
        self.velocity_out_ = rospy.Subscriber(self.topic_sub_velocity_out, Velocity, self.callback_velocity_out,
                                              queue_size=1)
        self.in_velocity_out = Velocity()
        self.in_velocity_out_updated = bool()
        # publishers
        self.measured_velocity_ = rospy.Publisher(self.topic_pub_measured_velocity, Velocity, queue_size=1)
        self.out_measured_velocity = Velocity()
        self.out_measured_velocity_active = bool()

        # ********** protected region user member variables begin **********#
        # ********** protected region user member variables end   **********#

    def callback_steering_angles(self, msg1, msg2):
        """
        callback at reception of message on topic LEC_steering_angle_out and topic OpenCV_steering_angle_out
        """
        self.in_LEC_steering_angle_out = msg1
        self.in_LEC_steering_angle_out_updated = True

        self.in_OpenCV_steering_angle_out = msg2
        self.in_OpenCV_steering_angle_out_updated = True

        LEC_angle = msg1.angle
        OpenCV_angle = msg2.angle

        if ((float(LEC_angle) - float(OpenCV_angle)) > 2.0):
            w1 = 0.2
            w2 = 0.8
            steer = w1 * float(OpenCV_angle) + w2 * float(LEC_angle)
        else:
            steer = LEC_angle

        print(steer)
        pi.hardware_PWM(19, 100, int(steer * 10000))

        pass

    # callback for subscriber - LEC_steering_angle_out
    def callback_LEC_steering_angle_out(self, msg):
        """
        callback at reception of message on topic LEC_steering_angle_out
        """
        self.in_LEC_steering_angle_out = msg
        self.in_LEC_steering_angle_out_updated = True

        # ********** protected region user implementation of subscriber callback for LEC_steering_angle_out begin **********#
        # ********** protected region user implementation of subscriber callback for LEC_steering_angle_out end   **********#
        pass

    # callback for subscriber - OpenCV_steering_angle_out
    def callback_OpenCV_steering_angle_out(self, msg):
        """
        callback at reception of message on topic OpenCV_steering_angle_out
        """
        self.in_OpenCV_steering_angle_out = msg
        self.in_OpenCV_steering_angle_out_updated = True

        # ********** protected region user implementation of subscriber callback for OpenCV_steering_angle_out begin **********#
        # ********** protected region user implementation of subscriber callback for OpenCV_steering_angle_out end   **********#
        pass

    # callback for subscriber - velocity_out
    def callback_velocity_out(self, msg):
        """
        callback at reception of message on topic velocity_out
        """
        self.in_velocity_out = msg
        self.in_velocity_out_updated = True

        # ********** protected region user implementation of subscriber callback for velocity_out begin **********#
        speed = msg.velocity
        print(speed)
        pi.hardware_PWM(18, 100, int(speed * 10000))
        # ********** protected region user implementation of subscriber callback for velocity_out end   **********#
        pass


# ********** protected region user additional functions begin **********#
def initGPIO(freq, steer, speed):
    pi.hardware_PWM(18, freq, 0)
    pi.hardware_PWM(19, freq, 0)
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    # GPIO.setup(sensor,GPIO.IN,GPIO.PUD_UP)


def changeDutyCycle(freq, steer, speed):
    pi.hardware_PWM(19, freq, int(steer * 10000))
    pi.hardware_PWM(18, freq, int(speed * 10000))


def cleanGPIO():
    signal.alarm(0)
    print('Cleaning up GPIO')
    GPIO.cleanup()
    pi.hardware_PWM(18, 100, 0)
    pi.hardware_PWM(19, 100, 0)
    pi.stop()


# ********** protected region user additional functions end   **********#


def main():
    """
    @brief Entry point of the package.
    Instanciate the node interface containing the Developer implementation
    @return nothing
    """
    rospy.init_node("DeepNNCar_Connector", anonymous=False)
    node = DeepNNCar_ConnectorImplementation()
    rospy.spin()


if __name__ == '__main__':
    main()