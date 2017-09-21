#!/usr/bin/env python

import rospy
import numpy as np

from sensor_msgs.msg import LaserScan
from ca_msgs.msg import Bumper

class iRobotBumper:
    def __init__(self,
                 bumper_topic   = '/bumper',
                 ir_array_topic = '/create/sim_ir_array',
                 laser_topic    = '/create/sim_scan'):

        self.left  = False
        self.right = False

        rospy.Subscriber(laser_topic, LaserScan, self.laser_cb)
        rospy.Subscriber(ir_array_topic, LaserScan, self.ir_array_cb)

        self.bumper_pub = rospy.Publisher(bumper_topic, Bumper, queue_size=2)


    def laser_cb(self,
                 laser_msg):

        r = np.array(laser_msg.ranges)

        self.left  = (r[:270]<0.1725).any()
        self.right = (r[90:]<0.1725).any()

    def ir_array_cb(self,
                    ir_msg):

        d = np.nan_to_num(np.array(ir_msg.ranges)) - 0.165

        ir_dist = (2e4 * np.sqrt(d) * np.exp(-20.0 * d)).astype(np.int16) + np.random.randint(1,20,6)
        ir_bool = (ir_dist > 150)

        bumper_msg = Bumper()
        bumper_msg.header = ir_msg.header

        bumper_msg.is_left_pressed
        bumper_msg.is_right_pressed
        bumper_msg.is_light_left
        bumper_msg.is_light_front_left
        bumper_msg.is_light_center_left
        bumper_msg.is_light_center_right
        bumper_msg.is_light_front_right
        bumper_msg.is_light_right
        bumper_msg.light_signal_left
        bumper_msg.light_signal_front_left
        bumper_msg.light_signal_center_left
        bumper_msg.light_signal_center_right
        bumper_msg.light_signal_front_right
        bumper_msg.light_signal_right

        
