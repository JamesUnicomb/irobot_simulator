#!/usr/bin/env python

import os, time
import rospy
import numpy as np
from math import *

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from ca_msgs.msg import Bumper
from maze_walker_msgs.msg import MazeWalk

from std_srvs.srv import Empty


class iRobotMazeWalker:
    def __init__(self,
                 cmd_vel_topic    = '/cmd_vel',
                 odom_topic       = '/odom',
                 bumper_topic     = '/bumper',
                 imu_topic        = '/imu/data',
                 maze_walk_topic  = '/maze_walk',
                 control_reward   = 0.0,
                 bump_reward      = -50.0,
                 move_reward_w    = 2.0,
                 move_reward_b    = -0.1):

        self.control_reward     = control_reward
        self.bump_reward        = bump_reward
        self.move_reward_weight = move_reward_w
        self.move_reward_bias   = move_reward_b

        self.done         = False
        self.reward       = 0.0
        self.action       = 0
        self.observations = [0, 0, 0, 0, 0, 0]

        self.linear_velocity     = 0.0
        self.angular_velocity    = 0.0        
        self.linear_acceleration = 0.0

        rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        rospy.Subscriber(bumper_topic, Bumper, self.bumper_cb)
        rospy.Subscriber(imu_topic, Imu, self.imu_cb)

        self.cmd_vel_pub   = rospy.Publisher(cmd_vel_topic, Twist, queue_size=3)
        self.maze_walk_pub = rospy.Publisher(maze_walk_topic, MazeWalk, queue_size=0)

        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)


    def reward_function(self):
        reward_move = self.move_reward_weight * (self.linear_velocity + self.move_reward_bias)

        reward_imu  = 0.0
        
        reward_bump = self.bump_reward * self.done

        return reward_move + reward_bump + reward_imu


    def odom_cb(self,
                odom_msg):
        self.linear_velocity = odom_msg.twist.twist.linear.x
        self.angular_velocity = odom_msg.twist.twist.angular.z


    def imu_cb(self,
               imu_msg):
        self.linear_acceleration = imu_msg.linear_acceleration.x


    def bumper_cb(self,
                  bumper_msg):
        self.done         = (bumper_msg.is_left_pressed)or(bumper_msg.is_right_pressed)

        self.reward       = self.reward_function()

        self.observations = [bumper_msg.light_signal_left,
                             bumper_msg.light_signal_front_left,
                             bumper_msg.light_signal_center_left,
                             bumper_msg.light_signal_center_right,
                             bumper_msg.light_signal_front_right,
                             bumper_msg.light_signal_right]

        maze_walk_msg              = MazeWalk()
        maze_walk_msg.header       = bumper_msg.header
        maze_walk_msg.done         = self.done
        maze_walk_msg.reward       = self.reward
        maze_walk_msg.observations = self.observations

        self.maze_walk_pub.publish(maze_walk_msg)

        if self.done:
            self.reset_simulation()


    def set_discrete_action(self,
                            choice):
        cmd_msg = Twist()

        if choice==0:
            cmd_msg.linear.x  = 0.4
        elif choice==1:
            cmd_msg.angular.z = 1.0
        elif choice==2:
            cmd_msg.angular.z = -1.0
        
        self.cmd_vel_pub.publish(cmd_msg)


    def set_continuous_action(self,
                              lin_vel,
                              ang_vel):
        cmd_msg = Twist()

        cmd_msg.linear.x = lin_vel
        cmd_msg.angular.z = ang_vel
        
        self.cmd_vel_pub.publish(cmd_msg)
