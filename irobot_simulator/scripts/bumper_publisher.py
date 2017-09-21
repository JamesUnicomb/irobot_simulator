#!/usr/bin/env python

import rospy

from irobot_bumper import iRobotBumper

def main():
    rospy.init_node('sim_bumper_node')

    robot_bumper = iRobotBumper()

    rospy.spin()


if __name__=='__main__':
    main()
    
