#!/usr/bin/env python

import rospy
from irobot_maze_walker import iRobotMazeWalker
from maze_walker_msgs.msg import MazeWalk
import numpy as np
from math import *

import theano
import theano.tensor as T

import lasagne
from lasagne.updates import adam
from lasagne.layers import DenseLayer, InputLayer, get_output
from lasagne.nonlinearities import rectify, softmax, tanh

n_input  = 6
n_output = 3

def q_network(state):
    input_state = InputLayer(state,
                             shape = (None,n_input))

    dense_1     = DenseLayer(input_state,
                             num_units    = n_input,
                             nonlinearity = rectify)

    dense_2     = DenseLayer(dense_1,
                             num_units    = n_input,
                             nonlinearity = rectify)

    dense_out   = DenseLayer(dense_2,
                             num_units    = n_output,
                             nonlinearity = None)

    return dense_out




rospy.init_node('mase_walker_node')

maze_walker = iRobotMazeWalker()

class learnerHandler:
    def __init__(self):
        self.episode_number = 0
        self.reset          = True
        self.next_state     = []
        self.current_state  = []
        self.actions        = []
        self.rewards        = []
 
        rospy.Subscriber('/maze_walk', MazeWalk, self.walker_cb)

    def walker_cb(self,
                  maze_walk_msg):
        self.next_state = maze_walk_msg.observations
        self.reward     = maze_walk_msg.reward

        self.replay_memory.append((self.state, self.action, self.reward, self.next_state))

        self.state = self.next_state

        self.action = greedy_eps(state)







        if self.reset:
            self.state           = maze_walk_msg.observations
            self.reset           = False
        else:
            if maze_walk_msg.done:
                self.next_state      = maze_walk_msg.observations
                self.rewards         = maze_walk_msg.reward
                self.reset           = True
                self.episode_number += 1

            self.action = np.random.randint(0,3) #TODO Function(self.state)

            replay_memory.append((self.state, self.action, 
            self.state = self.next_state



            print 'ep: %d; len: %d' % (self.episode_number, len(self.rewards))

            maze_walker.set_discrete_action(action)

lh = learnerHandler()

def main():
    rospy.spin()


if __name__=='__main__':
    main()
