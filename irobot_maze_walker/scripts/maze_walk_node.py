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

from sklearn.utils import shuffle

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
        self.episode_length = 0
        self.state          = list()
        self.next_state     = list()
        self.action         = int()
        self.reward         = float()

        self.replay_memory  = list()
        self.buffer_size    = 1000
 
        rospy.Subscriber('/maze_walk', MazeWalk, self.walker_cb)

    def walker_cb(self,
                  maze_walk_msg):
        self.next_state = maze_walk_msg.observations
        self.reward     = maze_walk_msg.reward

        self.state = self.next_state
        self.action = np.random.randint(0,3) # TODO greedy_eps(state)

        if maze_walk_msg.done:
            self.next_state      = maze_walk_msg.observations
            self.reward          = maze_walk_msg.reward
            self.episode_number += 1
            self.episode_length  = 0

        self.replay_memory.append((self.state, self.action, self.reward, self.next_state))
        self.replay_memory = self.replay_memory[-self.buffer_size:]

        states, actions, rewards, next_states = zip(*shuffle(self.replay_memory)[:10])


        print 'ep: %d; len: %d' % (self.episode_number, self.episode_length)
        self.episode_length += 1

        maze_walker.set_discrete_action(self.action)



lh = learnerHandler()

def main():
    rospy.spin()


if __name__=='__main__':
    main()
