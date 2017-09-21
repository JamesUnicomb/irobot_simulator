import numpy as np
from math import *

import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot

import lasagne
from lasagne.updates import adam
from lasagne.layers import DenseLayer, InputLayer, get_output
from lasagne.nonlinearities import rectify, softmax, tanh

n_input  = 6
n_output = 3

def Q_network(state):
    input_state = InputLayer(input_var = state,
                             shape     = (None, n_input))

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


X_state  = T.fmatrix()
X_action = T.bvector()
X_reward = T.fmatrix()

X_action_hot = to_one_hot(X_action, n_output)

Q_network_values = Q_network(X_state)

Q_values_ = get_output(Q_network_values)
Q_values = theano.function(inputs               = [X_state],
                           outputs              = Q_values_,
                           allow_input_downcast = True)

Q_max_ = T.argmax(Q_values_)
Q_max  = theano.function(inputs               = [X_state],
                         outputs              = Q_max_,
                         allow_input_downcast = True)


Q_max_value_ = T.max(Q_values_)
Q_max_value  = theano.function(inputs               = [X_state],
                               outputs              = Q_max_value_,
                               allow_input_downcast = True)

Q_action_ = T.batched_dot(Q_values_, X_action_hot)
Q_action  = theano.function(inputs               = [X_state, X_action],
                            outputs              = Q_action_,
                            allow_input_downcast = True)


#print Q_max([[100,220,50,0,100,200]])
#print Q_max_value([[100,220,50,0,100,200]])

#print Q_values([[100,220,50,0,100,200]])

#print Q_action([[100,220,50,0,100,200],[100,220,50,0,100,200],[100,220,50,0,100,200]], [0,1,2])


