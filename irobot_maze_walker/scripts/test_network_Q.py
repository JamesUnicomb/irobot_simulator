import numpy as np
from math import *

import theano
import theano.tensor as T
from theano.tensor.extra_ops import to_one_hot

import lasagne
from lasagne.updates import adam
from lasagne.objectives import squared_error
from lasagne.layers import DenseLayer, InputLayer, get_output, \
                           get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, softmax, tanh

n_input       = 6
n_output      = 3
gamma         = 0.5
learning_rate = 0.01

eps_max         = 1.0
eps_min         = 0.05
eps_decay_steps = 2000

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


X_current_state  = T.fmatrix()
X_next_state     = T.fmatrix()
X_action         = T.bvector()
X_reward         = T.fvector()

X_action_hot = to_one_hot(X_action, n_output)

Q_target_values = Q_network(X_next_state)
Q_actor_values = Q_network(X_current_state)

Q_target_ = get_output(Q_target_values)
Q_target  = theano.function(inputs               = [X_next_state],
                           outputs              = Q_target_,
                           allow_input_downcast = True)

Q_actor_     = get_output(Q_actor_values)
Q_actor_max_ = T.argmax(Q_actor_)
Q_actor_max  = theano.function(inputs               = [X_current_state],
                               outputs              = Q_actor_max_,
                               allow_input_downcast = True)


Q_target_max_ = T.max(Q_target_)
Q_target_max  = theano.function(inputs               = [X_next_state],
                               outputs              = Q_target_max_,
                               allow_input_downcast = True)

Q_action_ = T.batched_dot(Q_actor_, X_action_hot)
Q_action  = theano.function(inputs               = [X_current_state, X_action],
                            outputs              = Q_action_,
                            allow_input_downcast = True)

loss = squared_error(X_reward + gamma * Q_target_max_, Q_action_)
loss = loss.mean()

params = get_all_params(Q_actor_values)

updates = adam(loss,
               params,
               learning_rate = learning_rate)

update_network = theano.function(inputs               = [X_current_state, 
                                                         X_action, 
                                                         X_reward,
                                                         X_next_state],
                                 outputs              = loss,
                                 updates              = updates,
                                 allow_input_downcast = True)

def update_target():
    param_values = get_all_param_values(Q_actor_values)
    set_all_param_values(Q_target_values, param_values)


def get_action(state, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.random() < epsilon:
        return np.random.randint(0,n_output)
    else:
        return Q_actor_max(state)


for j in range(3000):
    update_network([[0,0,1,1,0,0],[1,1,1,1,0,0],[1,1,0,0,0,0],[0,0,0,0,0,0]], 
                   [0,1,2,0], 
                   [-32.0, 16.0, 16.0, 32.0],
                   [[0,1,1,1,1,0],[1,1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    if j%5==0:
        update_target()
    print Q_target([[0,0,1,1,0,0]])
    print get_action([[0,0,1,1,0,0]],j)
