# This is the valueparameters, to check the tensor shapes yourself
# you can use the following code:

# MDN-RNN with all the values of the parameters

# Importing the libraries
import numpy as np
import tensorflow as tf
from collections import namedtuple

# Setting the Hyperparameters
HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                         ])

# Making a function that returns all the default hyperparameters of the MDN-RNN model


def default_hps():
    return HyperParams(num_steps=2000,
                       # num_steps: the number of training steps
                       # This is how many times we will train the model
                       max_seq_len=1000,
                       # max_seq_len: the maximum length of the input sequences
                       # This is the longest sequence of data we will feed into the model
                       input_seq_width=35,
                       # input_seq_width: the width of each input vector in the sequence
                       # This is how many features each input data point has
                       output_seq_width=32,
                       # output_seq_width: the width of each output vector in the sequence
                       # This is how many features each output data point has
                       rnn_size=256,
                       # rnn_size: the number of units in the RNN cell
                       # This is how many neurons are in each RNN cell
                       batch_size=100,
                       # batch_size: the number of sequences in a batch
                       # This is how many sequences we will process at once
                       grad_clip=1.0,
                       # grad_clip: the value to clip the gradients to
                       # This helps prevent the gradients from getting too large and causing problems
                       num_mixture=5,
                       # num_mixture: the number of mixture components in the MDN
                       # This is how many different distributions we will use to model the data
                       learning_rate=0.001,
                       # learning_rate: the learning rate for the optimizer
                       # This controls how much we adjust the model with each step
                       decay_rate=1.0,
                       # decay_rate: the rate at which the learning rate decays
                       # This controls how quickly the learning rate decreases over time
                       min_learning_rate=0.00001,
                       # min_learning_rate: the minimum learning rate
                       # This is the smallest value the learning rate can reach
                       use_layer_norm=0,
                       # use_layer_norm: whether to use layer normalization
                       # This decides if we should normalize the layers in the RNN
                       use_recurrent_dropout=0,
                       # use_recurrent_dropout: whether to use recurrent dropout
                       # This decides if we should use dropout between the steps in the RNN
                       recurrent_dropout_prob=0.90,
                       # recurrent_dropout_prob: the probability of keeping the dropout in the recurrent layers
                       # This controls how much dropout we use between the steps in the RNN
                       use_input_dropout=0,
                       # use_input_dropout: whether to use input dropout
                       # This decides if we should use dropout on the input data
                       input_dropout_prob=0.90,
                       # input_dropout_prob: the probability of keeping the dropout in the input layers
                       # This controls how much dropout we use on the input data
                       use_output_dropout=0,
                       # use_output_dropout: whether to use output dropout
                       # This decides if we should use dropout on the output data
                       output_dropout_prob=0.90,
                       # output_dropout_prob: the probability of keeping the dropout in the output layers
                       # This controls how much dropout we use on the output data
                       is_training=1)
    # is_training: whether the model is in training mode
    # This decides if we are training the model or using it for inference


# Getting these default hyperparameters
hps = default_hps()

# Building the RNN
num_mixture = hps.num_mixture
KMIX = num_mixture
INWIDTH = hps.input_seq_width
OUTWIDTH = hps.output_seq_width
LENGTH = hps.max_seq_len
if hps.is_training:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # global_step: a variable to keep track of the number of training steps
    # This keeps track of how many times we have trained the model

cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
# cell_fn: function to create the RNN cell, in this case, a LayerNormBasicLSTMCell
# This is the type of cell we are using in our RNN

use_recurrent_dropout = False if hps.use_recurrent_dropout == 0 else True
# use_recurrent_dropout: whether to use recurrent dropout
# This decides if we should use dropout between the steps in the RNN

use_input_dropout = False if hps.use_input_dropout == 0 else True
# use_input_dropout: whether to use input dropout
# This decides if we should use dropout on the input data

use_output_dropout = False if hps.use_output_dropout == 0 else True
# use_output_dropout: whether to use output dropout
# This decides if we should use dropout on the output data

use_layer_norm = False if hps.use_layer_norm == 0 else True
# use_layer_norm: whether to use layer normalization
# This decides if we should use layer normalization in the RNN

if use_recurrent_dropout:
    cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm,
                   dropout_keep_prob=hps.recurrent_dropout_prob)
    # cell: the RNN cell with recurrent dropout and layer normalization
    # This creates the RNN cell with dropout between steps and layer normalization
else:
    cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm)
    # cell: the RNN cell with layer normalization only
    # This creates the RNN cell with only layer normalization

if use_input_dropout:
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, input_keep_prob=hps.input_dropout_prob)
    # cell: the RNN cell wrapped with input dropout
    # This adds dropout to the input data of the RNN cell

if use_output_dropout:
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=hps.output_dropout_prob)
    # cell: the RNN cell wrapped with output dropout
    # This adds dropout to the output data of the RNN cell

sequence_lengths = LENGTH
# sequence_lengths: the maximum length of the input sequences
# This is the maximum length of the input sequences

input_x = tf.placeholder(dtype=tf.float32, shape=[
                         hps.batch_size, hps.max_seq_len, INWIDTH])
# input_x: placeholder for the input tensor
# dtype: data type of the input, in this case, float32
# shape: shape of the input tensor, [batch_size, max_seq_len, input_seq_width]
# This is where we will feed the input data to the RNN

output_x = tf.placeholder(dtype=tf.float32, shape=[
                          hps.batch_size, hps.max_seq_len, OUTWIDTH])
# output_x: placeholder for the output tensor
# dtype: data type of the output, in this case, float32
# shape: shape of the output tensor, [batch_size, max_seq_len, output_seq_width]
# This is where we will feed the output data to the RNN

actual_input_x = input_x
# actual_input_x: the input to the RNN, it is the same as the input_x
# This is the actual input to the RNN, which is the same as input_x

initial_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)
# initial_state: the initial state of the RNN, it is a zero state
# batch_size: number of sequences in a batch
# dtype: data type of the state, in this case, float32
# This creates the initial state of the RNN with all zeros

NOUT = OUTWIDTH * KMIX * 3
# NOUT: the number of output units, calculated as the product of output_seq_width, num_mixture, and 3
# This calculates the number of output units for the MDN

with tf.variable_scope('RNN'):
    output_w = tf.get_variable("output_w", [hps.rnn_size, NOUT])
    # output_w: weights for the output layer
    # name: name of the variable
    # shape: shape of the weights tensor, [rnn_size, NOUT]
    # This creates the weights for the output layer of the RNN

    output_b = tf.get_variable("output_b", [NOUT])
    # output_b: biases for the output layer
    # name: name of the variable
    # shape: shape of the biases tensor, [NOUT]
    # This creates the biases for the output layer of the RNN

output, last_state = tf.nn.dynamic_rnn(cell,
                                       actual_input_x,
                                       initial_state=initial_state,
                                       time_major=False,
                                       swap_memory=True,
                                       dtype=tf.float32,
                                       scope="RNN")
# output: the output of the RNN
# last_state: the final state of the RNN
# cell: the RNN cell to use
# inputs: the input tensor to the RNN
# initial_state: the initial state of the RNN
# time_major: whether the time dimension is the first dimension
# swap_memory: whether to swap memory from GPU to CPU during training
# dtype: data type of the inputs and outputs, in this case, float32
# scope: variable scope for the RNN
# This runs the RNN with the input data and initial state, and gets the output and final state

# Building the MDN
output = tf.reshape(output, [-1, hps.rnn_size])
# output: the output tensor of the RNN
# shape: shape of the output tensor, [-1, hps.rnn_size]
# -1: infers the size of the dimension from the remaining dimensions
# hps.rnn_size: number of units in the RNN cell
# This reshapes the output of the RNN to be a 2D tensor

output = tf.nn.xw_plus_b(output, output_w, output_b)
# output: the output tensor after applying the weights and biases
# tf.nn.xw_plus_b: computes the sum of the matrix multiplication of 'output' and 'output_w' and the bias 'output_b'
# output_w: weights for the output layer
# output_b: biases for the output layer
# This applies the weights and biases to the output of the RNN

output = tf.reshape(output, [-1, KMIX * 3])
# output: the reshaped output tensor
# shape: shape of the output tensor, [-1, KMIX * 3]
# -1: infers the size of the dimension from the remaining dimensions
# KMIX * 3: the number of mixture components times 3 (for logmix, mean, and logstd)
# This reshapes the output again to get the final output of the MDN

final_state = last_state
# final_state: the final state of the RNN
# This is the final state of the RNN after processing the input data

logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
# logSqrtTwoPI: the log of the square root of 2*pi, used in the log-normal distribution
# This is a constant used in the log-normal distribution


def tf_lognormal(y, mean, logstd):
    # y: the target tensor
    # mean: the mean of the Gaussian distributions
    # logstd: the log of the standard deviation of the Gaussian distributions
    return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI
    # tf.square: computes the square of the input tensor
    # This calculates the log-normal distribution


def get_lossfunc(logmix, mean, logstd, y):
    # logmix: the log of the mixture coefficients
    # mean: the mean of the Gaussian distributions
    # logstd: the log of the standard deviation of the Gaussian distributions
    # y: the target tensor

    v = logmix + tf_lognormal(y, mean, logstd)
    v = tf.reduce_logsumexp(v, 1, keepdims=True)
    # input_tensor: the input tensor to the reduce_logsumexp function
    # axis: the axis along which to perform the operation
    # keepdims: whether to keep the reduced dimensions in the output tensor
    return -tf.reduce_mean(v)
    # input_tensor: the input tensor to the reduce_mean function
    # This calculates the loss function for the MDN


def get_mdn_coef(output):
    # output: the output tensor of the MDN

    logmix, mean, logstd = tf.split(output, 3, 1)
    # logmix: the log of the mixture coefficients
    # mean: the mean of the Gaussian distributions
    # logstd: the log of the standard deviation of the Gaussian distributions
    logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
    # input_tensor: the input tensor to the reduce_logsumexp function
    # axis: the axis along which to perform the operation
    # keepdims: whether to keep the reduced dimensions in the output tensor
    return logmix, mean, logstd
    # This splits the output tensor into the mixture coefficients, means, and standard deviations


out_logmix, out_mean, out_logstd = get_mdn_coef(output)
# out_logmix: the log of the mixture coefficients
# out_mean: the mean of the Gaussian distributions
# out_logstd: the log of the standard deviation of the Gaussian distributions
# This gets the mixture coefficients, means, and standard deviations from the output

out_logmix = out_logmix
out_mean = out_mean
out_logstd = out_logstd

# Implementing the training operations
flat_target_data = tf.reshape(output_x, [-1, 1])
# flat_target_data: the flattened target tensor
# shape: shape of the flattened target tensor, [-1, 1]
# this turns a 3-D Tensor into a 2D Tensor with the shape of (3200000,1)
# This reshapes the target data to be a 2D tensor

lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)
# lossfunc: the loss function to minimize during training
# - out_logmix: the log of the mixture coefficients
# - out_mean: the mean of the Gaussian distributions
# - out_logstd: the log of the standard deviation of the Gaussian distributions
# - flat_target_data: the flattened target tensor
# This calculates the loss function for the MDN

cost = tf.reduce_mean(lossfunc)
# cost: the cost function to minimize during training
# input_tensor: the input tensor to the reduce_mean function
# This calculates the average loss

if hps.is_training == 1:
    lr = tf.Variable(hps.learning_rate, trainable=False)
    # lr: the learning rate variable, not trainable
    # This sets the learning rate for the optimizer

    optimizer = tf.train.AdamOptimizer(lr)
    # optimizer: the Adam optimizer with the specified learning rate
    # This creates the Adam optimizer with the learning rate

    gvs = optimizer.compute_gradients(cost)
    # gvs: the gradients and variable pairs
    # cost: the cost function to minimize during training
    # This calculates the gradients of the cost function with respect to the variables

    capped_gvs = [(tf.clip_by_value(grad, -hps.grad_clip,
                   hps.grad_clip), var) for grad, var in gvs]
    # capped_gvs: the clipped gradients and variable pairs
    # we use a for loop to loop over all of the values in the gvs pairs
    # grad: the gradient of the loss function with respect to the variable (this is the first element of the gvs pairs)
    # var: the variable to optimize (this is the second element of the gvs pairs)
    # tf.clip_by_value: clips the values of a tensor to a specified range
    # This clips the gradients to avoid exploding gradients

    train_op = optimizer.apply_gradients(
        capped_gvs, global_step=global_step, name='train_step')
    # train_op: the operation to minimize the cost function during training
    # apply_gradients: applies the gradients to the variables
    # capped_gvs: the clipped gradients and variable pairs
    # global_step: the global step variable to increment during training
    # This applies the gradients to the variables and updates the global step

init = tf.global_variables_initializer()
# init: initializes all the global variables in the TensorFlow graph
# This initializes all the variables in the TensorFlow graph
