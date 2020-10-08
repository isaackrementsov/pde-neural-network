import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

import neural_net.math_utils as math
import neural_net.ml_utils as ml

# Boundary length
L = 10
# Time to model wave over
T = 10
# Number of pieces to break t and x into
Nt = T*8
Nx = Nt
# Number of "pieces" for breaking up the domain (x, t)
Nd = (2, 2)

# Boundary conditions
v = 1
f = lambda x: tf.sin(x)
g = lambda x: tf.cos(x)

def run(socket, database):
    # Range of both variables
    domain = math.discretize((0,L), Nx, (0,T), Nt)
    # Break the domain into pieces
    i = tf.convert_to_tensor(math.breakup(domain, Nd))

    # Input shape
    shape = i.shape
    # Make sure shape works with the Conv2D network
    assert (shape[1] % 4 == 0 and shape[2] % 4 == 0 and shape[1] >= 40 and shape[2] >= 40), "Input dimensions must be divisible by 4 and greater than 40"

    # Initialize and train the neural network for 10 epochs
    pde_nn = ml.PDENet(i, 0.02, L, v, 0, f, g)
    pde_nn.train(10, socket, database)
