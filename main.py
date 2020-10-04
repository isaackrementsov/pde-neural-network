import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

import math_utils as math
import ml_utils as ml

# Boundary length
L = 10
# Time to model wave over
T = 10
# Number of pieces to break t and x into
Nt = T
Nx = Nt
# Number of "pieces" for breaking up the domain (x, t)
Nd = (2, 2)

# Boundary conditions
v = 1
f = lambda x: np.sin(x)
g = lambda x: np.cos(x)

# Input shape
shape = [Nd[0]*Nd[1], int(Nt/Nd[0]), int(Nx/Nd[1]), 2]
# Range of both variables
domain = math.discretize((0,L), Nx, (0,T), Nt)
# Break the domain into pieces
i = tf.convert_to_tensor(math.breakup(domain, Nd))

model.compile(
    optimizer='adam',
    loss=pde_loss,
    metrics=['mae']
)

model(i)
