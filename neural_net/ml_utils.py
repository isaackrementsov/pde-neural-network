import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import time

import neural_net.math_utils as math


class PDENet:

    # Initialize neural network
    def __init__(self, input, learning_rate, L, v, d_x_boundary, d_t_boundary, n_t_boundary):
        # Conv2D model that takes in a domain matrix and returns a hypothesis value for each point on the domain
        self.model = keras.Sequential([
            layers.Conv2D(32, (4,4), input_shape=input.shape[1:]),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2DTranspose(3, (4,4), strides=(2,2)),
            layers.Conv2DTranspose(2, (4,4), strides=(2,2)),
            layers.Conv2DTranspose(1, (6,6), strides=(2,2))
        ])
        # Use the Adam optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)
        self.input = input

        # Get an example domain matrix
        D = input[0]
        # Change in x and t in input matrices
        self.dt = D[1][0][0] - D[0][0][0]
        self.dx = D[0][1][1] - D[0][0][1]

        # Wave equation parameters
        self.L = L
        self.v = v

        # Boundary conditions (Dirichlet and Neumann)
        self.d_x_boundary = d_x_boundary
        self.d_t_boundary = d_t_boundary
        self.n_t_boundary = n_t_boundary

        # Initialize loss variable for metrics
        self.loss = 0


    def train(self, epochs, socket, database):
        steps_per_epoch = 100

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                self.train_step()

            # Report and save progress after each epoch
            cursor = database.cursor()

            insert_statement = 'INSERT INTO `epochs` (`num`, `total`, `loss`) VALUES (%s, %s, %s)'
            insert_data = (epoch + 1, epochs, self.loss)
            cursor.execute(insert_statement, insert_data)

            database.commit()
            cursor.close()

            try:
                socket.emit('data', {'num': epoch + 1, 'total': epochs, 'loss': self.loss})
            except Exception:
                pass


    # Calculate the loss function and backpropagate the gradient
    @tf.function()
    def train_step(self):
        input = self.input

        with tf.GradientTape() as tape:
            # Get the model output and calculate loss
            output = self.model(input)
            self.loss = self.pde_loss(output)

        # Apply the gradient weight to the model
        grad = tape.gradient(loss, self.model)
        self.optimizer.apply_gradients([(grad, self.model)])


    # Find the loss from a given PDE output
    def pde_loss(self, H):
        scalar_loss = 0
        input = self.input

        # Loop though each hypothesis value and input domain
        for k in range(len(input)):
            D = input[k]
            H = input[k]

            for i in range(len(H)):
                for j in range(len(H[i])):
                    loss = 0

                    # Hypothesis function values
                    h = H[i][j]
                    h_xx = 0
                    h_tt = 0

                    # Approximate second partial derivative wrt x, including on edges
                    if i == 0:
                        # Forward difference
                        h_xx = math.d2(d=self.dx, u=h, u_b=H[i + 1][j], u_b2=H[i + 2][j])
                    elif i == len(H) - 1:
                        # Backwards difference
                        h_xx = math.d2(d=self.dx, u=h, u_a=H[i - 1][j], u_a2=H[i - 2][j])
                    else:
                        h_xx = math.d2(d=self.dx, u=h, u_a=H[i - 1][j], u_b=H[i + 1][j])

                    # Approximate second partial derivative wrt t, including on edges
                    if j == 0:
                        # Forward difference
                        h_tt = math.d2(d=self.dt, u=h, u_b=H[i][j + 1], u_b2=H[i][j + 2])
                    elif j == len(H[i]) - 1:
                        # Backwards difference
                        h_tt = math.d2(d=self.dt, u=h, u_a=H[i][j - 1], u_a2=H[i][j - 2])
                    else:
                        h_tt = math.d2(d=self.dt, u=h, u_a=H[i][j - 1], u_b=H[i][j + 1])

                    # Calculate deviation from wave equation (should equal 0)
                    wave_dev = h_tt - self.v**2*h_xx
                    loss += wave_dev**2

                    # Input values
                    x = D[i][j][0]
                    t = D[i][j][1]

                    # Boundary loss

                    if x == 0 or x == self.L:
                        # Add x boundary condition loss
                        loss += (h - self.d_x_boundary)**2

                    if t == 0:
                        # Add t boundary condition loss
                        loss += (h - self.d_t_boundary(x))**2
                        loss += (h - self.n_t_boundary(x))**2

                    scalar_loss += loss

        return scalar_loss
