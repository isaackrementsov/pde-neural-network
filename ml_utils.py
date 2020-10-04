from tensorflow import keras

# Initialize neural network
def get_model(input_shape):
    model = keras.Sequential([
        layers.Conv2D(64, 3, 3, input_shape=shape[1:]),
        layers.Flatten(),
        layers.Dense(2, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(3, activation='relu'),
        layers.Dense(1)
    ])

    return model


def train_model():
    model = get_model()

    model.compile(
        optimizer='adam',
        loss=pde_loss,
        metrics=['mae']
    )

    print('training....')


# Loss function for PDE accuracy
def pde_loss(d, h):
    tf.print(type(d), type(h))
