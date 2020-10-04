# Useful math helper functions

import numpy as np

# Discretize a 2-dimensional domain into a matrix
def discretize(x1_range, nx1, x2_range, nx2):
    matrix = []

    for x1 in np.linspace(*x1_range, nx1):
        row = []

        for x2 in np.linspace(*x2_range, nx2):
            row.append([x1, x2])

        matrix.append(row)

    return matrix


# Break a matrix into equally sized "chunks"
def breakup(matrix, resize_dimensions):
    dimensions = (len(matrix), len(matrix[0]))

    parts = []
    try:
        x1_size = int(dimensions[0]/resize_dimensions[0])
        x2_size = int(dimensions[1]/resize_dimensions[1])

        for i1 in range(resize_dimensions[0]):
            for i2 in range(resize_dimensions[1]):
                part = []

                for j in range(i1*x1_size, (i1 + 1)*x1_size):
                    row = []

                    for k in range(i2*x2_size, (i2 + 1)*x2_size):
                        row.append(matrix[j][k])

                    part.append(row)

                parts.append(part)

    except ValueError:
        return None

    return parts
