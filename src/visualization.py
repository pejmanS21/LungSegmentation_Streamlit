import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

logger = logging.getLogger("OutputSaver")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('../logs/visualize.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def visualize_output(X, predicted):
    if X.shape[3] == 1: 
        # X.shape = (n, 256, 256, 1)
        output_figure = np.zeros((X.shape[1] * X.shape[0], X.shape[2] * 2, 1))

        for i in range(len(X)):
            output_figure[i * 256: (i + 1) * 256,
            0: 1 * 256] = ((X[i] * 127.) + 127.)
            output_figure[i * 256: (i + 1) * 256,
            1 * 256: 2 * 256] = (predicted[i] * 255)

        fig_shape = np.shape(output_figure)
        output_figure = output_figure.reshape((fig_shape[0], fig_shape[1]))
        cv2.imwrite('../images/output_figure.png', output_figure)
        logger.debug('Output figure saved!')
        return output_figure
    
    elif X.shape[3] == 3:
        X = X[:, :, :, :1]
        # X.shape = (n, 256, 256, 3)
        output_figure = np.zeros((X.shape[1] * X.shape[0], X.shape[2] * 2, 1))

        for i in range(len(X)):
            output_figure[i * 256: (i + 1) * 256,
            0: 1 * 256] = ((X[i] * 127.) + 127.)
            output_figure[i * 256: (i + 1) * 256,
            1 * 256: 2 * 256] = (predicted[i] * 255)

        fig_shape = np.shape(output_figure)
        output_figure = output_figure.reshape((fig_shape[0], fig_shape[1]))
        cv2.imwrite('../images/output_figure.png', output_figure)
        logger.debug('Output figure saved!')
        return output_figure
    


def visualize_vae(decoder, output_number, vae_range):
    dim = 256
    figure = np.zeros((dim * output_number, dim * output_number, 1))

    grid_x = np.linspace(-vae_range, vae_range, output_number)
    grid_y = np.linspace(-vae_range, vae_range, output_number)[::-1]

    # decoder for each square in the grid
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(dim, dim, 1)
            figure[i * dim: (i + 1) * dim,
            j * dim: (j + 1) * dim] = digit

    plt.figure(figsize=(10, 10))
    # Reshape for visualization
    fig_shape = np.shape(figure)
    figure = figure.reshape((fig_shape[0], fig_shape[1]))
    cv2.imwrite('../images/output_vae.png', figure * 255.)
    logger.debug('Output figure saved!')
    return figure
