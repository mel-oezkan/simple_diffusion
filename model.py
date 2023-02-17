import math
import tensorflow as tf
from tensorflow import keras
from keras import layers

import run_config

def sinusoidal_embedding(x):

    EMBEDDING_MIN_FREQUENCY = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(EMBEDDING_MIN_FREQUENCY),
            tf.math.log(run_config.EMBEDDING_MAX_FREQUENCY),
            run_config.EMBEDDING_DIMS // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width: int):
    """Creates a block with residual connection based on the given width value.

    Args:
        width (int): with of the residual block.
    """

    # create the anonymous function based on the given width
    def apply(x):

        # checks if skip connection or new input
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)

        # apply the residual block layers
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", 
            activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width: int, block_depth: int):
    """Creates a down block based on the given width value.
    
    Args:
        width (int): with of the down block.
        block_depth (int): number of residual blocks in the down block.
    """

    # create the anonymous function based on the given width
    def apply(x):
        x, skips = x

        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)

        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width: int, block_depth: int):

    # create the anonymous function based on the given width
    def apply(x):
        """Takes the input of previous layer and the skip connections.

        Args:
            x (tuple): output of prev. layer and skip connection.

        Returns:
            Any: outptu of the up block.
        """
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)

        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)

        return x

    return apply


def get_network(image_size, widths, block_depth):
    """Create the u-net model based on the given parameters.

    Args:
        image_size (int): shape of the square input images.
        widths (int): widht of the images.
        block_depth (int): depth of the residual blocks.

    Returns:
        tf.keras.Model: Keras model of the u-net.
    """

    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    # apply the embeddings on the noise variances
    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # process image and concat with embeddings
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    # final convolution to get the output
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    # return the sequential model
    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")