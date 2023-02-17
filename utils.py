import tensorflow as tf
import tensorflow_datasets as tfds

import config

def preprocess_image(data: dict):
    """Resizes and crops the images to the dimensions in the config."""

    # center crop image
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    image = tf.image.resize(
        image, 
        size=[
            config.IMAGE_SIZE, 
            config.IMAGE_SIZE
        ], antialias=True
    )

    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(split):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimations
    return (
        tfds.load(
            config.DATASET_NAME, 
            split=split, 
            shuffle_files=True)
        .map(
            preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(config.DATASET_REPS)
        .shuffle(10 * config.BATCH_SIZE)
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
