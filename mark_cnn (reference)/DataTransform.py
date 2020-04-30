from keras_preprocessing.image import ImageDataGenerator


def augment(images):
    batch = 32

    data_gen_args = dict(rescale=1. / 255,
                         rotation_range=5,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=False,
                         zoom_range=0.2,
                         validation_split=0.15 / 0.90
                         )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1

    train_image_generator = image_datagen.flow_from_directory(
        f'{images.path}/training/images',
        subset='training',
        class_mode=None,
        seed=seed,
        target_size=(images.height, images.width),
        batch_size=batch
    )

    train_mask_generator = mask_datagen.flow_from_directory(
        f'{images.path}/training/masks',
        subset='training',
        class_mode=None,
        seed=seed,
        target_size=(images.height, images.width),
        color_mode='grayscale',
        batch_size=batch
    )

    val_image_generator = image_datagen.flow_from_directory(
        f'{images.path}/training/images',
        subset='validation',
        class_mode=None,
        seed=seed,
        target_size=(images.height, images.width),
        batch_size=batch
    )

    val_mask_generator = mask_datagen.flow_from_directory(
        f'{images.path}/training/masks',
        subset='validation',
        class_mode=None,
        seed=seed,
        target_size=(images.height, images.width),
        color_mode='grayscale',
        batch_size=batch
    )

    train_generator = (pair for pair in zip(train_image_generator, train_mask_generator))
    val_generator = (pair for pair in zip(val_image_generator, val_mask_generator))

    return train_generator, val_generator
