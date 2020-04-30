import tensorflow as tf
from DataProperties import DataProperties
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, \
    Activation, ZeroPadding2D, Cropping2D


class Model:

    def __init__(self, net_type):
        self.net = unet_pretrained(DataProperties())
        self.name = self.net.name
        self.save_name = self.name + '.h5'

    def visualize(self):
        return tf.keras.utils.plot_model(self.net, show_shapes=True, to_file=self.name + '.png')

    def load(self):
        return tf.keras.models.load_model(self.save_name)


def conv_block(tensor, nfilters, size=3, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters=nfilters,
                               kernel_size=(size, size),
                               padding=padding,
                               kernel_initializer=initializer))

    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result(tensor)


def deconv_block(tensor, residual, nfilters, size=3):
    y = upsample(nfilters, size)(tensor)
    y = tf.keras.layers.concatenate([y, residual], axis=3)
    return y


def unet_pretrained(data_prop):
    new_height = data_prop.height + data_prop.padding
    new_width = data_prop.width + data_prop.padding
    base_model = tf.keras.applications.MobileNetV2(input_shape=[new_height, new_width, 3], include_top=False)

    print('base', len(base_model.layers))
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input,
                                outputs=layers,
                                name="MobileNet_V2"
                                )

    down_stack.trainable = False

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = Input(shape=[data_prop.height, data_prop.width, 3])
    padded_input = ZeroPadding2D(padding=data_prop.padding)(inputs)
    x = padded_input

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # check activation output layer
    output_layer = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3,
                                                   strides=2, padding='same',
                                                   activation='sigmoid'
                                                   )(x)

    cropped_output = Cropping2D(cropping=data_prop.padding)(output_layer)

    return tf.keras.Model(inputs=inputs, outputs=cropped_output, name='unet_preTrained')


def upsample(filters, size):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=True))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=True))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    return result
