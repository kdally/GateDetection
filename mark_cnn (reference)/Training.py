from mask_cnn import DataTransform
from DataProperties import DataProperties
from Networks import Model
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

data = DataProperties()

model = Model('preTrained')
unet = model.net
unet.summary()
# model.visualize()

print(len(model.net.layers))

unet.compile(loss="mean_squared_error", optimizer='adam', metrics=["accuracy"])

train_gen, val_gen = DataTransform.augment(data)

checkpoint = tf.keras.callbacks.ModelCheckpoint(model.save_name, monitor='loss', verbose=1,
                                                save_best_only=True, mode='auto', period=1)

history = unet.fit_generator(
    train_gen,
    steps_per_epoch=50,
    epochs=5,
    validation_data=val_gen,
    validation_steps=10,
    callbacks=[checkpoint]
)

unet.save(model.save_name)

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
