import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Dense, Flatten
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt
from keras import datasets
import keras_tuner as kt
from keras_tuner import Hyperband, HyperModel, HyperParameters
#from keras_tuner.tuners import Hyperband
from keras import Input
from sklearn import metrics


def build_model(hp):
    inputs = Input(shape=(32,32,3))
    x = inputs
    for i in range(hp.Int("conv_blocks", min_value=3, max_value=5, step=1, default=3)):
        filters = hp.Int("filters_" + str(i), 32, 256, step=32)
        for _ in range(2):
            x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        if hp.Choice("pooling_" + str(i), ["avg", "max"]) == 'max':
            x =  MaxPooling2D()(x)
        else:
            x = AveragePooling2D()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int("hidden_size", min_value=30, max_value=100, step=10, default=50),
        Activation('relu'))(x)
    x = Dropout(hp.Float("dropout", min_value=0, max_value=0.5, step=0.1, default=0.5))(x)
    outputs = Dense(10, activation='softmax')(x)
    model =Model(inputs, outputs)
    model.compile(
        optimizer = optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss = 'sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    hyperband_iterations=2)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#def standardize_record(record):
#   return tf.cast(record['image'], tf.float32) /255., record['label']

#train_images = train_images
#test_labels = test_labels

print(train_images.shape)
print(test_labels.shape)

#stop training early after reaching a certain value for the validation loss
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

#Run the hyperparameters search
tuner.search(train_images, train_labels, epochs=30, validation_split=0.2,
            callbacks=[stop_early])

#tuner.search(train_images,
#            validation_data = test_labels,
#            epochs=30,
#            callbacks=callbacks.EarlyStopping(patience=1))

        
best_model = tuner.get_best_models(1)[0]

beat_hyperparameters = tuner.get_best_hyperparameters(1)[0]

    


