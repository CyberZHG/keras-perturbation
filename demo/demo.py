import os
import pickle
import keras
import keras.backend as K
import numpy as np
from keras_perturbation import Perturbation, Clip


BATCH_SIZE = 256


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)


def get_model(train_perturbation=False, with_perturbation=False):
    input_layer = keras.layers.Input(shape=input_shape, name='Input')
    if train_perturbation:
        perturbation = Perturbation(
            max_variable_shape=(BATCH_SIZE,),
            constraint=Clip(min_value=0.0, max_value=0.8),
            name='Perturbation',
        )
        last_layer = perturbation(input_layer)
    elif with_perturbation:
        perturbation = keras.layers.Input(shape=input_shape, name='Input-Perturbation')
        last_layer = keras.layers.Add(name='Add')([input_layer, perturbation])
    else:
        last_layer = input_layer
    conv = keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        trainable=not train_perturbation,
        name='Conv-1',
    )(last_layer)
    conv = keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        activation='relu',
        trainable=not train_perturbation,
        name='Conv-2',
    )(conv)
    pool = keras.layers.MaxPool2D(name='Pool')(conv)
    flatten_layer = keras.layers.Flatten(name='Flatten')(pool)
    dense = keras.layers.Dense(
        units=64,
        activation='relu',
        trainable=not train_perturbation,
        name='Dense-1',
    )(flatten_layer)
    dense = keras.layers.Dense(
        units=10,
        activation='softmax',
        trainable=not train_perturbation,
        name='Dense-2',
    )(dense)
    inputs = [input_layer]
    loss = keras.losses.sparse_categorical_crossentropy
    if train_perturbation:
        curr_loss = lambda x, y: -loss(x, y)
    else:
        curr_loss = loss
    if with_perturbation:
        inputs.append(perturbation)
    model = keras.models.Model(inputs=inputs, outputs=dense)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=curr_loss,
        metrics=[keras.metrics.sparse_categorical_accuracy],
    )
    return model


print('Train base model...')
model_path = 'base_model.h5'
model = get_model(train_perturbation=False, with_perturbation=False)
model.summary()
if os.path.exists(model_path):
    model.load_weights(model_path)
else:
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        epochs=30,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
        ],
    )
    model.save_weights(model_path)
before_score = model.evaluate(x_test, y_test, verbose=False)[1]
print('Before perturbation: %.4f' % before_score)

print('Train perturbation...')
model_training_path = 'training_model.h5'
perturbation_path = 'perturbations.pickle'
model = get_model(train_perturbation=True, with_perturbation=False)
model.summary()
model.load_weights(model_path, by_name=True)
model.save_weights(model_training_path)
train_len = int(len(x_train) * 0.9)
if os.path.exists(perturbation_path):
    with open(perturbation_path, 'rb') as reader:
        perturbations = pickle.load(reader)
else:
    perturbations = np.zeros_like(x_train, dtype=K.floatx())
    steps = (train_len + BATCH_SIZE - 1) // BATCH_SIZE
    perturbation_layer = model.get_layer(name='Perturbation')
    for i in range(steps):
        interval = slice(i * BATCH_SIZE, min(train_len, (i + 1) * BATCH_SIZE))
        print('%d/%d' % (interval.stop, train_len), end='\r')
        x_sub = x_train[interval]
        y_sub = y_train[interval]
        model.optimizer.iterations.assign(0)
        model.load_weights(model_training_path, by_name=True)
        model.fit(
            x_sub,
            y_sub,
            shuffle=False,
            batch_size=BATCH_SIZE,
            epochs=1000,
            verbose=False,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='loss', patience=2),
            ],
        )
        perturbations[interval] = perturbation_layer.get_perturbation_values((interval.stop - interval.start,))
    print('')
    with open(perturbation_path, 'wb') as writer:
        pickle.dump(perturbations, writer)

print('Train with perturbation...')
model_with_path = 'with_model.h5'
model = get_model(train_perturbation=False, with_perturbation=True)
model.summary()
if not os.path.exists(model_with_path):
    model.load_weights(model_path, by_name=True)
    x_train = np.concatenate((
        x_train[:train_len],
        x_train[:train_len],
        x_train[train_len:],
        x_train[train_len:],
    ))
    zeros = np.zeros_like(x_train, dtype=K.floatx())
    zeros[:train_len] = perturbations[:train_len]
    perturbations = zeros
    y_train = np.concatenate((
        y_train[:train_len],
        y_train[:train_len],
        y_train[train_len:],
        y_train[train_len:],
    ))
    model.fit(
        [x_train, perturbations],
        y_train,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        epochs=30,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
        ],
    )
    model.save_weights(model_with_path)
model = get_model(train_perturbation=False, with_perturbation=False)
model.summary()
model.load_weights(model_with_path, by_name=True)
after_score = model.evaluate(x_test, y_test, verbose=False)[1]
print('Before perturbation: %.4f' % before_score)
print('After perturbation: %.4f' % after_score)
