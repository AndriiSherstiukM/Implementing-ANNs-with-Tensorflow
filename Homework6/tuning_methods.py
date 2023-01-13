import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import datasets, layers, models
from bayes_opt import  BayesianOptimization
from functools import partial

import kerastuner
from kerastuner.tuners import Hyperband, RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from hyperopt import hp
# hyperparameter optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

def build_hyper_model(hp):
    model = keras.Sequential()
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(
        filters = hp.Int('Conv_1_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('Conv_1_kernel', values=[3,5]),
        padding='same',
        activation='relu',
        input_shape=(32,32,1)
    ))
    model.add(layers.Conv2D(
        filters = hp.Int('Conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('Conv_2_kernel', values=[3,5]),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(
        filters = hp.Int('Conv_3_filter', min_value=64, max_value=128, step=16),
        kernel_size=hp.Choice('Conv_3_kernel', values=[3,5]),
        padding='same',
        activation='relu'
    ))
    model.add(layers.Conv2D(
        filters = hp.Int('Conv_4_filter', min_value=64, max_value=128, step=16),
        kernel_size=hp.Choice('Conv_4_kernel', values=[3,5]),
        activation='relu'
    ))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(keras.optimizers.Adam(
        hp.Choice('learning_rate', values=[1e-2,1e-3,1e-4])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

# build keras classifier
def build_randomize_cls(unit, lr):
    model = keras.Sequential()
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        input_shape=(32,32,1)
    ))
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu'
    ))
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu'
    ))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=unit, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model 

def build_best_param_Model(best_params):
    dense_layer_units = best_params['units']
    lr = best_params['lr']
    Conv2d_filters = best_params['Conv2d_filters']
    No_of_CONV_and_Maxpool_layers = best_params['No_of_CONV_and_Maxpool_layers']

    model = keras.Sequential()

    for item in range(No_of_CONV_and_Maxpool_layers):
        model.add(layers.Conv2D(
            Conv2d_filters,
            kernel_size=3,
            padding='same',
            activation='relu',
        ))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=dense_layer_units, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

def bayes_optimisation_tuning(train_loop, input_shqpe):
    verbose = 1
    fit_with_partial = partial(train_loop, input_shqpe, verbose)

    # Bounded region of parameter space

    pbounds = {
                'dropout_rate': [0.1,0.25],
                'lr': [1e-4,1e-3,1e-2],
                'batch_size': [16,32],
                'nb_epochs': [10,15,20]
            }

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1
    )

    optimizer.maximize(init_points=10, n_iter=10)

    for i, res in enumerate(optimizer.res):
        print(f'Iteration {i}: \n\t {res}')

    print(optimizer.max)

def random_search_sklearn_tuning(X_train, Y_train, X_test, Y_test):
    params = {
                'batch_size': [16,32],
                'nb_epochs': [10,15,20],
                'units': [5,6,10],
                'No_of_CONV_and_Maxpool_layers':[2,3,4],
                'lr': [1e-4, 1e-3, 1e-2],
                'Conv2d_filters': [32,64]
            }

    # model class to use in the scikit random search CV
    grid = RandomizedSearchCV(
        estimator=build_randomize_cls,
        cv = KFold(3),
        param_distributions=params)

    grid_result = grid.fit(X_train, Y_train)
    best_params = grid_result.best_params_

    print(best_params, '\n')

    y=grid_result.predict(X_test)
    random = accuracy_score(y, Y_test)
    print(f"Base Accuracy {random}")

    best_random = grid_result.best_estimator_
    y1 = best_random.predict(X_test)
    best=accuracy_score(y1, Y_test)
    print(f"Best Accuracy {best}")    

    print(f"Improvement of {'0.3f' % (100 * (best - random) / random) }%")

    best_param_model = build_best_param_Model(best_params)
    history = best_param_model.fit(X_train, Y_train, epochs=15, validation_data=(X_test,Y_test))
    
    # Plot training & validation accuracy values 

    # Plot training & test accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['train_acc'])

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train accuracy', 'Test accuracy'], loc='upper left')
    plt.show()

    # Plot training & test loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['train_loss'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Test loss'], loc='upper left')
    plt.show()


"""
def hyperband_keras_tuning(model, X_train, Y_train):
    
    tuner = Hyperband(
        hypermodel=model,
        objective='val_accuracy',
        max_epochs = 15,
        factor=3,
        hyperband_iterations=3,
        directory='kt_dir',
        project_name='kt_hyperband'
    )

    # Display search space summary
    tuner.search_space_summary()

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, Y_train, epochs=15, validation_split=0.2, callbacks=[stop_early], verbose=2)
    # Get the optimal hyperparameters from the results
    best_hps = tuner.get_best_hyperparameters()[0]

    h_model = tuner.hypermodel.build(best_hps)

"""
def random_search_keras_tuning(X_train, Y_train, epochs=15, validation_split=0.2):

    tuner = RandomSearch(
        hypermodel=build_hyper_model,
        objective='val_accuracy',
        max_trials=5,
        directory='kt_dir',
        project_name='kt_random_search'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, Y_train, epochs=15, validation_split=0.2, callbacks=stop_early)

    model = tuner.get_best_models(num_models=1)[0]
    model.summary()

    history = model.fit(X_train, Y_train, epochs=15, validation_split=0.2, initial_epoch=0)
    model.save('kt_random_search_model')

    # Plot training & test accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['train_acc'])

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train accuracy', 'Test accuracy'], loc='upper left')
    plt.show()

    # Plot training & test loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['train_loss'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train loss', 'Test loss'], loc='upper left')
    plt.show()