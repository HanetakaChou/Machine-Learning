import os
import numpy
import tensorflow
import matplotlib.pyplot

# Learning Rate
alpha = 1e-3

iteration_count = 2000

# Cost Change Threshold
gamma = 1e-4

n = 2

X = numpy.array([
    [10.0, 52.0],
    [2.0, 73.0],
    [5.0, 55.0],
    [12.0, 49.0]
])
assert X.shape[1] == n

y = numpy.array([
    1, 
    0,
    0,
    1
])

keras_model = tensorflow.keras.models.Sequential([
    # Feature Scaling
    tensorflow.keras.layers.Normalization(input_shape=(n,)), 
    # Logistic Regression (We use "from_logits=True" to be more numerical stable)
    tensorflow.keras.layers.Dense(units=1, activation="linear", kernel_initializer="zeros", bias_initializer="zeros")
])

keras_model.layers[0].adapt(X)

keras_normalization_layer_weights = keras_model.layers[0].get_weights() 

Mu_X = keras_normalization_layer_weights[0]
Sigma_2_X = keras_normalization_layer_weights[1]

print("Mu_X:", Mu_X)
print("Sigma_2_X:", Sigma_2_X)

keras_model.compile(optimizer=tensorflow.keras.optimizers.SGD(learning_rate=alpha), loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True))

# Gradient Descent Loop
keras_history = keras_model.fit(X, y, epochs=iteration_count, callbacks=[ tensorflow.keras.callbacks.EarlyStopping(monitor='loss', min_delta=gamma)])

keras_dense_layer_weights = keras_model.layers[1].get_weights() 

cost_history = keras_history.history['loss']

theta = numpy.vstack([keras_dense_layer_weights[1], keras_dense_layer_weights[0]])

matplotlib.pyplot.clf()
matplotlib.pyplot.plot(cost_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost-history.png"))

print("Logistic Regression Coefficients:", theta)

# Serialization
keras_model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistic-regression.keras"), overwrite=True)
