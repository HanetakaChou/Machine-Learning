import os
import numpy
import tensorflow
import matplotlib.pyplot

# Learning Rate
alpha = 1e-3

iteration_count = 2000

# Cost Change Threshold
gamma = 1e-6

# Data
m = 60000
n = 28*28

(X_train, y_train), (_, _) = tensorflow.keras.datasets.mnist.load_data()

# Normalization
X_train = X_train.astype(numpy.float32) / 255.0

# Flatten
X_train = numpy.array([image.flatten() for image in X_train])
assert X_train.shape[0] == m
assert X_train.shape[1] == n

# Model
keras_model = tensorflow.keras.models.Sequential([
    # Feature Scaling
    tensorflow.keras.layers.Normalization(input_shape=(n,)), 
    # Imitate LeNet-5
    tensorflow.keras.layers.Dense(units=25, activation="relu"),
    tensorflow.keras.layers.Dense(units=15, activation="relu"),
    # Multinomial Logistic Regression (We use "from_logits=True" to be more numerical stable)
    tensorflow.keras.layers.Dense(units=10, activation="linear")
])

keras_model.layers[0].adapt(X_train)

y_train = tensorflow.keras.utils.to_categorical(y_train, 10)

# We can use "tensorflow.keras.optimizers.SGD(learning_rate=alpha)" as well but "Adam" is the improved version
# We can use "sparse_categorical_crossentropy" as well and we can get the index directly  
keras_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=alpha), loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True))

# Gradient Descent Loop
keras_history = keras_model.fit(X_train, y_train, epochs=iteration_count, callbacks=[ tensorflow.keras.callbacks.EarlyStopping(monitor='loss', min_delta=gamma, patience=7, restore_best_weights=True)])

cost_history = keras_history.history['loss']

matplotlib.pyplot.clf()
matplotlib.pyplot.plot(cost_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost-history.png"))

# Serialization
keras_model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Multinomial-Logistic-Regression.keras"), overwrite=True)

