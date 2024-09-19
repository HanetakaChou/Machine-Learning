import os
import numpy
import tensorflow
import matplotlib.pyplot

# Learning Rate
alpha = 1e-3

iteration_count = 5000

# Cost Change Threshold
gamma = 1e-4

# Regularization Parameter
lambda_ = 0.0

# Data

# Number of Users
m = 4

# Number of Items
n = 5

# Sparse User-Item Interaction Matrix R
data_user_ids     = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=numpy.int32)
data_item_ids     = numpy.array([0, 1, 3, 4, 0, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3], dtype=numpy.int32)
data_interactions = numpy.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=numpy.float32)

# Model

# Number of Latent Factors
k = 2

# Include the Bias Term
# During Training, We can hopefully learn the Constant Feature(ones) to support the Bias Term
k = k + 1

user_ids = tensorflow.keras.layers.Input(shape=(1,), dtype=tensorflow.int32)
item_ids = tensorflow.keras.layers.Input(shape=(1,), dtype=tensorflow.int32)

# Map User ID to Non-Missing Row of User Feature Matrix U
_U = tensorflow.keras.layers.Embedding(input_dim=m, output_dim=k, embeddings_regularizer=tensorflow.keras.regularizers.L2(l2=lambda_))(user_ids)

# Map Item ID to Non-Missing Row of User Feature Matrix V
_V = tensorflow.keras.layers.Embedding(input_dim=n, output_dim=k, embeddings_regularizer=tensorflow.keras.regularizers.L2(l2=lambda_))(item_ids)

# From (non-missing (<= m), 1, k)  to (non-missing (<= m), k)
U = tensorflow.keras.layers.Flatten()(_U)

# From (non-missing (<= n), 1, k)  to (non-missing (<= n), k)
V = tensorflow.keras.layers.Flatten()(_V)

# We use "from_logits=True" to be more numerical stable
UoV = tensorflow.keras.layers.Dot(axes=1)([U, V])

keras_model = tensorflow.keras.Model(inputs=[user_ids, item_ids], outputs=UoV)

keras_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=alpha), loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True))

# Gradient Descent Loop
keras_history = keras_model.fit([data_user_ids, data_item_ids], data_interactions, epochs=iteration_count, callbacks=[ tensorflow.keras.callbacks.EarlyStopping(monitor='loss', min_delta=gamma, patience=7, restore_best_weights=True)])

cost_history = keras_history.history['loss']

matplotlib.pyplot.clf()
matplotlib.pyplot.plot(cost_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost-history.png"))

# Serialization
keras_model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification-based-collaborative-filtering.keras"), overwrite=True)
