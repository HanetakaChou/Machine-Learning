import os
import numpy
import tensorflow

# Data
m = 4

X = numpy.array([
    [10.0, 52.0],
    [2.0, 73.0],
    [5.0, 55.0],
    [12.0, 49.0]
])

y = numpy.array([
    1, 
    0,
    0,
    1
])
assert y.shape[0] == m

# Model
keras_model = tensorflow.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistic-regression.keras"))

keras_normalization_layer_weights = keras_model.layers[0].get_weights() 

Mu_X = keras_normalization_layer_weights[0]
Sigma_2_X = keras_normalization_layer_weights[1]

print("Mu_X:", Mu_X)
print("Sigma_2_X:", Sigma_2_X)

keras_dense_layer_weights = keras_model.layers[1].get_weights() 

theta = numpy.vstack([keras_dense_layer_weights[1], keras_dense_layer_weights[0]])

print("Logistic Regression Coefficients:", theta)

# Inference
logit_prediction = keras_model.predict(X)

prediction = (logit_prediction > 0.0).flatten().astype(int)

classification_error = numpy.sum(prediction != y) / m

print("Classification Error:", classification_error)

# Convert
tflite_model = tensorflow.lite.TFLiteConverter.from_keras_model(keras_model).convert()

# Serialization
file_tflite_model_binary = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistic-regression.tflite"), 'wb')
file_tflite_model_binary.write(tflite_model)
file_tflite_model_binary.close()

file_tflite_model_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistic-regression.inl"), 'w')
file_tflite_model_text.write(", ".join([f"0X{ubyte:02X}" for ubyte in tflite_model]))
file_tflite_model_text.close()
