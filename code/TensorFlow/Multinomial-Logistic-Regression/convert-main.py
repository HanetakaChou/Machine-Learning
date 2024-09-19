import os
import numpy
import tensorflow

# Data
m = 10000
n = 28*28

(_, _), (X_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

# Normalization
X_test = X_test.astype(numpy.float32) / 255.0

# Flatten
X_test = numpy.array([image.flatten() for image in X_test])
assert X_test.shape[0] == m
assert X_test.shape[1] == n

assert y_test.shape[0] == m

# Serialization
for i in range(7):
    file_X_test_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"MNIST_X_test_{i}.inl"), 'w')
    file_X_test_text.write(", ".join([f"{pixel:.8f}F" for pixel in X_test[i]]))
    file_X_test_text.close() 
    file_y_test_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"MNIST_y_test_{i}.inl"), 'w')
    file_y_test_text.write(f"{y_test[i]:d}")
    file_y_test_text.close()

# Model
keras_model = tensorflow.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Multinomial-Logistic-Regression.keras"))

# Inference
logit_prediction = keras_model.predict(X_test)

prediction = numpy.argmax(logit_prediction, axis=1)

classification_error = numpy.sum(prediction != y_test) / m

print("Classification Error:", classification_error)

# Convert
tflite_model = tensorflow.lite.TFLiteConverter.from_keras_model(keras_model).convert()

# Serialization
file_tflite_model_binary = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Multinomial-Logistic-Regression.tflite"), 'wb')
file_tflite_model_binary.write(tflite_model)
file_tflite_model_binary.close()

file_tflite_model_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Multinomial-Logistic-Regression.inl"), 'w')
file_tflite_model_text.write(', '.join([f"0X{ubyte:02X}" for ubyte in tflite_model]))
file_tflite_model_text.close()
