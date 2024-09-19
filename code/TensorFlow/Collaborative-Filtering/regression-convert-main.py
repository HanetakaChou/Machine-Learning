import os
import numpy
import tensorflow

# Data
data_user_ids = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=numpy.int32)
data_item_ids = numpy.array([0, 1, 3, 4, 0, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3], dtype=numpy.int32)

# Model
keras_model = tensorflow.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression-based-collaborative-filtering.keras"))

# Inference
prediction = keras_model.predict([data_user_ids, data_item_ids])

print("Prediction:", prediction)

# Convert
tflite_model = tensorflow.lite.TFLiteConverter.from_keras_model(keras_model).convert()

# Serialization
file_tflite_model_binary = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression-based-collaborative-filtering.tflite"), 'wb')
file_tflite_model_binary.write(tflite_model)
file_tflite_model_binary.close()

file_tflite_model_text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression-based-collaborative-filtering.inl"), 'w')
file_tflite_model_text.write(", ".join([f"0X{ubyte:02X}" for ubyte in tflite_model]))
file_tflite_model_text.close()
