import os
import numpy
import tensorflow.lite

# Data
kappa = 15

data_user_ids    = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=numpy.int32)
assert data_user_ids.shape[0] == kappa

data_item_ids    = numpy.array([0, 1, 3, 4, 0, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3], dtype=numpy.int32)
assert data_item_ids.shape[0] == kappa

data_interactions = numpy.array([5, 5, 0, 0, 5, 4, 0, 0, 0, 0, 5, 5, 0, 0, 4])
assert data_interactions.shape[0] == kappa

# Model
tflite_interpreter = tensorflow.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression-based-collaborative-filtering.tflite"))

tflite_interpreter.allocate_tensors()

tflite_input_details = tflite_interpreter.get_input_details()
tflite_output_details = tflite_interpreter.get_output_details()

# Inference
for i in range(kappa):
    tflite_interpreter.set_tensor(tflite_input_details[0]["index"], numpy.array([[data_user_ids[i]]], dtype=numpy.int32))
    tflite_interpreter.set_tensor(tflite_input_details[1]["index"], numpy.array([[data_item_ids[i]]], dtype=numpy.int32))
    tflite_interpreter.invoke()   
    prediction = tflite_interpreter.get_tensor(tflite_output_details[0]["index"])
    print("Prediction:", prediction, "Target:", data_interactions[i])
