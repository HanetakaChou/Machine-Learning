import os
import numpy
import tensorflow.lite

# Data
m = 4

X = numpy.array([
    [10.0, 52.0],
    [2.0, 73.0],
    [5.0, 55.0],
    [12.0, 49.0]
], dtype=numpy.float32)
assert X.shape[0] == m

y = numpy.array([
    1, 
    0,
    0,
    1
])
assert y.shape[0] == m

# Model
tflite_interpreter = tensorflow.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistic-regression.tflite"))

tflite_interpreter.allocate_tensors()

tflite_input_details = tflite_interpreter.get_input_details()
tflite_output_details = tflite_interpreter.get_output_details()

# tflite_tensor_details = tflite_interpreter.get_tensor_details()
# for tflite_tensor_detail in tflite_tensor_details:
#    tflite_tensor_index = tflite_tensor_detail["index"]
#    try:
#        tflite_tensor_weights = tflite_interpreter.get_tensor(tflite_tensor_index)
#    except Exception:
#        tflite_tensor_weights = "N/A"
#    print("Tensor Name:", tflite_tensor_detail["name"], "Tensor Index:", tflite_tensor_detail["index"], tflite_tensor_index, "Tensor Shape:", tflite_tensor_detail['shape'], "Tensor Weights:", tflite_tensor_weights)

# Inference
for i in range(m):
    tflite_interpreter.set_tensor(tflite_input_details[0]["index"], X[i:i+1])
    tflite_interpreter.invoke()   
    logit_prediction = tflite_interpreter.get_tensor(tflite_output_details[0]["index"])
    prediction = (logit_prediction > 0.0).astype(int)
    print("Prediction:", prediction, "Target:", y[i])
