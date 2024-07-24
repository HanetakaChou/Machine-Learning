import os
import numpy
import tensorflow.lite
import matplotlib.pyplot

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

# Model
tflite_interpreter = tensorflow.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Multinomial-Logistic-Regression.tflite"))

tflite_interpreter.allocate_tensors()

tflite_input_details = tflite_interpreter.get_input_details()
tflite_output_details = tflite_interpreter.get_output_details()

# Inference
for i in range(7):
    tflite_interpreter.set_tensor(tflite_input_details[0]["index"], X_test[i:i+1])
    tflite_interpreter.invoke()   
    logit_prediction = tflite_interpreter.get_tensor(tflite_output_details[0]["index"])
    prediction = numpy.argmax(logit_prediction)
    matplotlib.pyplot.clf()
    matplotlib.pyplot.imshow(X_test[i].reshape(28, 28), cmap="gray")
    matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"X_test_{i:d}.png"))
    print("Prediction:", prediction, "Target:", y_test[i])
