import sys
import os
import numpy
import tensorflow
import matplotlib.pyplot

# Learning Rate
alpha = 1e-3

iteration_count = 2000

# Cost Change Threshold
gamma = 1e-4

m = 4

X = numpy.array([
    [10.0, 52.0],
    [2.0, 73.0],
    [5.0, 55.0],
    [12.0, 49.0]
])
assert X.shape[0] == m

y = numpy.array([
    1, 
    0,
    0,
    1
])
assert y.shape[0] == m

# Feature Scaling
Mu_X = numpy.mean(X, axis=0)
Sigma_X = numpy.std(X, axis=0)
X = (X - Mu_X) / Sigma_X

# Intercept Term
X = numpy.hstack([numpy.ones((m, 1)), X])

# Model
theta = tensorflow.Variable([0.0, 0.0, 0.0], dtype=tensorflow.float64)

previous_cost = sys.float_info.max

cost_history = numpy.zeros(iteration_count) 

optimizer = tensorflow.keras.optimizers.SGD(learning_rate=alpha)

# Gradient Descent Loop
for t in range(iteration_count):
    with tensorflow.GradientTape() as tape:
        logit_prediction = tensorflow.squeeze(tensorflow.matmul(X, tensorflow.expand_dims(theta, axis=-1)))
        current_cost = tensorflow.keras.losses.binary_crossentropy(y, logit_prediction, from_logits=True)
    
    cost_history[t] = current_cost.numpy()
    if previous_cost < current_cost.numpy() :
        print("Learning rate may be too high.")
        break
    elif (previous_cost - current_cost.numpy()) < gamma:
        print("Convergence achieved after", t + 1, "iterations.")
        break
    previous_cost = current_cost.numpy()

    # use the gradient tape to calculate the gradient
    gradient = tape.gradient(current_cost, theta)

    optimizer.apply_gradients(zip(gradient, theta))

matplotlib.pyplot.clf()
matplotlib.pyplot.plot(cost_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost-history.png"))

print("Logistic Regression Coefficients:", theta.numpy())
