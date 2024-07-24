import sys
import os
import numpy
import matplotlib.pyplot
# import matplotlib

# Learning Rate
alpha = 1e-3

iteration_count = 15000

# Cost Change Threshold
gamma = 1e-4

m = 4

X = numpy.array([
    [2104.0, 5.0, 1.0, 45.0],
    [1416.0, 3.0, 2.0, 40.0],
    [1534.0, 3.0, 2.0, 30.0],
    [852.0, 2.0, 1.0, 36.0]
    ])
assert X.shape[0] == m

y = numpy.array([
    400.0, 
    232.0,
    315.0,
    178.0
    ])
assert y.shape[0] == m

# Feature Scaling
Mu_X = numpy.mean(X, axis=0)
Sigma_X = numpy.std(X, axis=0)
X = (X - Mu_X) / Sigma_X

# Intercept Term
X = numpy.hstack([numpy.ones((m, 1)), X])

theta = numpy.array([
    0.0, 
    0.0,
    0.0,
    0.0,
    0.0
    ])

previous_cost = sys.float_info.max

cost_history = numpy.zeros(iteration_count) 

for t in range(iteration_count):
    prediction = numpy.dot(X, theta)
    error = numpy.subtract(prediction, y)
    current_cost = (0.5 / float(m)) * numpy.dot(error, error)
    
    cost_history[t] = current_cost
    if previous_cost < current_cost :
        print("Learning rate may be too high.")
        break
    elif (previous_cost - current_cost) < gamma:
        print("Convergence achieved after", t + 1, "iterations.")
        break
    previous_cost = current_cost
    
    gradient = numpy.multiply(1.0 / float(m), numpy.dot(numpy.transpose(X), error))
    
    theta_change = numpy.multiply(alpha, gradient)
    theta = numpy.subtract(theta, theta_change)

# print(matplotlib.rcsetup.all_backends)
matplotlib.pyplot.clf()
matplotlib.pyplot.plot(cost_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost-history.png"))

print("Linear Regression Coefficients:", theta)
