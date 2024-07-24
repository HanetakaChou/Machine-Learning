import sys
import os
import numpy
import matplotlib.pyplot
# import matplotlib

# Learning Rate
alpha = 1e-3

iteration_count = 2000

# Regularization Parameter
# lambda_ = 0.5

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
    1.0, 
    0.0,
    0.0,
    1.0
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
    0.0
    ])

previous_cost = sys.float_info.max

cost_history = numpy.zeros(iteration_count) 

for t in range(iteration_count):
    linear_hyperplane = numpy.dot(X, theta)
    prediction = 1.0 / (1.0 + numpy.exp(-linear_hyperplane))
    cross_entropy_loss_cost = -(1.0 / float(m)) * (numpy.dot(y, numpy.log(prediction)) + numpy.dot((1.0 - y), numpy.log(1.0 - prediction)))
    
    # l2_regularization_cost = (lambda_ / (2.0 * float(m))) * numpy.dot(theta[1:], theta[1:])
    current_cost = cross_entropy_loss_cost # + l2_regularization_cost
    cost_history[t] = current_cost
    if previous_cost < current_cost :
        print("Learning rate may be too high.")
        break
    elif (previous_cost - current_cost) < gamma:
        print("Convergence achieved after", t + 1, "iterations.")
        break
    previous_cost = current_cost
    
    error = numpy.subtract(prediction, y)
    cross_entropy_loss_gradient = numpy.multiply(1.0 / float(m), numpy.dot(numpy.transpose(X), error))
    # l2_regularization_gradient = (lambda_ / float(m)) * numpy.r_[[0], theta[1:]]
    gradient = cross_entropy_loss_gradient # + l2_regularization_gradient
    
    theta_change = numpy.multiply(alpha, gradient)
    theta = numpy.subtract(theta, theta_change)

# print(matplotlib.rcsetup.all_backends)

matplotlib.pyplot.clf()
matplotlib.pyplot.plot(cost_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost-history.png"))

matplotlib.pyplot.clf()
matplotlib.pyplot.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color="red")
matplotlib.pyplot.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color="blue")
linear_hyperplane_x = [numpy.min(X[:, 1] - 1), numpy.max(X[:, 1] + 1)]
linear_hyperplane_y = -(theta[0] + numpy.dot(theta[1], linear_hyperplane_x)) / theta[2]
matplotlib.pyplot.plot(linear_hyperplane_x, linear_hyperplane_y, color="green")
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision-boundary.png"))

print("Logistic Regression Coefficients:", theta)
