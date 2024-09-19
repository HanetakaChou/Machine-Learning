
import sys
import os
import numpy
import matplotlib.pyplot

iteration_count = 100

# Cost Change Threshold
gamma = 1e-4

m = 9
n = 2

X = numpy.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
    ])  
assert X.shape[0] == m
assert X.shape[1] == n

K = 3

previous_cost = sys.float_info.max

cost_history = numpy.zeros(iteration_count) 

# K-Means

# random intialization
mu = X[numpy.random.choice(m, size=K, replace=False)]
assert mu.shape[0] == K

# coordinate descent  
for t in range(iteration_count):
    # assignment step 
    squares = numpy.zeros((m, K))
    for k in range(K):
        squares[:, k] = numpy.sum(numpy.square(X - mu[k]), axis=1)
    c = numpy.argmin(squares, axis=1)

    # update step
    mu = numpy.zeros((K, n))
    for i in range(K):
        points = X[c == i]
        if points.shape[0] > 0:
            mu[i] = numpy.mean(points, axis=0)
        else:
            # TODO: eliminate k
            pass
    # WCSS 
    current_cost = numpy.sum(numpy.square(X - mu[c]))
    cost_history[t] = current_cost
    if previous_cost < current_cost :
        print("There is bug!")
        break
    elif (previous_cost - current_cost) < gamma:
        print("Convergence achieved after", t + 1, "iterations.")
        break
    previous_cost = current_cost
    

matplotlib.pyplot.clf()
matplotlib.pyplot.plot(cost_history)
matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost-history.png"))

matplotlib.pyplot.clf()

matplotlib.pyplot.scatter(X[c == 0][:, 0], X[c == 0][:, 1], color="red")
matplotlib.pyplot.scatter(X[c == 1][:, 0], X[c == 1][:, 1], color="blue")
matplotlib.pyplot.scatter(X[c == 2][:, 0], X[c == 2][:, 1], color="green")

matplotlib.pyplot.scatter(mu[0, 0], mu[0, 1], color="red", marker="x")
matplotlib.pyplot.scatter(mu[1, 0], mu[1, 1], color="blue", marker="x")
matplotlib.pyplot.scatter(mu[2, 0], mu[2, 1], color="green", marker="x")

matplotlib.pyplot.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering.png"))
