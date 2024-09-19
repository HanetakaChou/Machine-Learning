import numpy

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

XT_mul_X = numpy.dot(numpy.transpose(X), X)

XT_mul_X_inv = numpy.linalg.inv(XT_mul_X)

XT_mul_y = numpy.dot(X.T, y)

theta = numpy.dot(XT_mul_X_inv, XT_mul_y)

print("Linear Regression Coefficients:", theta)
