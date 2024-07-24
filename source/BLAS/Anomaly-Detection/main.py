import numpy

threshold = 1e-4

X = numpy.array([10, 12, 11, 13, 12, 14, 13, 10, 12, 11, 100, 12, 13, 14, 15])

# Train
mu = numpy.mean(X, axis=0)
sigma_2 = numpy.var(X, axis=0)

# Inference
gaussian = (1.0 / (numpy.sqrt(2.0 * numpy.pi * sigma_2))) * numpy.exp(- numpy.square(X - mu) / (2 * sigma_2))

anomalies = numpy.where(gaussian < threshold)

print("Anomalies:", X[anomalies])
