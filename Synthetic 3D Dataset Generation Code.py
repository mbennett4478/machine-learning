import matplotlib.pyplot as plt
import numpy as np


# The following binary classification dataset is created by drawing data from two 3D Gaussians
# Each data point has 3 dimensions


# Mean and covariance for Class 0
mean0 = [0, 0, 0]
cov0 = [[2550, 2000, 1500], [2000, 1500, 1200], [1500, 1200, 1900]]  


# Number of datapoints for class 0
m0 = 100


# Generate class 0 data points from a multivariate (3D) Gaussian distribution
#    Here x0_1, x0_2 and x0_3 are 3 dimensions for each data (feature) point

x0_1, x0_2, x0_3 = np.random.multivariate_normal(mean0, cov0, m0).T


# Concatenate the 3 dimensions of each feature to create the data matrix for class 0 
X0 = np.concatenate((x0_1.reshape(-1, 1), x0_2.reshape(-1, 1), x0_3.reshape(-1, 1)), axis=1)

# Create the target vector for class 0 (target is coded with zero)
X0_target = np.zeros((m0,), dtype=np.int).reshape(-1, 1)



# Mean and covariance for Class 1
mean1 = [3, 3, 3]
cov1 = [[2550, 2000, 1500], [2000, 1500, 1200], [1500, 1200, 1900]] 

# Number of datapoints for class 1
m1 = 100


# Generate class 1 data points from a multivariate (3D) Gaussian distribution
#    Here x1_1, x1_2 and x1_3 are 2 dimensions for each data (feature) point
x1_1, x1_2, x1_3 = np.random.multivariate_normal(mean1, cov1, m1).T

# Concatenate the 3 dimensions of each feature to create the data matrix for class 1
X1 = np.concatenate((x1_1.reshape(-1, 1), x1_2.reshape(-1, 1), x1_3.reshape(-1, 1)), axis=1)

# Create the target vector for class 1 (target is coded with one)
X1_target = np.ones((m1,), dtype=np.int).reshape(-1, 1)


#  Class 0 and 1 data are combined to create a single data matrix X
X = np.append(X0, X1, axis=0)

# Target values for class 0 & 1 are combined to create a single target vector
y = np.concatenate((X0_target, X1_target), axis=0)

