# Simulate an inhomogeneous Poisson point process on a rectangle

import numpy as np;  
import matplotlib.pyplot as plt 
from scipy.optimize import minimize  
from scipy import integrate  


# Simulate window and get area
x_min=-1
x_max=1
y_min=-1
y_max=1
x_delta = x_max-x_min
y_delta = y_max-y_min # rectangle dimensions
area = x_delta*y_delta
 

n_sim = 10 ** 3  # number of simulations

s = 0.5  # scale parameter


# Point process parameters
def fun_lambda(x, y):
    return 100 * np.exp(-(x ** 2 + y ** 2) / s ** 2)  # intensity function -> driving intensity Λ


############################################ FIND MAX LAMBDA ############################################
# For an intensity function lambda, given by function fun_lambda,
# finds the maximum of lambda in a rectangular region given by above parameters
def fun_Neg(x):
    return -fun_lambda(x[0], x[1])  # negative of lambda


xy0 = [(x_min + x_max) / 2, (y_min + y_max) / 2]  # initial value(ie centre)
# Find largest lambda value
resultsOpt = minimize(fun_Neg, xy0, bounds=((x_min, x_max), (y_min, y_max)))
lambda_neg_min = resultsOpt.fun  # retrieve minimum value found by minimize
lambda_max = -lambda_neg_min


############################################

# define thinning probability function
def fun_p(x, y):
    return fun_lambda(x, y) / lambda_max

# for collecting statistics -- set numbSim=1 for one simulation
n_kept = np.zeros(n_sim)  # vector to record number of points
for ii in range(n_sim):
    # Simulate a Poisson point process
    n_points = np.random.poisson(area * lambda_max)  # Poisson number of points
    xx = np.random.uniform(0, x_delta, ((n_points, 1))) + x_min  # x coordinates of Poisson points
    yy = np.random.uniform(0, y_delta, ((n_points, 1))) + y_min  # y coordinates of Poisson points

    # calculate spatially-dependent thinning probabilities
    p = fun_p(xx, yy)

    # Generate Bernoulli variables for thinning
    boolean_kept = np.random.uniform(0, 1, ((n_points, 1))) < p;  # points to be retained

    # x/y locations of retained points
    xx_kept = xx[boolean_kept]
    yy_kept = yy[boolean_kept]
    n_kept[ii] = xx_kept.size 

# Plotting
plt.scatter(xx_kept, yy_kept, edgecolor='b', facecolor='none', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')

# run empirical test on number of points generated
if n_sim >= 10:
    # total mean measure (average number of points)
    LambdaNumerical = integrate.dblquad(fun_lambda, xMin, xMax, lambda x: yMin, lambda y: yMax)[0]
    # Test: as numbSim increases, numbPointsMean converges to LambdaNumerical
    numbPointsMean = np.mean(n_kept)
    # Test: as numbSim increases, numbPointsVar converges to LambdaNumerical
    numbPointsVar = np.var(n_kept)