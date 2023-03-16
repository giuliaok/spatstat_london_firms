import numpy as np 
import matplotlib.pyplot as plt
 
# Simulate window and get area
x_min=-1
x_max=1
y_min=-1
y_max=1
x_delta = x_max-x_min
y_delta = y_max-y_min # rectangle dimensions
area = x_delta*y_delta
 
# Point process parameters
lambda_pp = 100 # intensity (ie mean density) of the Poisson process
 
# Thinning probability
p = 0.25
 
# Simulate a Poisson point process
n_points = np.random.poisson(lambda_pp*area) # Poisson number of points
xx = np.random.uniform(0,x_delta,((n_points,1)))+x_min #x coordinates of Poisson points
yy = np.random.uniform(0,y_delta,((n_points,1)))+y_min #y coordinates of Poisson points
 
# Generate Bernoulli variables for thinning
boolean_thinned = np.random.uniform(0,1,((n_points,1)))>p # points to be thinned
boolean_kept =~ boolean_thinned # points to be retained
 
# x/y locations of thinned points
xx_thinned = xx[boolean_thinned]
yy_thinned = yy[boolean_thinned]
# x/y locations of retained points
xx_kept = xx[boolean_kept]
yy_kept = yy[boolean_kept]
 
# Plotting
plt.scatter(xx_kept,yy_kept, edgecolor='b', facecolor='none', alpha=0.5 )
plt.scatter(xx_thinned,yy_thinned, edgecolor='r', facecolor='none', alpha=0.5 )
plt.xlabel("x"); plt.ylabel("y")
plt.show()