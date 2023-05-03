import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
from pointpats.geometry import(TREE_TYPES)
import numpy as np
from pointpats import PointPattern
from pointpats import PoissonPointProcess, PoissonClusterPointProcess, Window, poly_from_bbox, PointPattern
from pointpats import PoissonPointProcess as csr
from scipy import stats
from scipy.optimize import minimize  
import math
from multiprocessing import set_start_method
from multiprocessing import get_context
from multiprocessing import pool 
import pickle


tang = pd.read_pickle('/Users/gocchini/Desktop/paper_3/data/tang_non_sus.pkl')
intang = pd.read_pickle('/Users/gocchini/Desktop/paper_3/data/data_non_sus_10.pkl')

all_industries = pd.concat([tang, intang])

a = all_industries['lat'].to_numpy()
b = all_industries['long'].to_numpy()

x_min = a.min()
x_max = a.max()
y_min = b.min()
y_max = b.max()
x_delta = x_max - x_min
y_delta = y_max - y_min 
area = x_delta*y_delta 

X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([a, b])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

values = np.vstack([a, b])
kernel = stats.gaussian_kde(values)

def fun_lambda(kernel, x, y):
    values = np.vstack([x, y])
    return kernel.pdf(values)

fun_Neg = lambda x: -fun_lambda(kernel, x[0], x[1])

xy0 = [(x_min + x_max) / 2, (y_min + y_max) / 2]  # initial value(ie centre)
# Find largest lambda value
resultsOpt = minimize(fun_Neg, xy0, bounds=((x_min, x_max), (y_min, y_max)))
lambda_neg_min = resultsOpt.fun  # retrieve minimum value found by minimize
lambda_max = -lambda_neg_min

# define thinning probability function
def fun_p(kernel, x, y):
    return fun_lambda(kernel, x, y) / lambda_max

def sampler(n_iterations):
    x_all = []
    y_all = []
    for i in range(len(n_iterations)):
        np.random.seed()
        # Simulate a Poisson point process
        lat = []
        lon = []
        n_accepted = 0
        n_required = len(intang)
        n_sim = len(intang) + 10**4

        while n_accepted < n_required:

            xx = np.random.uniform(0, x_delta, n_sim) + x_min  # x coordinates of Poisson points
            yy = np.random.uniform(0, y_delta, n_sim) + y_min  # y coordinates of Poisson points

            # calculate spatially-dependent thinning probabilities
            p = fun_p(kernel, xx, yy)

            # Generate Bernoulli variables (ie coin flips) for thinning
            booleThinned = np.random.uniform(0, 1, n_sim) < p  # points to be thinned
            booleRetained = ~booleThinned  # points to be retained
            lat.append(xx[booleThinned])
            lon.append(yy[booleThinned])
            n_accepted += sum(booleThinned)
        
        x_all.append(lat)
        y_all.append(lon)
        print('done')

    return pd.DataFrame(zip(x_all, y_all))


def parallel_sampler(n_iterations):
    n_cores = os.cpu_count()
    array = np.array(list(np.arange(0,n_iterations)))
    split_array = np.array_split(array, n_cores)
    with get_context('fork').Pool(processes = n_cores) as pool:
        results = pool.map(sampler, split_array)
    pool.close()
    pool.join()
    return pd.concat(results)

if __name__ == '__main__':

    trial = parallel_sampler(100)
    trial.to_pickle('/Users/gocchini/Desktop/paper_3/kinhom_work/simuls_df.pkl')