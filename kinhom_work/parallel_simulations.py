import pandas as pd
from pointpats.geometry import(TREE_TYPES)
import numpy as np
from pointpats import PointPattern
from pointpats import PoissonPointProcess, PoissonClusterPointProcess, Window, poly_from_bbox, PointPattern
import libpysal as ps
from libpysal.cg import shapely_ext
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pointpats.quadrat_statistics as qs
from pointpats import PoissonPointProcess as csr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
from scipy import stats
from scipy.optimize import minimize  
from scipy import integrate 
from multiprocessing import set_start_method
from multiprocessing import get_context
from multiprocessing import pool 


tang = pd.read_pickle('/Users/gocchini/Desktop/paper_3/data/tang_non_sus.pkl')
intang = pd.read_pickle('/Users/gocchini/Desktop/paper_3/data/data_non_sus_10.pkl')

all_industries = pd.concat([tang, intang])

x = all_industries['lat'].to_numpy()
y = all_industries['long'].to_numpy()

x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()
x_delta = x_max - x_min
y_delta = y_max - y_min 
area = x_delta*y_delta 

X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

values = np.vstack([x, y])
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

laty = np.concatenate(x).ravel()
longy = np.concatenate(y).ravel()
df = pd.DataFrame({'laty': [laty], 'longy': [longy]})



def sampler(x, y, kernel, intang):
    
    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()
    x_delta = x_max - x_min
    y_delta = y_max - y_min 
    area = x_delta*y_delta 

    x_all = []
    y_all = []
    for i in range(0):
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

            # x/y locations of thinned points
            xxThinned = xx[booleThinned]
            yyThinned = yy[booleThinned]
            # x/y locations of retained points
            xxRetained = xx[booleRetained]
            yyRetained = yy[booleRetained]
        
        x_all.append(lat)
        y_all.append(lon)
    return x_all, y_all

def parallelize_df(df, function):
    n_cores = os.cpu_count()
    df_split = np.array_split(df, n_cores)
    with get_context('fork').Pool(processes = n_cores) as pool:
        df = pd.concat(pool.map(function, df_split))
        pool.close()
        pool.join()
    return df

if __name__ == '__main__':