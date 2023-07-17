import pandas as pd
import os 
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy import spatial
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pointpats import PointPattern, as_window
from pointpats import PoissonPointProcess as csr
from pointpats.geometry import (prepare_hull as _prepare_hull, area as _area,
    build_best_tree as _build_best_tree,
    prepare_hull as _prepare_hull,
    TREE_TYPES,
)
from scipy.spatial import distance
from pointpats.ripley import _prepare # very important to prepare data :)

def coords_to_array(df): 
    df['array'] = df.apply(lambda x: np.array(list(zip(x['lat'], x['long']))), axis = 1) 
    return df

def flatten(l):
    return [item for sublist in l for item in sublist]

def kernel_getter(list_of_controls):
    x = []
    for i in list_of_controls:
            x.append(i[0])
    y = []
    for i in list_of_controls:
            y.append(i[1])
    x = np.array(x)
    y = np.array(y)
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    return kernel 

def find_lambda(kernel, x, y):
    values = np.vstack([x, y])
    return kernel.pdf(values)

def fun_find_kinhom(distance, support, multiplied_lambdas):
    kinhom_values = []
    for x in support:
        if x > distance:
            x = 1/multiplied_lambdas
        else: 
            x = 0 
        kinhom_values.append(x)
    return np.array(kinhom_values)

def find_lambdas_and_distances(array_of_simulated_coordinates, support, kernel):
    all_pairs = [(a, b) for idx, a in enumerate(array_of_simulated_coordinates) for b in array_of_simulated_coordinates[idx + 1:]]
    print(len(all_pairs))
    df = pd.DataFrame(all_pairs)
    df.columns = [['x', 'y']]
    df['lambda_values_x'] = df.apply(lambda x: find_lambda(kernel, x['x'][0], x['x'][1]), axis = 1) 
    print("done")
    df['lambda_values_y'] = df.apply(lambda x: find_lambda(kernel, x['y'][0], x['y'][1]), axis = 1)
    print("done")
    df['inhom_lambda'] = df.apply(lambda x: x['lambda_values_x']*x['lambda_values_y'], axis =1) 
    print("done")
    df['distance'] = df.apply(lambda x: distance.euclidean(x['x'], x['y']), axis = 1)
    print("done")
    support_names = map(str, support)
    support_pd = pd.DataFrame(zip(support_names, support),
               columns =['distance_name', 'distance'])
    support_pd = support_pd.T
    support_pd.columns = support_pd.iloc[0]
    support_pd = support_pd.iloc[1: , :]
    support_pd = pd.DataFrame(np.repeat(support_pd.values, len(df), axis = 0))
    support_pd.columns = support_pd.iloc[0]
    support_pd['support'] = support_pd.values.tolist()
    df = pd.concat([df, support_pd['support']], axis=1)
    df.columns = [['x', 'y', 'lambda_values_x','lambda_values_y', 'multiplied_lambdas', 'distance', 'support']]
    df['kinhom'] = df.apply(lambda x: fun_find_kinhom(x['distance'], x['support'], x['multiplied_lambdas']), axis = 1)
    print("done")
    n_pairs_less_than_d = df['kinhom'].values.tolist()
    internal_argument = np.sum(n_pairs_less_than_d, 0)
    #return df, n_pairs_less_than_d, internal_argument
    return internal_argument

def L_function(list_of_estimates):
    L = [math.sqrt(x/math.pi) for x in list_of_estimates]
    return L

if __name__ == '__main__':

    #data 
    controls = pd.concat([pd.read_pickle(r'/Users/gocchini/Desktop/paper_3/kinhom_work/controls/'+x) for x in os.listdir('/Users/gocchini/Desktop/paper_3/kinhom_work/controls')])
    controls.columns = [['lat', 'long']]
    controls[['lat', 'long']] = controls[['lat', 'long']].apply(flatten)
    controls = coords_to_array(controls)
    cases = pd.concat([pd.read_pickle(r'/Users/gocchini/Desktop/paper_3/kinhom_work/cases/'+x) for x in os.listdir('/Users/gocchini/Desktop/paper_3/kinhom_work/cases')])
    cases.columns = [['lat', 'long']]
    cases[['lat', 'long']] = cases[['lat', 'long']].apply(flatten)
    cases = coords_to_array(cases)
    cases['kernel'] = controls.apply(lambda x: kernel_getter(x['array']), axis = 1)
    all_data = cases['array'].values.tolist()
    all_data = [item for sublist in all_data for item in sublist]   
    all_kernels = cases['kernel'].values.tolist()
    all_kernels = [item for sublist in all_kernels for item in sublist]

    #subset
    all_data = all_data[151:250]
    all_kernels = all_kernels[151:250]

    #distances
    support = [0.        , 0.00584394, 0.01168788, 0.01753182, 0.02337576,
       0.0292197 , 0.03506364, 0.04090759, 0.04675153, 0.05259547]

    #lessgo
    all_kinhoms = []
    for i in range(len(all_data)):
        data = all_data[i]
        kernels = all_kernels[i]
        kinhom = find_lambdas_and_distances(data, support, kernels)
        all_kinhoms.append(kinhom)

    with open('res_5.pkl', 'wb') as f:
        pickle.dump(all_kinhoms, f)



    