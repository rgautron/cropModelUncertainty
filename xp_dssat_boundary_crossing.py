import numpy as np
import pickle
import argparse
from joblib import Parallel, delayed
import multiprocessing as mp
from time import time
import os
from tqdm import tqdm
from utils import load_data
from concentration_lib import empirical_student_bound
from concentration_lib import empirical_bernstein_bound
from concentration_lib import empirical_bentkus_bound
from concentration_lib import empirical_hedged_capital_bound
from concentration_lib import empirical_small_samples_ptlm


# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-N', default=200, type=int)
parser.add_argument('-M', default=1000, type=int)
parser.add_argument('-d', default=0.05, type=float)
parser.add_argument('--path', '-p', default='results', type=str)
parser.add_argument('--parallel', default=1, type=int)

# Set arguments
args = parser.parse_args()
N = args.N
M = args.M
delta = args.d
pickle_path = args.path
parallel = args.parallel > 0

samples, n_samples, supp, action_values = load_data('./dssat_samples/dssat_mcgill_100000_MCGI100001_MG0001_samples_st_50.pkl')

idx = np.where(np.array(action_values) == 135)[0][0]

samples = samples[idx]
n = len(samples)

# "True" mean, taken as empirical means over the whole sample
mu = np.mean(samples)

# "True" standard deviation, taken as empirical std_dev over the whole sample
sigma = np.std(samples)

# Shuffle samples
samples = samples[np.random.rand(n).argsort()]
lower_bound, upper_bound = supp

params = {
    'upper_bound': upper_bound,
    'lower_bound': lower_bound,
    'mu': mu,
    'sigma': sigma,
    'n_MC': 500,
    }


def concentration_bounds(
    sample: np.array,
    N: int,
    M: int,
    delta: float = 0.05,
    params: dict = {},
    verbose: bool = False
):
    """
    sample: np.array
    Size must be at least N * M.

    N: int
    Sample size for a given estimator.

    M: int
    Number of independent repetitions.

    delta: float
    Confidence level.

    params: dict
    Parameters that influence concentration (bounds, variance...).

    verbose: bool
    For tqdm.
    """
    sample = sample[np.random.rand(sample.shape[0]).argsort()]
    sample = sample[:N * M].reshape((N, M))

    mu_hats = np.zeros((N, M))

    bounds_student = np.zeros((N, M, 2)) * np.nan
    bounds_bernstein = np.zeros((N, M, 2)) * np.nan
    bounds_bentkus = np.zeros((N, M, 2)) * np.nan
    bounds_hedged_capital = np.zeros((N, M, 2)) * np.nan
    bounds_ptlm = np.zeros((N, M, 2)) * np.nan

    boundary_crossings_student = np.zeros((N, M, 2)) * np.nan
    boundary_crossings_bernstein = np.zeros((N, M, 2)) * np.nan
    boundary_crossings_bentkus = np.zeros((N, M, 2)) * np.nan
    boundary_crossings_hedged_capital = np.zeros((N, M, 2)) * np.nan
    boundary_crossings_ptlm = np.zeros((N, M, 2)) * np.nan

    upper_bound = params.get('upper_bound')
    lower_bound = params.get('lower_bound')
    mu = params.get('mu')
    n_MC = params.get('n_MC')

    for n in tqdm(range(1, N), disable=not verbose):
        for m in range(M):
            mu_hats[n, m] = np.mean(sample[:n, m])
            bounds_student[n, m, :] = empirical_student_bound(
                sample[:n, m], delta, side='both', mode='mean')
            bounds_bernstein[n, m, :] = empirical_bernstein_bound(
                sample[:n, m], delta, upper_bound, lower_bound, side='both', mode='mean')
            bounds_bentkus[n, m, :] = empirical_bentkus_bound(
                sample[:n, m], delta, upper_bound, lower_bound, side='both', mode='mean')
            bounds_hedged_capital[n, m, :] = empirical_hedged_capital_bound(
                sample[:n, m], delta, upper_bound, lower_bound, side='both', mode='mean')
            try:
                bounds_ptlm[n, m, :] = empirical_small_samples_ptlm(
                    sample[:n, m], delta=delta,
                    lower_bound=lower_bound, upper_bound=upper_bound, n_MC=n_MC,
                    side='both', mode='mean',
                    )
            except:
                continue

            boundary_crossings_student[n, m, 0] = mu < mu_hats[n, m] - bounds_student[n, m, 0]
            boundary_crossings_bernstein[n, m, 0] = mu < mu_hats[n, m] - bounds_bernstein[n, m, 0]
            boundary_crossings_bentkus[n, m, 0] = mu < mu_hats[n, m] - bounds_bentkus[n, m, 0]
            boundary_crossings_hedged_capital[n, m, 0] = mu < mu_hats[n, m] - bounds_hedged_capital[n, m, 0]
            boundary_crossings_ptlm[n, m, 0] = mu < mu_hats[n, m] - bounds_ptlm[n, m, 0]

            boundary_crossings_student[n, m, 1] = mu > mu_hats[n, m] + bounds_student[n, m, 1]
            boundary_crossings_bernstein[n, m, 1] = mu > mu_hats[n, m] + bounds_bernstein[n, m, 1]
            boundary_crossings_bentkus[n, m, 1] = mu > mu_hats[n, m] + bounds_bentkus[n, m, 1]
            boundary_crossings_hedged_capital[n, m, 1] = mu > mu_hats[n, m] + bounds_hedged_capital[n, m, 1]
            boundary_crossings_ptlm[n, m, 1] = mu > mu_hats[n, m] + bounds_ptlm[n, m, 1]

    return {
        'mu_hat': mu_hats,
        'bound': {
            'student': bounds_student,
            'bernstein': bounds_bernstein,
            'bentkus': bounds_bentkus,
            'hedged_capital': bounds_hedged_capital,
            'ptlm': bounds_ptlm,
        },
        'boundary_crossing': {
            'student': boundary_crossings_student,
            'bernstein': boundary_crossings_bernstein,
            'bentkus': boundary_crossings_bentkus,
            'hedged_capital': boundary_crossings_hedged_capital,
            'ptlm': boundary_crossings_ptlm,
        }
    }


def MC_xp(args, pickle_path=None, caption='xp'):
    sample, N, M, delta, params, verbose = args
    res = concentration_bounds(sample, N, M, delta, params, verbose)

    if pickle_path is not None:
        pickle.dump(res, open(os.path.join(pickle_path, caption+'.pkl'), 'wb'))
    return res


def multiprocess_MC(args, pickle_path=None, caption='xp', parallel=True):
    t0 = time()
    cpu = mp.cpu_count()
    print('Running on %i clusters' % cpu)
    sample, N, M, delta, params, verbose = args
    new_args = (sample, N, M // cpu + 1, delta, params, verbose)
    if parallel:
        res_ = Parallel(n_jobs=cpu)(delayed(MC_xp)(new_args) for _ in range(cpu))
        res = {}
        res['mu_hat'] = np.concatenate([res_[i]['mu_hat'] for i in range(cpu)], axis=1)
        res['student'] = np.concatenate([res_[i]['bound']['student'] for i in range(cpu)], axis=1)
        res['bernstein'] = np.concatenate([res_[i]['bound']['bernstein'] for i in range(cpu)], axis=1)
        res['bentkus'] = np.concatenate([res_[i]['bound']['bentkus'] for i in range(cpu)], axis=1)
        res['hedged_capital'] = np.concatenate([res_[i]['bound']['hedged_capital'] for i in range(cpu)], axis=1)
        res['ptlm'] = np.concatenate([res_[i]['bound']['ptlm'] for i in range(cpu)], axis=1)
        res['bcp_student'] = np.concatenate([res_[i]['boundary_crossing']['student'] for i in range(cpu)], axis=1)
        res['bcp_bernstein'] = np.concatenate([res_[i]['boundary_crossing']['bernstein'] for i in range(cpu)], axis=1)
        res['bcp_bentkus'] = np.concatenate([res_[i]['boundary_crossing']['bentkus'] for i in range(cpu)], axis=1)
        res['bcp_hedged_capital'] = np.concatenate([res_[i]['boundary_crossing']['hedged_capital'] for i in range(cpu)], axis=1)
        res['bcp_ptlm'] = np.concatenate([res_[i]['boundary_crossing']['ptlm'] for i in range(cpu)], axis=1)
    else:
        res = MC_xp(new_args)

    info = {'N': N, 'M': M, 'delta': delta, 'params': params}
    xp_container = {'results': res, 'info': info}

    if pickle_path is not None:
        pickle.dump(
            xp_container,
            open(os.path.join(pickle_path, caption+'.pkl'), 'wb')
            )
    print('Execution time: {:.0f} seconds'.format(time()-t0))
    return xp_container


cap = 'DSSAT_' + str(int(np.random.uniform() * 1e6))
print(cap)
res, traj = multiprocess_MC(
    (samples, N, M, delta, params, True),
    pickle_path=pickle_path,
    caption=cap,
    parallel=parallel,
)
