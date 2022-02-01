from concentration_lib.empirical_concentration_bounds import empirical_small_samples_ptlm
from concentration_lib.concentration_variance_bounds import empirical_chernoff_zero_mean_std_dev_bound, \
    chi2_std_dev_bound, bentkus_pinelis_std_dev_bound
from matplotlib.lines import Line2D
import utils
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import pickle

# plt.rcParams.update({
#     "text.usetex": True,
#     'text.latex.preamble': r'\usepackage{amsfonts}'})
sns.set_style("whitegrid")
from cvarBandits.utils.static_cis import set_fontsize

set_fontsize(12)
from itertools import cycle


def _get_empricial_means(sub_samples):
    sub_samples = np.asarray(sub_samples)
    n_rep, n_by_rep = sub_samples.shape
    empirical_means = []
    for sub_sample in sub_samples:
        empirical_mean = sub_sample.cumsum() / np.asarray(range(1, n_by_rep + 1))
        empirical_means.append(empirical_mean)
    return np.asarray(empirical_means)


def _get_empricial_vars(sub_samples, std=False):
    sub_samples = np.asarray(sub_samples)
    n_rep, n_by_rep = sub_samples.shape
    empirical_vars = []
    for sub_sample in sub_samples:
        empirical_var = []
        for index in range(n_by_rep):
            window = sub_sample[:index + 1]
            if std:
                empirical_var.append(np.std(window, ddof=1))
            else:
                empirical_var.append(np.var(window, ddof=1))
        empirical_vars.append(empirical_var)
    return np.asarray(empirical_vars)


def _parallel_bound_func(args):
    sub_sample, k, empirical_values, deltas, supp, mode = args
    all_bounds = []
    for j, delta in tqdm(enumerate(deltas)):
        bounds = []
        for i in tqdm(range(len(sub_sample)), leave=False):
            window = sub_sample[:i + 1]
            if i < 5:
                bound = [np.nan, np.nan]
            else:
                try:
                    if mode == 'mean':
                        f_ = empirical_small_samples_ptlm
                    else:
                        f_ = bentkus_pinelis_std_dev_bound
                    bound = f_(samples=window, delta=delta, lower_bound=supp[0], upper_bound=supp[1], side='both',
                               mode='mean', n=i + 1, tight=True)
                except Exception as e:
                    print(e)
                    bound = [np.nan, np.nan]
            bounds.append(bound)
        bounds = np.asarray(bounds).T
        bounds[0] = empirical_values - np.abs(bounds[0])
        bounds[1] = empirical_values + np.abs(bounds[1])
        all_bounds.append(bounds)
    return all_bounds


def get_bounds(sub_samples, deltas, supp, mode, clip=True, changing_lw=False, loading_path=None, plot=True, date=None):
    # linestyles = cycle([(0, (5, 10)), (0, (3, 5, 1, 5)), (0, (5, 1)), (0, (1, 1))])
    greys = cycle(reversed(['#262626', '#666666', '#999999', '#cccccc']))
    rep, n_by_rep = sub_samples.shape
    if mode == 'mean':
        empirical_values = _get_empricial_means(sub_samples)
        true_value = samples.mean()
    elif mode == 'std':
        empirical_values = _get_empricial_vars(sub_samples, std=True)
        true_value = samples.std(ddof=1)
    else:
        raise ValueError(f'mode value "{mode}" not in ["mean", "std"]')
    if loading_path is None:
        args = []
        for k, sub_sample in enumerate(sub_samples):
            args.append((sub_sample, k, empirical_values[k], deltas, supp, mode))
        with Pool() as pool:
            all_bounds = pool.map(_parallel_bound_func, args)
        bound_saving_path = f'./output/{mode}_bounds_rep_{rep}_n_{n_by_rep}.pkl'
        all_bounds = np.asarray(all_bounds)
        with open(bound_saving_path, 'wb') as f_:
            dict_to_save = {'deltas': deltas, 'all_bounds': all_bounds}
            pickle.dump(dict_to_save, f_)
    else:
        with open(loading_path, 'rb') as f_:
            saved_bounds = pickle.load(f_)
            deltas = saved_bounds['deltas']
            all_bounds = saved_bounds['all_bounds']
    if not plot:
        return
    all_bounds = all_bounds.mean(axis=0)
    fig, ax = plt.subplots()
    ax.axhline(y=true_value, linestyle='--', c='black')
    for j, (delta, bound, grey) in enumerate(zip(deltas, all_bounds, greys)):
        if clip:
            lower = [max(val, supp[0]) for val in bound[0].tolist()]
            upper = [min(val, supp[1]) for val in bound[1].tolist()]
        else:
            lower, upper = bound
        if changing_lw:
            lw = (len(deltas) - j) / len(deltas) * 2 + 1
        else:
            lw = 1
        ax.plot(range(n_by_rep), lower, label=f'{delta}', c=grey, linewidth=lw, linestyle='-')
        ax.plot(range(n_by_rep), upper, c=grey, linewidth=lw, linestyle='-')
    for empirical_value in empirical_values:
        ax.plot(range(n_by_rep), empirical_value, c='#808080', alpha=10 / rep, linewidth=1)
    ax.legend(title='risk level $\delta$', loc=1)
    deltas_legend = ax.get_legend()
    legend_empirical_values = Line2D([None], [None], linestyle='-', marker='None', c='#808080', alpha=.5, linewidth=.5)
    legend_true_value = Line2D([None], [None], linestyle='--', c='black', marker='None')
    legends = [legend_empirical_values, legend_true_value]
    labels = ('empirical values', 'true value')
    if mode == 'mean':
        loc_emp = 4
    else:
        loc_emp = 9
    legend_empirical_values = plt.legend(legends, labels, loc=loc_emp)
    ax.add_artist(legend_empirical_values)
    ax.add_artist(deltas_legend)
    ax.set_xlabel('sample number (n)')
    if mode == 'mean':
        ax.set_ylabel('mean simulated yield (kg/ha)')
        plt.title(f'Uncertainty for mean of simulated yield response, planting DOY {date}\n(\#{rep} replications)')
        saving_suffix = 'smallSampleMeanCI'
    else:
        ax.set_ylabel('standard deviation of simulated yield (kg/ha)')
        plt.title(f'Uncertainty for standard deviation of simulated yield response, planting DOY {date}\n(\#{rep} replications)')
        saving_suffix = 'BentkusStd'
    plt.savefig(f'./figures/{saving_suffix}.pdf', bbox_inches='tight')


def plot_error_delta(errors, type, supp):
    deltas = np.arange(start=1e-5, stop=.5 - 1e-5, step=1e-4)
    emp_std = errors.std(ddof=0)
    if type == 'comparison':
        comparison = True
    else:
        comparison = False
    if comparison:
        types = ['gaussian', 'chernoff', 'bentkus']
    else:
        types = [type]
    fig, ax = plt.subplots()
    for type_ in types:
        bounds = []
        for delta in deltas:
            if type_ == 'gaussian':
                bound_ = chi2_std_dev_bound(errors,
                                            delta=delta,
                                            side='both',
                                            mode='mean',
                                            )
                upper = emp_std + bound_[0]
                lower = emp_std - bound_[0]
                bound = [lower, upper]
            elif type_ == 'chernoff' or type_ == 'bentkus':
                if type_ == 'chernoff':
                    f_ = empirical_chernoff_zero_mean_std_dev_bound
                else:
                    f_ = bentkus_pinelis_std_dev_bound
                if delta >= np.exp(-len(errors) / 4):
                    try:
                        bound_ = f_(samples=errors,
                                    delta=delta,
                                    side='both',
                                    mode='mean',
                                    tight=True,
                                    lower_bound=-supp[1],
                                    upper_bound=supp[1],
                                    )
                        upper = emp_std + bound_[0]
                        lower = emp_std - bound_[0]
                        bound = [lower, upper]
                    except:
                        bound = [np.nan, np.nan]
                else:
                    bound = [np.nan, np.nan]
            else:
                raise ValueError('specified type incorrect: must lie in ["gaussian", "chernoff", "bentkus"]')
            bounds.append(bound)
        bounds = np.asarray(bounds).T
        if not comparison:
            ax.fill_between(x=deltas, y1=bounds[0], y2=bounds[1], color='none', hatch='X', edgecolor='black',
                            label='uncertainty')
        if comparison:
            if type_ == 'gaussian':
                label = 'Gaussian'
                c = '#000000'
                lw = 1
            elif type_ == 'chernoff':
                label = 'Centered second order sub-Gaussian'
                c = '#383838'
                lw = 2
            else:
                label = 'Sole boundedness'
                c = '#696969'
                lw = 3
        else:
            label = None
            c = 'black'
            lw = 2
        ax.plot(deltas, bounds[0], c=c, label=label, lw=lw)
        ax.plot(deltas, bounds[1], c=c, label=None, lw=lw)
    ax.set_xlabel('risk level $\delta$')
    ax.set_ylabel('standard deviation of model error (kg/ha)')
    ax.set_ylim([0, 4000])
    if comparison:
        legend_title = 'hypothesis'
    else:
        legend_title = None
    ax.legend(title=legend_title, loc=1)
    if comparison:
        ax.set_title('Comparison of model error confidence bands')
        saving_path = f'./figures/errorComp.pdf'
    else:
        if type == 'gaussian':
            ax.set_title('Model error standard deviation uncertainty\n(Gaussian hypothesis)')
            saving_suffix = 'gaussianError'
        elif type == 'chernoff':
            ax.set_title('Model error standard deviation uncertainty\n(centered second order sub-Gaussian hypothesis)')
            saving_suffix = 'stdErrorBM'
        else:
            ax.set_title('Model error standard deviation uncertainty\n(sole boundedness hypothesis)')
            saving_suffix = 'stdErrorBentkus'
        saving_path = f'./figures/{saving_suffix}.pdf'
    plt.tight_layout()
    plt.savefig(saving_path, bbox_inches='tight')


if __name__ == '__main__':
    plot = True  # if plots to performed
    compute_bounds = not True  # if precomputed data is loaded or bounds to be computed

    dirs = ['./figures', './output']
    utils.make_folder(dirs)

    calibration_df = pd.read_csv('./evaluation/mcgill_eval_df.csv', header=0, index_col=False).sort_values(['true'])
    calibration_df = calibration_df.dropna(axis=0, how='any').reset_index()

    ############### ERROR PART ###############
    errors = (calibration_df['true'] - calibration_df['pred']).to_numpy()
    if plot:
        for type in ['chernoff', 'comparison']:
            plot_error_delta(errors, type=type, supp=[0, 20000])

    ############### SAMPLE PART ###############
    data_path = './dssat_samples/dssat_mcgill_100000_MCGI100001_MG0001_samples_st_50.pkl'
    dates_to_select = [135]
    samples, n_samples, supp = utils.sample_loader(path=data_path,
                                                   dates_to_select=dates_to_select)
    samples, = samples
    n_by_rep = 100
    rep = 960
    sub_samples = samples[:rep * n_by_rep].reshape((rep, n_by_rep))
    deltas = [0.01, 0.05, 0.1, 0.3]
    if compute_bounds:
        mean_loading_path = None
        mv_loading_path = None
    else:
        mean_loading_path = f'./output/mean_bounds_rep_{rep}_n_{n_by_rep}.pkl'
        mv_loading_path = f'./output/std_bounds_rep_{rep}_n_{n_by_rep}.pkl'
    get_bounds(sub_samples, deltas, supp, 'mean', changing_lw=True, loading_path=mean_loading_path, plot=plot,
               date=dates_to_select[0])
    get_bounds(sub_samples, deltas, supp, 'std', changing_lw=True, loading_path=mv_loading_path, plot=plot,
               date=dates_to_select[0])
