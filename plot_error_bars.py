import numpy as np
import matplotlib.pyplot as plt
from concentration_lib.empirical_concentration_bounds import empirical_small_samples_ptlm, \
    empirical_hedged_capital_bound
from concentration_lib.concentration_variance_bounds import empirical_chernoff_zero_mean_std_dev_bound, \
    bentkus_pinelis_std_dev_bound
import utils
import seaborn as sns
import pandas as pd

# plt.rcParams.update({
#     "text.usetex": True,
#     'text.latex.preamble': r'\usepackage{amsfonts}'})
sns.set_style("whitegrid")
from cvarBandits.utils.static_cis import set_fontsize

set_fontsize(12)


def find_delta_bisection(samples, supp, errors, rho=1, tolerance=1e-6, n_limit=None, MV=False, relative=False):
    assert len(samples) > 1
    n_options = len(samples)
    delta_low = tolerance
    delta_up = .5 - tolerance
    solution = False

    def compute_bounds(sample, delta, supp, n_limit, MV):
        if MV:
            delta_ = delta / (6 * n_options)  # delta_ is the risk level for each bound
        else:
            delta_ = delta / (4 * n_options)
        if len(sample) < 100:
            f_ = empirical_small_samples_ptlm
        else:
            f_ = empirical_hedged_capital_bound
        if n_limit is None:
            n_limit = len(sample)
        else:
            assert n_limit <= len(sample)
        sample = sample[:n_limit]
        lower, upper = 0, 0
        lower += sample.mean() - f_(samples=sample,
                                    delta=delta_,
                                    upper_bound=supp[-1],
                                    lower_bound=supp[0],
                                    mode='mean',
                                    side='lower')
        upper += sample.mean() + f_(samples=sample,
                                    delta=delta_,
                                    upper_bound=supp[-1],
                                    lower_bound=supp[0],
                                    mode='mean',
                                    side='lower')
        # print(f'sample ci: {lower, upper}')
        if MV:
            lower_std_samples = sample.std(ddof=1) - bentkus_pinelis_std_dev_bound(samples=sample,
                                                                                   delta=delta_,
                                                                                   upper_bound=supp[-1],
                                                                                   lower_bound=supp[0],
                                                                                   mode='mean',
                                                                                   side='lower')
            upper_std_samples = sample.std(ddof=1) + bentkus_pinelis_std_dev_bound(samples=sample,
                                                                                   delta=delta_,
                                                                                   upper_bound=supp[-1],
                                                                                   lower_bound=supp[0],
                                                                                   mode='mean',
                                                                                   side='upper')
            lower_std_error = errors.std(ddof=0) - empirical_chernoff_zero_mean_std_dev_bound(samples=errors,
                                                                                              delta=delta_,
                                                                                              mode='mean',
                                                                                              side='lower')
            upper_std_error = errors.std(ddof=0) + empirical_chernoff_zero_mean_std_dev_bound(samples=errors,
                                                                                              delta=delta_,
                                                                                              mode='mean',
                                                                                              side='upper')
            # print(f'error ci: {lower_std_error, upper_std_error}')
            lower -= rho * np.sqrt(upper_std_samples ** 2 + upper_std_error ** 2)
            upper -= rho * np.sqrt(lower_std_samples ** 2 + lower_std_error ** 2)
        return lower, upper

    while delta_up - delta_low > tolerance:
        delta = .5 * (delta_up + delta_low)
        bounds = []
        bounds_ = []
        for sample in samples:
            lower, upper = compute_bounds(sample=sample, delta=delta, supp=supp, n_limit=n_limit, MV=MV)
            lower_absolute = lower
            upper_absolute = upper
            if relative:
                if MV:
                    empirical_value = sample.mean() - rho * np.sqrt(errors.var(ddof=0) + sample.var(ddof=1))
                else:
                    empirical_value = sample.mean()
                lower = empirical_value - lower_absolute
                upper = upper_absolute - empirical_value
            else:
                lower = lower_absolute
                upper = upper_absolute
            bounds.append([lower, upper])
            bounds_.append([lower_absolute, upper_absolute])
        bounds_T = np.asarray(bounds_).T
        max_lower_bound_index = np.argmax(bounds_T[0])
        max_lower_bound = bounds_T[0][max_lower_bound_index]
        other_indexes = []
        for index in range(len(samples)):
            if index != max_lower_bound_index:
                other_indexes.append(index)
        other_upper_bounds = bounds_T[1][other_indexes]
        if np.all(max_lower_bound > other_upper_bounds):
            solution = True
            delta_up = delta
        else:
            delta_low = delta
        print(delta)
    if not solution:
        print('Warning: no risk level found to satisfy conditions!')
    else:
        print('Succeeded to have disjoint confidence intervals!')
    return solution, delta, bounds


def error_bar_plot(samples, supp, errors, saving_name, rho=1, n_limit=None, MV=False, xlabels=None, tolerance=1e-6):
    errors = np.asarray(errors)
    n_limits = []

    for i, sample in enumerate(samples):
        sample = np.asarray(sample, dtype=float)
        samples[i] = sample
        if n_limit > len(sample):
            n_limit = len(sample)
        n_limits.append(n_limit)

    empirical_means = np.asarray([sample[:n_limit_].mean() for sample, n_limit_ in zip(samples, n_limits)])
    empirical_sample_vars = np.asarray([sample[:n_limit_].var(ddof=1) for sample, n_limit_ in zip(samples, n_limits)])
    empirical_error_var = errors.var(ddof=1)

    if MV:
        empirical_values = empirical_means - rho * np.sqrt(empirical_sample_vars + empirical_error_var)
    else:
        empirical_values = empirical_means
    solution, delta, bounds = find_delta_bisection(samples, supp, errors, rho=1, n_limit=n_limit, MV=MV, relative=True,
                                                   tolerance=tolerance)
    bounds = np.asarray(bounds).T
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.errorbar(range(len(samples)), empirical_values, yerr=bounds, linestyle='None', color='black', capsize=10,
                elinewidth=2, capthick=2)
    y_lim = ax.get_ylim()
    max_value = y_lim[-1]
    ax.set_ylim([0, 1.05 * max_value])
    x_lim = ax.get_xlim()
    ax.set_xlim([x_lim[0] - .1, x_lim[1] + .1])
    if xlabels is None:
        xlabels = [f'planting date {i}' for i in range(len(samples))]
    plt.xticks(range(len(samples)), xlabels)
    delta_ = np.ceil(100000 * delta) / 100000
    # formatted_delta = f'{delta_:0.2e}'
    formatted_delta = f'{100 * delta_:.03f}'
    if MV and example_1:
        loc = 1
    else:
        loc = 4
    if MV:
        label = 'empirical mean-variance'
        plt.title(f'Mean-variance uncertainty\nat risk level $\delta=$ {formatted_delta}\%')
    else:
        label = 'empirical mean'
        plt.title(f'Mean uncertainty\nat risk level $\delta=$ {formatted_delta}\%')
    ax.plot(range(len(samples)), empirical_values, linestyle='None', marker='*', markersize=10, color='black',
            label=label)
    # plt.tight_layout()
    if not MV:
        ax.set_ylabel('mean yield (kg/ha)')
    else:
        ax.set_ylabel('mean-variance yield (kg/ha)')
    ax.legend(loc=loc)
    plt.subplots_adjust(left=0.2)
    plt.savefig(f'./figures/{saving_name}.pdf')


if __name__ == '__main__':
    dirs = ['./figures']
    utils.make_folder(dirs)

    errors = pd.read_csv('./evaluation/mcgill_eval_df.csv', header=0, index_col=False)
    errors = (errors['true'] - errors['pred']).to_numpy()
    errors = errors[~np.isnan(errors)]
    data_path = './dssat_samples/dssat_mcgill_100000_MCGI100001_MG0001_samples_st_50.pkl'
    for example_1, MV in [(True, True), (True, False), (False, True)]:
        if MV:
            n_limit = 10000
        else:
            n_limit = 500
        if example_1:
            dates_to_select = [135, 165]
            if MV:
                saving_suffix = 'mvErrors1'
            else:
                saving_suffix = 'meanErrors1'
        else:
            dates_to_select = [135, 155]
            if MV:
                saving_suffix = 'mvErrors2'
            else:
                saving_suffix = 'meanErrors2'
        xlabels = [f'planting date {doy}' for doy in dates_to_select]
        samples, n_samples, supp = utils.sample_loader(path=data_path, dates_to_select=dates_to_select)
        error_bar_plot(samples, supp, errors, saving_suffix, rho=1, n_limit=n_limit, MV=MV, tolerance=1e-5,
                       xlabels=xlabels)
