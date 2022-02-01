import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{amsfonts}'})
from cvarBandits.utils.static_cis import set_fontsize

set_fontsize(12)
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_style("whitegrid")
from scipy.stats import gaussian_kde
from itertools import cycle
import numpy as np
import pandas as pd
import utils

def render_dists(samples, supp, n_samples, labels, saving_path):
    markers = cycle(['o', '^', 's', 'p', 'X', 'D', 'd'])
    greys = cycle(['#808080', '#2c2c2c', '#000000', '#bdbebd'])
    fig, ax = plt.subplots()
    legend_elements = []
    for sample, label in zip(samples, labels):
        x_kde = np.linspace(start=supp[0], stop=supp[1], num=10000)
        sample = np.asarray(sample, dtype=float)
        kde_f = gaussian_kde(sample)
        y_kde = kde_f(x_kde)
        x_kde_points_step = supp[1] // 10
        x_kde_points = range(x_kde_points_step, supp[1], x_kde_points_step)
        y_kde_points = kde_f(x_kde_points)
        mean = sample.mean()
        marker = next(markers)
        color = next(greys)
        dash = (1, 0, 0, 0)
        kde_p = ax.plot(x_kde, y_kde, alpha=1, color=color, zorder=1, linewidth=3)
        x_kde_points = range(x_kde_points_step, supp[1], x_kde_points_step)
        points_kde = ax.plot(x_kde_points, y_kde_points, color=color, marker=marker, zorder=3, markersize=8,
                             linestyle='')
        legend_element = Line2D([None], [None], label=label, marker=marker, color=color, markersize=8, linewidth=3)
        legend_elements.append(legend_element)
        ax.axvline(x=mean, color=color, linestyle='dashed', linewidth=2)
    upper_bound = ax.axvline(x=supp[1], color='black', linestyle='dashdot', linewidth=2, label='yield upper bound')
    metric_legend_label = 'empirical mean'
    metric_legend = Line2D([None], [None], color='black', lw=2, label=metric_legend_label, linestyle='--')
    legend_elements.append(metric_legend)
    legend_elements.append(upper_bound)
    ax.set_xlabel('dry grain yield (kg/ha)')
    ax.set_ylabel('density')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_aspect('auto')
    ax.set_xlim([supp[0], 1.05 * supp[1]])
    ax.set_xbound(lower=supp[0], upper=1.05 * supp[1])
    title = f'Empirical distributions estimated after \#{n_samples:.0e} samples'
    plt.title(title)
    ax.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    plt.savefig(saving_path)
    plt.close()


def empirical_mean_variance(sample, errors, rho=1):
    sample = np.asarray(sample)
    mu_sim_hat = sample.mean()
    print('mu_sim_hat ', mu_sim_hat)
    sigma2_sim_hat = sample.var(ddof=1)
    print('sigma_sim_hat ', np.sqrt(sigma2_sim_hat))
    sigma2_err_hat = errors.var(ddof=0)
    empirical_mv = mu_sim_hat - rho * np.sqrt(sigma2_sim_hat + sigma2_err_hat)
    return empirical_mv


if __name__ == '__main__':
    dirs = ['./figures']
    utils.make_folder(dirs)
    errors = pd.read_csv('./evaluation/mcgill_eval_df.csv', header=0, index_col=False)
    errors = (errors['true'] - errors['pred']).to_numpy()
    errors = errors[~np.isnan(errors)]
    data_path = './dssat_samples/dssat_mcgill_100000_MCGI100001_MG0001_samples_st_50.pkl'
    for example_1 in [False, True]:
        if example_1:
            dates_to_select = [135, 165]
            saving_path = f'./figures/dssatDistsMean.pdf'
        else:
            dates_to_select = [135, 155]
            saving_path = f'./figures/dssatDistsMV.pdf'
        xlabels = [f'planting date {doy}' for doy in dates_to_select]
        samples, n_samples, supp = utils.sample_loader(path=data_path, dates_to_select=dates_to_select)
        for sample, date in zip(samples, dates_to_select):
            print(f'\n##### DATE {date} #####')
            empirical_mv = empirical_mean_variance(sample[:10000], errors, rho=1)
            print(f'empirical mean-variance {empirical_mv}')
            print('\n###################')
        render_dists(samples=samples, supp=supp, n_samples=n_samples, labels=xlabels, saving_path=saving_path)