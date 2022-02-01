from cvarBandits.utils.static_cis import set_fontsize
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import utils

# plt.rcParams.update({
#     "text.usetex": True,
#     'text.latex.preamble': r'\usepackage{amsfonts}'})
sns.set_style("whitegrid")
set_fontsize(12)


def plot_pred_vs_true(y_true, y_pred, years, x_label=None, y_label=None, title=None, residue_plot=False,
                      predResCorr=False):
    if predResCorr:
        residue_plot = True
    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
    symbols = ['*', 'X', 'h']
    markers = []
    if x_label is None:
        if predResCorr:
            x_label = 'predicted yield (kg/ha)'
        else:
            x_label = 'measured yield (kg/ha)'
    if y_label is None:
        if not residue_plot:
            y_label = 'predicted yield (kg/ha)'
        else:
            y_label = 'residual [measured-pred] (kg/ha)'
    for year in years:
        if year <= 2000:
            markers.append(symbols[0])
        elif year <= 2007:
            markers.append(symbols[1])
        else:
            markers.append(symbols[2])
    if residue_plot:
        y_plot = y_true - y_pred
    else:
        y_plot = y_pred
    if predResCorr:
        X = y_pred
    else:
        X = y_true
    for x, y, marker in zip(X, y_plot, markers):
        ax.scatter(x=x, y=y, marker=marker, c='black', s=40, zorder=3, alpha=.5)
    marker1 = Line2D([None], [None], linestyle='none', marker=symbols[0], c='black', markersize=np.sqrt(40), zorder=3,
                     alpha=.5)
    marker2 = Line2D([None], [None], linestyle='none', marker=symbols[1], c='black', markersize=np.sqrt(40), zorder=3,
                     alpha=.5)
    marker3 = Line2D([None], [None], linestyle='none', marker=symbols[2], c='black', markersize=np.sqrt(40), zorder=3,
                     alpha=.5)
    legend_1_elements = (marker1, marker2, marker3)
    labels_1 = ('1', '2', '3')
    legend_1 = plt.legend(legend_1_elements, labels_1, title='cultivar', loc=2)
    ax.add_artist(legend_1)
    ax.set_aspect('equal')
    if residue_plot:
        ax.plot([min(X), max(X)], [0, 0], linestyle='--', color='black', linewidth=2, zorder=2)
    else:
        max_val = max(max(y_true), max(y_pred))
        ax.plot([0, max_val], [0, max_val], linestyle='--', color='black', linewidth=2, zorder=2)
        lines = [[(y_true_, y_pred_), (y_true_, y_true_)] for y_true_, y_pred_ in zip(y_true, y_pred)]
        lc = mc.LineCollection(lines, colors='#606060', linewidths=1, linestyles='-', zorder=1)
        ax.add_collection(lc)
        error_line = Line2D([None], [None], linewidth=1, linestyle='-', c='#606060')
        label_2 = ('prediction error',)
        legend_2_elements = (error_line,)
        legend_2 = plt.legend(legend_2_elements, label_2, loc=4)
        ax.add_artist(legend_2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)
    if predResCorr:
        plt.savefig(f'./figures/ysimEcorr.pdf', bbox_inches='tight')
    else:
        if residue_plot:
            saving_suffix = 'residualPlot'
        else:
            saving_suffix = 'dssatPredictions'
        plt.savefig(f'./figures/{saving_suffix}.pdf', bbox_inches='tight')


def qqplot(samples, errors=True, date=None):
    p_ = sm.qqplot(samples, line='s', fit=True)
    ax = plt.gca()
    lines = ax.get_lines()
    points, line = lines
    line.set_linestyle('--')
    line.set_color('black')
    line.set_linewidth('2')
    points.set_marker('x')
    points.set_markerfacecolor('#404040')
    points.set_markeredgecolor('#404040')
    points.set_markersize(np.sqrt(80))
    if errors:
        plt.title(f'Normal versus model error quantile plot ')
    else:
        points.set_alpha(.1)
        plt.title(f'Normal versus simulated yield quantile plot for planting date DOY {date}')
    if errors:
        saving_suffix = 'qqplotErrors'
    else:
        saving_suffix = 'qqplotYsim'
    plt.savefig(f'./figures/{saving_suffix}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    dirs = ['./figures']
    utils.make_folder(dirs)

    eval_df = pd.read_csv('./evaluation/mcgill_eval_df.csv', header=0, index_col=False)
    eval_df = eval_df.dropna()
    errors = (eval_df['true'] - eval_df['pred']).to_numpy()
    pred_plot_title = "Ground truth yield values versus DSSAT's predictions"
    years = eval_df['year']
    for residue_plot, predResCorr in [[True, False], [False, False], [True, True]]:
        plot_pred_vs_true(eval_df['true'], eval_df['pred'], title=pred_plot_title, years=years, residue_plot=residue_plot,
                          predResCorr=predResCorr)
    data_path = './dssat_samples/dssat_mcgill_100000_MCGI100001_MG0001_samples_st_50.pkl'
    dates_to_select = [135]
    samples, n_samples, supp = utils.sample_loader(path=data_path, dates_to_select=dates_to_select)
    sample, = samples
    for qqplot_error in [False, True]:
        if qqplot_error:
            qqplot(errors, errors=qqplot_error)
        else:
            qqplot(sample, errors=qqplot_error, date=dates_to_select[0])