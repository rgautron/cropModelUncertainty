import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.weightstats import ztest
from scipy.stats import kendalltau, spearmanr, pearsonr, normaltest
from pprint import pprint

if __name__ == '__main__':
    calibration_df = pd.read_csv('./evaluation/mcgill_eval_df.csv', header=0, index_col=False).sort_values(['true'])
    calibration_df = calibration_df.dropna(axis=0, how='any').reset_index()

    ############### ERROR PART ###############
    middle_index = len(calibration_df) // 2
    calibration_df_lower = calibration_df.iloc[:middle_index + 1]
    calibration_df_upper = calibration_df.iloc[middle_index + 1:]
    errors = (calibration_df['true'] - calibration_df['pred']).to_numpy()
    errors_lower = (calibration_df_lower['true'] - calibration_df_lower['pred']).to_numpy()
    errors_upper = (calibration_df_upper['true'] - calibration_df_upper['pred']).to_numpy()
    true_values = (calibration_df['true']).to_numpy()
    pred_values = (calibration_df['pred']).to_numpy()
    model_rmse = np.sqrt((errors ** 2).sum() / (len(errors) - 1))
    print(f'model_rmse: {model_rmse} kg/ha')
    test_dic = {
        'norm_test_res': normaltest(errors),
        'acorr_ljungbox': acorr_ljungbox(errors, lags=1),
        'ztest': ztest(errors, value=0, alternative='two-sided'),
        'kendalltau': kendalltau(errors, pred_values, method='exact'),
        'spearmanr': spearmanr(errors, pred_values),
        'pearsonr': pearsonr(errors, pred_values),
    }
    print('\n### Statistical tests###\n')
    pprint(test_dic)
