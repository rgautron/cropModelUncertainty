import pickle
import numpy as np
import pathlib
import pdb

def load_data(path):
    with open(path, 'rb') as f_:
        data = pickle.load(f_)
    samples = data['samples']
    n_samples = data['env_param_dic']['n_samples']
    supp = [0, data['env_param_dic']['eta_max']]
    action_values = [int(action_value[0][-3:]) for action_value in data['action_values']]
    return samples, n_samples, supp, action_values

def sample_loader(path, dates_to_select=None, n_limit=None):
    dssat_samples = load_data(path)
    samples, n_samples, supp, action_values = dssat_samples
    samples = [np.asarray(sample[:n_limit], dtype=float) for sample in samples]
    if dates_to_select is not None:
        restriction_list = [action_value in dates_to_select for action_value in action_values]
        samples = [sample for sample, bool_select in zip(samples, restriction_list) if bool_select]
    return samples, n_samples, supp

def load_from_multiple_files(file_paths, dates_to_select, n_limit=None):
    all_samples = []
    for file_path, dates_to_select_ in zip(file_paths, dates_to_select):
        samples, n_samples, supp = sample_loader(file_path, dates_to_select_, n_limit=n_limit)
        samples = [sample.astype('float') for sample in samples]
        all_samples.extend(samples)
    return all_samples, n_samples, supp

def make_folder(dirs):
    for dir_ in dirs:
        path = pathlib.Path(dir_)
        path.mkdir(exist_ok=True)
