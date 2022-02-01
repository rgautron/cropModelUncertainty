from gym_dssat.envs.dssat_env import DssatEnv
import logging
import shutil
import pathlib
import utils

if __name__ == '__main__':
    dirs = ['./dssat_samples', './figures']
    utils.make_folder(dirs)
    sampling = True
    date_steps = 40
    date_delta = 5
    n_loss_samples = int(1e5)
    id_soil = 'MCGI100001'
    render = True
    file_names_to_copy = ['MC.SOL', 'MCGI9001.WTH', 'MCGI.CLI']
    sources = [f'./dssat_files/{source}' for source in file_names_to_copy]
    dssat_path = f'{pathlib.Path.home()}/dssat'  # to adapt to the user's DSSAT location
    destination_folders = ['Soil', 'Weather', 'Weather/Climate']
    destinations = [f'{dssat_path}/{suffix}/{name}' for suffix, name in zip(destination_folders, file_names_to_copy)]
    for source, destination in zip(sources, destinations):
        shutil.copyfile(source, destination)
    start_doys_ = [[50]]
    cultivars = [1]
    seed = 1234
    for start_doys in start_doys_:
        for ingeno_index, start_doy in zip(cultivars, start_doys):
            sample_saving_path = f'./dssat_samples/dssat_mcgill_{n_loss_samples:.0f}_{id_soil}_MG000{ingeno_index}_samples_st_{start_doy}.pkl'
            sample_loading_path = f'./dssat_samples/dssat_mcgill_{n_loss_samples:.0f}_{id_soil}_MG000{ingeno_index}_samples_st_{start_doy}.pkl'
            render_path = f'./figures/dssat_render_{n_loss_samples:.0f}_{id_soil}_MG000{ingeno_index}_samples_mean_random_cultivar_st_{start_doy}.pdf'
            dssat_param_dic = {
                'fileX_prefix': 'MCGI9001',
                'fileX_extension': '.MZX',
                'ingeno': f'MG000{ingeno_index}',
                'output': 'HWAM',
                'icdat': f'92{start_doy:03d}',
                'planting_date': f'92{start_doy:03d}',
                'sdate': f'92{start_doy:03d}',
                'random_weather': True,
                'random_cultivar': False,
                'cultivar_path': 'cultivars.csv',
                'files_prefix': './dssat_files/',
                # 'auxiliary_files_names': file_names_to_copy,
                'id_soil': id_soil,
                'random_soil_init': False,
            }

            env_param_dic = {
                'cultivar': False,
                'sowing_date': True,
                'stateless': True,
                'date_steps': date_steps,
                'date_delta': date_delta,
                'n_loss_samples': n_loss_samples,
                'eta_max': 20000,
                'n_samples': n_loss_samples,
            }
            restriction_list = [True for _ in range(date_steps + 1)]
            env_instance = DssatEnv
            env = env_instance()
            env._init_(dssat_param_dic=dssat_param_dic, env_param_dic=env_param_dic, **env_param_dic)
            try:
                if sampling:
                    env.set_seed(seed)
                    env.get_dist_params(saving_path=sample_saving_path)
                else:
                    env.get_dist_params(loading_path=sample_loading_path, restriction_list=restriction_list)
                if render:
                    env.render_env(render_path, kde=True, hist=False)

            except Exception as e:
                logging.exception(e)

            finally:
                env.dssat.close()  # removing temporary folder