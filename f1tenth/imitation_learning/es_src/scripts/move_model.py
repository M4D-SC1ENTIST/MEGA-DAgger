import os
import fire
import shutil
import re

tranfer_file = ['opp', 'pareto']


def generate_model_from_training_data(run=31, origin='data', epoch=120):
    module_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    run_data_path = os.path.join(module_path, origin, str(run))
    model_path = os.path.join(module_path, 'es_model', str(run))
    os.makedirs(model_path, exist_ok=True)
    max_epoch = 0
    last_epoch_file = None
    for file in os.listdir(run_data_path):
        # print(file)
        # print(os.path.isdir(file))
        if not os.path.isdir(os.path.join(run_data_path, file)):
            shutil.copy(os.path.join(run_data_path, file), os.path.join(model_path, file))
        else:
            if file == 'batch_data':
                continue
            fs = os.listdir(os.path.join(run_data_path, file))
            n = len(fs)
            for i in range(n):
                f = fs[i]
                name = f.split('_')
                if 'opp' in name:
                    shutil.copy(os.path.join(run_data_path, file, f), os.path.join(model_path, f))
                else:
                    idx = re.findall(r'\d+', name[3])[1]
                    if not epoch:
                        if int(idx) >= max_epoch:
                            last_epoch_file = f
                            max_epoch = int(idx)
                    else:
                        if int(idx) == epoch:
                            last_epoch_file = f
                            break
            shutil.copy(os.path.join(run_data_path, file, last_epoch_file), os.path.join(model_path, last_epoch_file))


generate_model_from_training_data()
