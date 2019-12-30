import os
import filelock
import pandas as pd
import numpy as np
import torch


def save_agent(A, filename):
    torch.save(A, filename)


def load_model(filename, device=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.load(filename, map_location=device)


def _liststring_to_list(lstring):
    lst = []
    lstring_clean = lstring.replace('[', '').replace(']', '')
    if len(lstring_clean) > 0:
        if ',' not in lstring_clean:
            lst = [int(lstring_clean)]
        else:
            lst = [int(i) for i in lstring_clean.split(',')]
    return lst


def create_agent_from_row(row, aclass, amodel_class, additional_args={}):
    params_dict = row.to_dict()
    params_dict.update(additional_args)
    modelfile = params_dict.pop("model_filename")
    params_dict.pop("results_filename")
    params_dict['model'] = amodel_class  # Need a class rather than string
    params_dict['hidden_layers'] = _liststring_to_list(params_dict['hidden_layers'])
    if 'verbose' not in params_dict:
        params_dict['verbose'] = True
    agent = aclass(**params_dict)
    agent.set_weights(load_model(modelfile, agent.device))
    return agent


class PandasDF():
    def __init__(self, HOME_DIR, RESULTS_DIR, MODELS_DIR, PANDAS_DATAFRAME_FILENAME, RESULTS_KEY, MODELS_KEY):
        self.HOME_DIR = HOME_DIR
        self.RESULTS_DIR = RESULTS_DIR
        self.MODELS_DIR = MODELS_DIR
        self.PANDAS_DATAFRAME_FILENAME = PANDAS_DATAFRAME_FILENAME
        self.RESULTS_KEY = RESULTS_KEY
        self.MODELS_KEY = MODELS_KEY

        self.generate_folder_structure()
        self.df = self.load_pandas_df()

    def save_results(self, results):
        num_unique = self.df[self.RESULTS_KEY].nunique()
        filename = os.path.join(self.RESULTS_DIR, 'results_' + str(num_unique) + '.npy')
        np.save(filename, results)
        return filename

    def save_model(self, C):
        num_unique = self.df[self.MODELS_KEY].nunique()
        filename = os.path.join(self.MODELS_DIR, 'agent_' + str(num_unique) + '.npy')
        save_agent(C, filename)
        return filename

    def load_pandas_df(self, lock=None):
        if lock is None:
            lock = filelock.FileLock(self.PANDAS_DATAFRAME_FILENAME + ".lock")
        with lock:
            if os.path.exists(self.PANDAS_DATAFRAME_FILENAME):
                df = pd.read_excel(self.PANDAS_DATAFRAME_FILENAME)
                return df
            else:
                return None

    def generate_folder_structure(self):
        if not os.path.exists(self.HOME_DIR):
            os.mkdir(self.HOME_DIR)
        if not os.path.exists(self.RESULTS_DIR):
            os.mkdir(self.RESULTS_DIR)
        if not os.path.exists(self.MODELS_DIR):
            os.mkdir(self.MODELS_DIR)

        if not os.path.exists(self.PANDAS_DATAFRAME_FILENAME):
            self.df = pd.DataFrame()
            new_cols_list = [self.RESULTS_KEY, self.MODELS_KEY]
            for key in new_cols_list:
                if key not in self.df:
                    new_col = ['None'] * max(1, len(self.df))
                    self.df.insert(0, key, new_col)
            self.save_df()

    def search_rows(self, merged_dict):
        '''
        Search for rows with columns keys and entries values in merged_dict
        '''
        md = merged_dict.copy()  # Shallow copy so we don't overwrite merged_dict values
        # First go over keys and see if column present
        for key in md:
            if isinstance(md[key], list):
                md[key] = str(md[key])
            if key not in self.df:
                new_col = ['None'] * max(1, len(self.df))
                self.df.insert(0, key, new_col)

        bools = [self.df[key] == md[key] for key in list(md.keys())]
        conditions = bools[0]
        for b in bools[1:]:
            conditions &= b
        return self.df[conditions]

    def add_row(self, merged_dict, model=None, results=None):
        md = merged_dict.copy()
        for k in md.keys():
            if isinstance(md[k], list):
                md[k] = str(md[k])
        # Immediately synchronize to disk
        lock = filelock.FileLock(self.PANDAS_DATAFRAME_FILENAME + ".lock")
        with lock:
            self.load_pandas_df(lock=lock)
            if model is not None:
                md['model_filename'] = self.save_model(model)
            if results is not None:
                md['results_filename'] = self.save_results(results)
            self.df = self.df.append(md, ignore_index=True)
            self.save_df(lock=lock)
        return md['model_filename'], md['results_filename']

    def save_df(self, lock=None):
        if lock is None:
            lock = filelock.FileLock(self.PANDAS_DATAFRAME_FILENAME + ".lock")
        with lock:
            self.df.to_excel(self.PANDAS_DATAFRAME_FILENAME, index=False)
