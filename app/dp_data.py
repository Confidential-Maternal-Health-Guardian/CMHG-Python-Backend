import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.dataloader import DataLoader

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from dataclasses import dataclass

@dataclass
class FitData:
    original_train_df: pd.DataFrame
    syn_train_df: pd.DataFrame
    test_df: pd.DataFrame


class DataPerturbator:
    def __init__(self) -> None:
        dl = DataLoader()
        self.df = dl.load_df()

    def create_synthetic_data(self):
        df = self.df
        test_df = df.sample(n=100)

        train_indices = [i not in list(test_df.index) for i in df.index]
        df = df[train_indices]

        df_h = df[df['RiskLevel'] == 'high risk']
        df_m = df[df['RiskLevel'] == 'mid risk']
        df_l = df[df['RiskLevel'] == 'low risk']

        bandwidth_params = {'bandwidth': np.arange(0.01, 1, 0.05)}
        grid_search = GridSearchCV(KernelDensity(), bandwidth_params)
        grid_search.fit(df_h.drop("RiskLevel", axis=1))
        kde_h = grid_search.best_estimator_

        bandwidth_params = {'bandwidth': np.arange(0.01, 1, 0.05)}
        grid_search = GridSearchCV(KernelDensity(), bandwidth_params)
        grid_search.fit(df_m.drop("RiskLevel", axis=1))
        kde_m = grid_search.best_estimator_

        bandwidth_params = {'bandwidth': np.arange(0.01, 1, 0.05)}
        grid_search = GridSearchCV(KernelDensity(), bandwidth_params)
        grid_search.fit(df_l.drop("RiskLevel", axis=1))
        kde_l = grid_search.best_estimator_

        new_high = kde_h.sample(int(len(df_h) / 10),)
        new_mid = kde_m.sample(int(len(df_m) / 10),)
        new_low = kde_l.sample(int(len(df_l) / 10),)

        syn_values = np.vstack([np.hstack([new_high, np.repeat(2, len(new_high)).reshape(-1, 1)]), np.hstack([new_mid, np.repeat(1, len(new_mid)).reshape(-1, 1)]),np.hstack([new_low, np.repeat(0, len(new_low)).reshape(-1, 1)])])

        df.RiskLevel[df['RiskLevel'] == 'low risk'] = 0
        df.RiskLevel[df['RiskLevel'] == 'mid risk'] = 1
        df.RiskLevel[df['RiskLevel'] == 'high risk'] = 2

        newdf = pd.concat([df, pd.DataFrame(syn_values, columns=df.columns)])

        newdf.RiskLevel = newdf.RiskLevel.astype(int)
        newdf.RiskLevel = pd.Categorical(newdf.RiskLevel)

        test_df.RiskLevel[test_df['RiskLevel'] == 'low risk'] = 0
        test_df.RiskLevel[test_df['RiskLevel'] == 'mid risk'] = 1
        test_df.RiskLevel[test_df['RiskLevel'] == 'high risk'] = 2

        df.RiskLevel = pd.Categorical(df.RiskLevel)
        test_df.RiskLevel = pd.Categorical(test_df.RiskLevel)

        self.test_df = test_df
        self.df = df
        self.syn_df = newdf

    def get_data(self):
        fd = FitData(self.df, self.syn_df, self.test_df)
        #fd.original_train_df = self.df
        #fd.syn_train_df = self.syn_df
        #fd.test_df = self.test_df

        return fd
    