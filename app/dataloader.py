import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self):
        pass

    def load_df(self, name=os.path.join(os.path.dirname(__file__), 'Maternal Health Risk Data Set.csv')):
        df = pd.read_csv(name)

        return df
