import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        pass

    def load_df(self, name="Maternal Health Risk Data Set.csv"):
        df = pd.read_csv(name)

        return df