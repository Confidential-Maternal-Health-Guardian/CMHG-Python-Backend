from app.dp_data import DataPerturbator, FitData

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForest:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=30)

    def train(self, synthetic=True):
        dtp = DataPerturbator()
        dtp.create_synthetic_data()

        fd = dtp.get_data()

        if synthetic:
            self.train_df = fd.syn_train_df
        else:
            self.train_df = fd.original_train_df
        #self.original_train_df = fd.original_train_df
        #self.syn_train_df = fd.syn_train_df

        self.classifier.fit(self.train_df.drop('RiskLevel', axis=1), self.train_df.RiskLevel)


class DPRandomForest: #TODO: fix
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=30)

    def train(self, synthetic=True):
        dtp = DataPerturbator()
        dtp.create_synthetic_data()

        fd = dtp.get_data()

        if synthetic:
            self.train_df = fd.syn_train_df
        else:
            self.train_df = fd.original_train_df
        #self.original_train_df = fd.original_train_df
        #self.syn_train_df = fd.syn_train_df

        self.classifier.fit(self.train_df.drop('RiskLevel', axis=1), self.train_df.RiskLevel)
