import diffprivlib
from dp_data import DataPerturbator, FitData
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import math

DATA_BOUNDS = ([0, 5, 4, 60, 36, 7], [100, 25, 15, 250, 39.5, 95])

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

    def predict(self, data):
        return self.classifier.predict(data)[0]


class DPRandomForest: # TODO: fix
    def __init__(self, epsilon=1):
        self.classifier = diffprivlib.models.RandomForestClassifier(n_estimators=30, epsilon=epsilon, bounds=DATA_BOUNDS, shuffle=True)

    def train(self, synthetic=True):
        dtp = DataPerturbator()
        dtp.create_synthetic_data()
        
        fd = dtp.get_data()

        if synthetic:
            self.train_df = fd.syn_train_df
        else:
            self.train_df = fd.original_train_df

        self.classifier.fit(self.train_df.drop('RiskLevel', axis=1), self.train_df.RiskLevel)

    def predict(self, data):
        return self.classifier.predict(data)[0]


class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        st = str(input_dim)+"->"+str(int(hidden_dim/2))+"->"+str(hidden_dim)+"->"+str(int(hidden_dim/3))+"->"+str(output_dim)
        #print(st)
        
        self.fc1 = torch.nn.Linear(input_dim, int(hidden_dim/2))
        self.fc2 = torch.nn.Linear(int(hidden_dim/2), hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, int(hidden_dim/3))
        self.fc4 = torch.nn.Linear(int(hidden_dim/3), output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.fc3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.fc4(x)
        return x

class SGD:
    def __init__(self):
        self.neuron_param = 380
        self.lr = 10**-2
        self.batch_size = 16
        self.epoch_num = 274
        
        self.model = Net(6, self.neuron_param, 3)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self, synthetic):
        dtp = DataPerturbator()
        dtp.create_synthetic_data()

        fd = dtp.get_data()

        if synthetic:
            self.train_df = fd.syn_train_df
            self.test_df = fd.test_df
        else:
            self.train_df = fd.original_train_df
            self.test_df = fd.test_df

        X_train = torch.tensor(self.train_df.drop('RiskLevel', axis=1).values, dtype=torch.float)
        X_test = torch.tensor(self.test_df.drop('RiskLevel', axis=1).values, dtype=torch.float)
        y_train = torch.tensor(self.train_df.RiskLevel.astype('long').values, dtype=torch.long)
        y_test = torch.tensor(self.test_df.RiskLevel.astype('long').values, dtype=torch.long)

        dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch_num):
            for data, labels in dataloader_train:
                if torch.cuda.is_available():
                    data = data.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

        # testing
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in dataloader_test:
                if torch.cuda.is_available():
                    data = data.cuda()
                    labels = labels.cuda()

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            #print(f'Accuracy: {accuracy:.2f}%')
            return accuracy
    
    def predict(self, data):
        x = torch.tensor(data.values, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
        
        with torch.no_grad():
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.item()
            

class DPSGD:
    def __init__(self, epsilon, delta):
        #{'neuron_param': 358, 'learning_rate': -1, 'batch_param': 31, 'epoch_param': 233} acc 71
        self.neuron_param = 380
        self.lr = 10**-2
        self.batch_size = 16
        self.epoch_num = 274

        self.epsilon = epsilon
        self.delta = delta
        
        self.model = Net(6, self.neuron_param, 3)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.noise_scale = compute_noise_scale(self.epsilon, self.delta, self.model.parameters())

    def train(self, synthetic):
        dtp = DataPerturbator()
        dtp.create_synthetic_data()

        fd = dtp.get_data()

        if synthetic:
            self.train_df = fd.syn_train_df
            self.test_df = fd.test_df
        else:
            self.train_df = fd.original_train_df
            self.test_df = fd.test_df

        X_train = torch.tensor(self.train_df.drop('RiskLevel', axis=1).values, dtype=torch.float)
        X_test = torch.tensor(self.test_df.drop('RiskLevel', axis=1).values, dtype=torch.float)
        y_train = torch.tensor(self.train_df.RiskLevel.astype('long').values, dtype=torch.long)
        y_test = torch.tensor(self.test_df.RiskLevel.astype('long').values, dtype=torch.long)

        dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch_num):
            for data, labels in dataloader_train:
                if torch.cuda.is_available():
                    data = data.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                with torch.no_grad():
                    for param in self.model.parameters():
                        param.grad += self.noise_scale * torch.randn_like(param.grad)
                        
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.epsilon)
                self.optimizer.step()

        # testing
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in dataloader_test:
                if torch.cuda.is_available():
                    data = data.cuda()
                    labels = labels.cuda()

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            #print(f'Accuracy: {accuracy:.2f}%')
            return accuracy

    def predict(self, data):
        x = torch.tensor(data.values, dtype=torch.float)
        if torch.cuda.is_available():
            x = x.cuda()
        
        with torch.no_grad():
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.item()


def compute_noise_scale(epsilon, delta, parameters):
    num_elements = sum([param.numel() for param in parameters])
    noise_scale = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise_scale /= num_elements
    return noise_scale
