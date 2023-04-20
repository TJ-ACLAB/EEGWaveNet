chosen = [22, 9, 33, 24, 11, 124, 122, 45, 36, 104, 108, 58, 52, 62, 92, 98, 70, 83 ]
import torch
from torch import nn
from torch.optim import Adam
from sklearn.utils.class_weight import compute_class_weight as classweight
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn import CrossEntropyLoss
import time
import h5py
import os
import scipy
from architecture import Net


class trainer:
    lr = 0.0
    def __init__(self, Model, nExp, n_classes):
        self.Model = Model
        self.compiled = False

        self.X_train, self.y_train, self.X_val, self.y_val = self.get_source_data(nExp)

        self.tracker = {'train_tracker':[],'val_tracker':[]}
        print(self.y_train)

        # weights = classweight(class_weight="balanced",classes=np.arange(n_classes),y=self.y_train)
        # class_weights = torch.FloatTensor(weights).to(torch.device("mps"))
        # self.loss_func = CrossEntropyLoss(weight=class_weights)
        self.loss_func = CrossEntropyLoss()

    def get_source_data(self, nExp):        
        files = os.listdir("/Users/rikugen/repo/CausalEEG/NMED-T")
        trial = []
        label = []

        f = h5py.File("/Users/rikugen/repo/CausalEEG/behavioralRatings.mat")
        a_group_key = list(f.keys())[0]
        behave = f[a_group_key][()] 
        behave = behave.swapaxes(0, 1)
        behave = behave.swapaxes(1, 2)
        window_size = 1000
        sample_num = 16
        for file in files:
            fname = os.path.splitext(file)
            rawData = scipy.io.loadmat(os.path.join("/Users/rikugen/repo/CausalEEG/NMED-T", file))
            trigger = fname[0].split("_")[0]
            trigger = trigger.replace("song", "data")
            sub = rawData[str(trigger)]
            sub = sub.swapaxes(0, 2)
            sub = sub.swapaxes(1, 2)  
            func = lambda x: scipy.signal.resample(x, int(x.shape[0] / 4.0))
                # func = lambda x: x
            downsampled = np.swapaxes(np.array(list(func(np.swapaxes(sub[nExp, chosen], 0, 1)))), 0, 1)
            for t in range(0, downsampled.shape[1], (downsampled.shape[1]-window_size) // sample_num):
                if t+window_size > downsampled.shape[1]:
                    continue
                trial.append(downsampled[:,t:t+window_size])
                label.append(behave[int(trigger.strip("data")) - 21][nExp] / 10.0)
        trial = np.array(trial)
        label = np.array(label)

        # print(trial.shape)
        # print(label.shape)

        train_X, test_X, train_y, test_y = train_test_split(trial, label, test_size=0.2)
        # train data
        self.train_data = train_X
        self.train_label = train_y

        # self.train_data = np.expand_dims(self.train_data, axis=1)
        # print(self.train_data.shape)
        # self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label

        shuffle_num = np.random.permutation(len(self.allData))
        # print(shuffle_num)
        # print(self.allLabel.shape)
        self.allData = self.allData[shuffle_num]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        self.test_data = test_X
        self.test_label = test_y

        # self.test_data = np.expand_dims(self.test_data, axis=1)
        # print(self.test_data.shape)

        # self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label
        
        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def compile(self,learning_rate):
        self.optimizer = Adam(self.Model.parameters(), lr=learning_rate)
        self.compiled = True

    def train(self, epochs, batch_size=32, patience=10, directory='model.pt'):
        wait = 0
        best_model = self.Model
        if not self.compiled:
            raise Exception("You need to compile an optimizer first before training.")

        train_loss_tracker = []
        val_loss_tracker = []
        # print(self.X_train.shape)

        trainset = [[self.X_train[i],self.y_train[i]] for i in range(0,self.X_train.shape[0])]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        valset = [[self.X_val[i],self.y_val[i]] for i in range(0,self.X_val.shape[0])]
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

        self.Model.to(torch.device("mps"))

        for e in range(epochs):

            T0 = time.time()

            batch_train_loss = []
            acc = 0.0
            total = 0.0
            for data, target in trainloader:
                self.Model.train()
                pred = self.Model(data.float().to(torch.device("mps")))
                self.optimizer.zero_grad()
                train_loss = self.loss_func(pred, target.float().to(torch.device("mps")))
                #print(pred.shape)
                # print(target.shape)

                # _, predicted = torch.max(pred.float().t(), dim=0)
                label =  target.float().to(torch.device("mps"))
                # _, label = torch.max(target.float().t(), dim=0)
                acc = acc + r2_score(pred.detach().cpu().numpy(), label.cpu().numpy())
                total = total + 1.0
                train_loss.backward()       
                self.optimizer.step()
                batch_train_loss.append(train_loss)
            train_acc = acc / total
            final_train_loss = torch.mean(torch.tensor(batch_train_loss)) 
            Training_time = time.time()-T0
            batch_val_loss = []
            acc = 0.0
            total = 0.0
            with torch.no_grad():
                for data, target in valloader: 
                    pred = self.Model(data.float().to(torch.device("mps")))
                    val_loss = self.loss_func(pred, target.float().to(torch.device("mps")))
                    # _, predicted = torch.max(pred.float().t(), dim=0)
                    # _, label = torch.max(target.float().t(), dim=0)
                    label =  target.float().to(torch.device("mps"))
                    acc = acc + r2_score(pred.detach().cpu().numpy(), label.cpu().numpy())
                    total = total + 1.0
                    batch_val_loss.append(val_loss)
            val_acc = acc / total
  
            final_val_loss = torch.mean(torch.tensor(batch_val_loss))    

            print("Epoch Number \t: ",e, " Train Loss \t:","{:.5f}".format(final_train_loss), "Val Loss \t:","{:.5f}".format(final_val_loss))
            print("Train Acc:", "{:.5f}".format(train_acc),  " Val Acc:", "{:.5f}".format(val_acc))
            # print("Training Time \t:","{:.5f}".format(Training_time))
            
            if e>patience and val_loss.item()>=np.max(val_loss_tracker[-3:]):
                self.lr /= 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr   
            
            if e>patience:
                if val_loss.item()<=np.min(val_loss_tracker):
                    best_model = self.Model
                    torch.save(self.Model.state_dict(), directory)
                    wait = 0
                else:
                    wait += 1
            else:
                torch.save(self.Model.state_dict(), directory)
                         
                
            train_loss_tracker.append(final_train_loss)
            val_loss_tracker.append(final_val_loss)

            if wait >= patience:
                break

        self.tracker['train_tracker'] = train_loss_tracker
        self.tracker['val_tracker'] = val_loss_tracker
        self.Model = best_model

        return self.tracker


for i in range(0, 19):
    print("Subject", i)
    train = trainer( Model=Net(18, 2), nExp=i, n_classes=2)
    train.lr = 0.01

    train.compile(learning_rate=train.lr)
    train.train(epochs=100)
    