from decisionTree import Tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Bagging:
    def __init__(self, data, ntree, minSize):
        self.data = data.reset_index(drop=True)
        self.ntree = ntree
        self.minSize = minSize
        self.trees = []
        self.size = len(data)
        self.train()
    
    def makeBootstrap(self):
        picks = np.random.randint(0, self.size, size=self.size)
        sample = self.data.iloc[picks].reset_index(drop=True)
        
        cols = sample.shape[1]
        X = sample.iloc[:, 0:cols - 1]
        y = sample.iloc[:, cols - 1]
        return X, y

    def train(self):
        for i in range(self.ntree):
            X, y = self.makeBootstrap()
            decisionTree = Tree(X, y, self.minSize)
            self.trees.append(decisionTree)
    
    def predict(self, instance):
        predictions = []
        
        for t in self.trees:
            predictions.append(t.predict(instance))
        
        mem = {}
        for p in predictions:
            if p not in mem:
                mem[p] = 0
            mem[p] += 1
        
        topPrediction = None
        topCount = float('-inf')
        
        for k, v in mem.items():
            if v > topCount:
                topCount = v
                topPrediction = k
        
        return topPrediction

class kFold:
    def __init__(self, data, k):
        self.data = data.reset_index(drop=True)
        self.k = k
        self.size = len(data)
        self.buckets = self.makeFolds()
        self.folds = []
        for b in self.buckets:
            self.folds.append(self.data.loc[b])
    
    def makeFolds(self):
        cols = self.data.shape[1]
        y = self.data.iloc[:, cols - 1]
        
        buckets = [[] for _ in range(self.k)]
        counter = 0
        
        yList = y.to_numpy()
        classes = y.unique()
        for cl in classes:
            indices = np.where(yList == cl)[0].tolist()
            np.random.shuffle(indices)
            
            for idx in indices:
                buckets[counter % self.k].append(idx)
                counter += 1
        
        return buckets

def evaluate(folds, ntree, minSize):
    accuracy = []
    precision = []
    recall = []
    f1Score = []
    
    for i in range(len(folds)):
        train = pd.concat(folds[:i] + folds[i+1:])
        test = folds[i].reset_index(drop=True)
        
        randomForest = Bagging(train, ntree, minSize)
        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]

        predictions = []
        for j in range(len(X_test)):
            predictions.append(randomForest.predict(X_test.iloc[j]))
            
        tp = fp = fn = tn = 0
        for pred, actual in zip(predictions, y_test):
            if actual == 1:
                if actual == pred:
                    tp += 1
                else:
                    fn += 1
            else:
                if actual == pred:
                    tn += 1
                else:
                    fp += 1
        
        acc = (tp + tn) / (tp + fp + fn + tn)
        accuracy.append(acc)
        
        prec = tp / (fp + tp) if (tp + fp) > 0 else 0
        precision.append(prec)
        
        rec = tp / (fn + tp) if (tp + fn) > 0 else 0
        recall.append(rec)
        
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1Score.append(f1)
            
    return (np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1Score))
                    
if __name__ == "__main__":
    data = pd.read_csv("datasets/loan.csv")
    
    kf = kFold(data, 5)
    for ntree in [1, 5, 10, 20, 30, 40, 50]:
        print(ntree, evaluate(kf.folds, ntree, 1))