from decisionTree import Tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

# class to create our random forest
class Bagging:
    def __init__(self, data, ntree, minSize):
        self.data = data.reset_index(drop=True)
        self.ntree = ntree
        self.minSize = minSize
        self.trees = []
        self.size = len(data)
        self.train()
    
    # make bootstrap datasets with the same size as the original
    def makeBootstrap(self):
        # if original dataset is size n, pick n random numbers in that range
        picks = np.random.randint(0, self.size, size=self.size)
        
        # create the bootstrap dataset
        sample = self.data.iloc[picks].reset_index(drop=True)
        
        # return the dataset and label column
        cols = sample.shape[1]
        X = sample.iloc[:, 0:cols - 1]
        y = sample.iloc[:, cols - 1]
        return X, y

    # train a tree on each bootstrap and add it to a list of trees to create our random forest
    def train(self):
        for i in range(self.ntree):
            X, y = self.makeBootstrap()
            decisionTree = Tree(X, y, self.minSize)
            self.trees.append(decisionTree)
    
    # given a new, instance get a prediction using majority voting
    def predict(self, instance):
        predictions = []
        
        # append the label returned by each tree to a list
        for t in self.trees:
            predictions.append(t.predict(instance))
        
        mem = {}
        for p in predictions:
            if p not in mem:
                mem[p] = 0
            mem[p] += 1
        
        topPrediction = None
        topCount = float('-inf')
        
        # find and return the top prediction
        for k, v in mem.items():
            if v > topCount:
                topCount = v
                topPrediction = k
        
        return topPrediction

# create k folds
class kFold:
    def __init__(self, data, k):
        self.data = data.reset_index(drop=True)
        self.k = k
        self.size = len(data)
        
        # list of indices for each fold
        self.buckets = self.makeFolds()
        
        # partition based on the indices in each bucket. Each fold is stratified
        self.folds = []
        for b in self.buckets:
            self.folds.append(self.data.loc[b])
    
    def makeFolds(self):
        cols = self.data.shape[1]
        y = self.data.iloc[:, cols - 1]
        
        # buckets holds the rows for each fold, pulled from the original dataset
        buckets = [[] for _ in range(self.k)]
        counter = 0
        
        yList = y.to_numpy()
        classes = y.unique()
        for cl in classes:
            # filter the dataset based on rows of class cl
            indices = np.where(yList == cl)[0].tolist()
            np.random.shuffle(indices)
            
            # for each index in the filtered dataset, add to each bucket in a round robin fashion
            # this is what allows it to be stratified
            for idx in indices:
                buckets[counter % self.k].append(idx)
                counter += 1
        
        return buckets

def evaluate(folds, ntree, minSize):
    accuracy = []
    precision = []
    recall = []
    f1Score = []
    
    # for each fold, keep track of various metrics such as accuracy, precision, recall, f1 score
    for i in range(len(folds)):
        train = pd.concat(folds[:i] + folds[i+1:])
        test = folds[i].reset_index(drop=True)
        
        randomForest = Bagging(train, ntree, minSize)
        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]

        predictions = []
        for j in range(len(X_test)):
            predictions.append(randomForest.predict(X_test.iloc[j]))
        
        # compare predictions for each row with the true labels
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
    
    # return the mean for each metric to see average performance 
    return (np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1Score))
                    
if __name__ == "__main__":
    outputDir = "graphs"
    os.makedirs(outputDir, exist_ok=True)

    datasets = {
        "wdbc": ("datasets/wdbc.csv", -1),
        "loan": ("datasets/loan.csv", -1),
        "raisin": ("datasets/raisin.csv", -1),
        "titanic": ("datasets/titanic.csv", 0),
    }
    ntrees = [1, 5, 10, 20, 30, 40, 50]
    metricNames = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for name, (path, labelPos) in datasets.items():
        datasetDir = os.path.join(outputDir, name)
        os.makedirs(datasetDir, exist_ok=True)
        data = pd.read_csv(path)

        # move the label column to the end so the rest of the code works as-is
        labelCol = data.columns[labelPos]
        data = data[[c for c in data.columns if c != labelCol] + [labelCol]]

        kf = kFold(data, 5)

        results = []
        for ntree in ntrees:
            scores = evaluate(kf.folds, ntree, 1)
            results.append(scores)
            print(name, ntree, scores)

        for i, metric in enumerate(metricNames):
            values = [r[i] for r in results]

            plt.figure()
            plt.plot(ntrees, values, marker="o")
            plt.xlabel("ntree")
            plt.ylabel(metric)
            plt.title(f"{metric} vs. ntree ({name})")
            plt.grid(True)

            fileName = f"{name}_{metric.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(datasetDir, fileName))
            plt.close()