import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

# load data
data = pd.read_csv("datasets/loan.csv")

rows, cols = data.shape

# split into attributes and label columns
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1]

# find entropy of the overall dataset
def findEntropy(labelset):
    # create a hashmap to count each unique label
    classes = labelset.unique()
    mem = {c: 0 for c in classes}

    # increment each specific label's counter
    for i in labelset.index:
        label = labelset[i]
        mem[label] += 1

    # find overall entropy using the entropy equation
    overallEntropy = 0
    numRows = len(labelset)
    for k, v in mem.items():
        overallEntropy += ((v / numRows) * math.log2(v / numRows))

    return -1 * overallEntropy

# find information gain for a specific column
def findInformationGain(dataset, labelset, columnName, overallEntropy):
    # find the unique classes in the labelset
    classes = labelset.unique()
    numRows = len(labelset)

    # make a hashmap for each unique attribute in the column
    attributeClasses = dataset[columnName].unique()
    attributeMem = {}

    # initialize a counter for labels for each unique attribute and also keep track of the total
    for ac in attributeClasses:
        attributeMem[ac] = {c: 0 for c in classes}
        attributeMem[ac]["total"] = 0

    # increment the specific attribute while iterating through rows
    for i in dataset[columnName].index:
        attributeVal = dataset[columnName][i]
        label = labelset[i]
        attributeMem[attributeVal][label] += 1
        attributeMem[attributeVal]["total"] += 1

    # find the entropies of the partition
    columnEntropy = 0
    for v in attributeMem.values():
        temp = 0
        # calculate entropy for a specific attribute
        for key, val in v.items():
            if key == "total" or val == 0:
                continue
            temp += ((val / v["total"]) * math.log2(val / v["total"]))
        
        # find the average entripy of the resulting partitions
        temp *= -1
        columnEntropy += (v["total"] / numRows) * temp

    return overallEntropy - columnEntropy

# create a hashmap for each column with their information gain
def getInfoGainMap(dataset, labelset):
    # find overall entropy
    overallEntropy = findEntropy(labelset)
    gainMap = {}

    # using the overall entropy, find the information gain of each column and add it to the hashmap
    for col in dataset.columns:
        gain = findInformationGain(dataset, labelset, col, overallEntropy)
        gainMap[col] = gain

    return gainMap

# recursively build the decision tree
class Node():
    def __init__(self, dataset, labelset, parentMajority=None):
        # track the current dataset and labelset
        self.dataset = dataset
        self.labelset = labelset

        # fields for the node so the new instance knows what to split on
        self.isLeaf = False
        self.leafLabel = None
        self.majority = None
        self.focusColumn = None
        self.children = {}

        # counter for each unique label
        labelCounts = {lbl: 0 for lbl in labelset.unique()}
        for i in labelset.index:
            labelCounts[labelset.loc[i]] += 1

        # pick the majority label
        self.majority = ""
        temp = float('-inf')
        for k, v in labelCounts.items():
            if v > temp:
                temp = v
                self.majority = k

        # base case: if all labels are the same then we are at a leaf node
        uniqueLabels = labelset.unique()
        if len(uniqueLabels) == 1:
            self.isLeaf = True
            self.leafLabel = uniqueLabels[0]
            return

        # find the information gain map
        gainMap = getInfoGainMap(dataset, labelset)
        highestGainCol = None
        currGainVal = float('-inf')

        for k, v in gainMap.items():
            if v > currGainVal:
                currGainVal = v
                highestGainCol = k

        # field to track the column to split on for this node
        self.focusColumn = highestGainCol

        # find all the unique attributes for the focus column
        uniqueAttributes = dataset[self.focusColumn].unique()
        groups = {}

        # for each unique attribute, get the indexes for those ows
        for attr in uniqueAttributes:
            groups[attr] = dataset[dataset[self.focusColumn] == attr].index.tolist()

        # based on the indexes collected, split into each child branch and drop the current focus column to create the child nodes
        for attr, indexList in groups.items():
            subDataset = dataset.loc[indexList].drop(columns=[self.focusColumn])
            subLabelset = labelset.loc[indexList]
            self.children[attr] = Node(subDataset, subLabelset, parentMajority=self.majority)

# create a tree using the Node class
class Tree():
    def __init__(self, dataset, labelset):
        self.root = Node(dataset, labelset)

    # predicts a label given an instance
    def predict(self, instance):
        return self.helper(self.root, instance)

    # recursively go down the tree based on the attribute in each column
    def helper(self, node, instance):
        # base case 1: return a leaf label
        if node.isLeaf:
            return node.leafLabel

        attrValue = instance[node.focusColumn]

        # base case 2: if we cant recurse down to the next level, just return the best guess (which is the majority label for this node)
        if attrValue not in node.children:
            return node.majority

        # recurse down the tree based on the current focus column
        return self.helper(node.children[attrValue], instance)

# get the accuracy based on a constructed decision tree
def calculateAccuracy(tree, dataset, labelset):
    # counter to track correct predictions
    correct = 0
    
    # pass in a row and check if the predicted label matches the actual label
    for i in dataset.index:
        instance = dataset.loc[i]
        prediction = tree.predict(instance)
        if prediction == labelset.loc[i]:
            correct += 1
    
    # return an accuracy percentage
    return correct / len(dataset)

if __name__ == "__main__":
    trainAccuracies = []
    testAccuracies = []

    # run experiment 100 times
    for iteration in range(100):
        # 80/20 split for each iteration with shuffle
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # create a new decision tree based on the current training set
        tree = Tree(X_train, y_train)
        
        # get the accuracy ratio for train and test dataset
        trainRatio = calculateAccuracy(tree, X_train, y_train)
        testRatio = calculateAccuracy(tree, X_test, y_test)
        
        # append to the list to have a measurement for all 100 runs
        trainAccuracies.append(trainRatio)
        testAccuracies.append(testRatio)

    # compute summary statistics
    trainMean = np.mean(trainAccuracies)
    trainDeviation = np.std(trainAccuracies)
    testMean = np.mean(testAccuracies)
    testDeviation = np.std(testAccuracies)

    # Training Set Accuracy
    plt.figure(figsize=(6, 5))
    plt.hist(trainAccuracies, bins=15, color='skyblue', edgecolor='black')
    plt.title(f"Training Set Accuracy\nMean: {trainMean:.4f} | Std: {trainDeviation:.4f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.show()


    # Testing Set Accuracy
    plt.figure(figsize=(6, 5))
    plt.hist(testAccuracies, bins=15, color='salmon', edgecolor='black')
    plt.title(f"Testing Set Accuracy\nMean: {testMean:.4f} | Std: {testDeviation:.4f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.show()