import pandas as pd
import numpy as np
import math
import random

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

# find information gain for a specific categorical column
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

    return (overallEntropy - columnEntropy, None)

# handle numerical columns and find their info gain
def handleNumericalCols(dataset, labelset, columnName, overallEntropy):
    # sort the column in increasing order and also move around the labels
    sortedCol = dataset[columnName].sort_values()
    sortedLabels = labelset.loc[sortedCol.index]
    
    # lists of only the data we want to focus on
    values = sortedCol.to_numpy()
    labels = sortedLabels.to_numpy()
    
    # create a map for left and right values
    # values on the left are <= the threshold
    # values on the right are > the threshold
    size = len(values)
    left = {"total":0}
    right = {"total":0}
    
    # start with all values on the right
    for l in labels:
        if l not in right:
            right[l] = 0
            left[l] = 0
        
        right[l] += 1
        right["total"] += 1
    
    # track the best gain value and threshold value
    bestGain = float('-inf')
    bestVal = None
    
    # loop through every row and calculate the entropy for every possible threshold value and keep the best
    for i in range(size - 1):
        right[labels[i]] -= 1
        left[labels[i]] += 1
        
        right["total"] -= 1
        left["total"] += 1
        
        if values[i] == values[i+1]:
            continue
        
        # threshold is the mean for every two unique rows
        threshold = (values[i] + values[i+1]) / 2
        avgEntropy = ((left["total"] / size) * getColumnEntropy(left)) + ((right["total"] / size) * getColumnEntropy(right))
        
        # update tracking variables if info gain is higher
        infoGain = overallEntropy - avgEntropy
        if infoGain > bestGain:
            bestGain = infoGain
            bestVal = threshold
    
    return (bestGain, bestVal)

# helper to calculate the entropy of a column for a specific label
def getColumnEntropy(mem):
    temp = 0
    for key, val in mem.items():
        if key == "total" or val == 0:
            continue
        temp += ((val / mem["total"]) * math.log2(val / mem["total"]))
    
    return -temp
           
# create a hashmap for each column with their information gain
def getInfoGainMap(dataset, labelset, m):
    # find overall entropy
    overallEntropy = findEntropy(labelset)
    gainMap = {}
    
    # pick the random columns we want to split on
    m = max(1, m)
    columns = list(dataset.columns)
    columns = random.sample(columns, min(m, len(columns)))

    # using the overall entropy, find the information gain of each column and add it to the hashmap
    for col in columns:
        if pd.api.types.is_numeric_dtype(dataset[col]):
            gain = handleNumericalCols(dataset, labelset, col, overallEntropy)
        else:
            gain = findInformationGain(dataset, labelset, col, overallEntropy)
        gainMap[col] = gain

    return gainMap

# recursively build the decision tree
class Node():
    def __init__(self, dataset, labelset, minSize, m, parentMajority=None):
        # track the current dataset and labelset
        self.dataset = dataset
        self.labelset = labelset
        self.minSize = minSize
        self.m = m

        # fields for the node so the new instance knows what to split on
        self.isLeaf = False
        self.leafLabel = None
        self.majority = None
        self.focusColumn = None
        self.children = {}
        self.isNumerical = False
        self.threshold = None

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

        # base case 1: less than or equal to minSize, then create a leaf node
        if len(labelset) <= self.minSize:
            self.isLeaf = True
            self.leafLabel = self.majority
            return
        
        # base case 2: if all labels are the same then we are at a leaf node
        uniqueLabels = labelset.unique()
        if len(uniqueLabels) == 1:
            self.isLeaf = True
            self.leafLabel = uniqueLabels[0]
            return

        # find the information gain map
        gainMap = getInfoGainMap(dataset, labelset, self.m)
        highestGainCol = None
        currGainVal = (float('-inf'), None)

        for k, v in gainMap.items():
            if v[0] > currGainVal[0]:
                currGainVal = v
                highestGainCol = k

        # base case 3: no column could split these rows/no columns are left
        if highestGainCol is None:
            self.isLeaf = True
            self.leafLabel = self.majority
            return

        # field to track the column to split on for this node
        self.focusColumn = highestGainCol
        self.threshold = currGainVal[1]
        if self.threshold != None:
            self.isNumerical = True

        if not self.isNumerical:
            # find all the unique attributes for the focus column
            uniqueAttributes = dataset[self.focusColumn].unique()
            groups = {}

            # for each unique attribute, get the indexes for those rows
            for attr in uniqueAttributes:
                groups[attr] = dataset[dataset[self.focusColumn] == attr].index.tolist()

            # based on the indexes collected, split into each child branch and drop the current focus column to create the child nodes
            for attr, indexList in groups.items():
                subDataset = dataset.loc[indexList].drop(columns=[self.focusColumn])
                subLabelset = labelset.loc[indexList]
                self.children[attr] = Node(subDataset, subLabelset, self.minSize, self.m, self.majority)
        else:
            # split on the threshold using a numpy mask so rows are picked by position
            maskList = [False] * len(self.dataset)
            temp = dataset[self.focusColumn]
            
            # if a row is <= threshold, mark it as true, otherwise false
            for i in range(len(dataset)):
                if temp.iloc[i] <= self.threshold:
                    maskList[i] = True
                    
            mask = np.array(maskList)
                
            # partition children into greater than or less than or equal to branches. Don't need to drop columns
            self.children["leq"] = Node(dataset[mask], labelset[mask], self.minSize, self.m, self.majority)
            self.children["gt"] = Node(dataset[~mask], labelset[~mask], self.minSize, self.m, self.majority)

# create a tree using the Node class
class Tree():
    def __init__(self, dataset, labelset, minSize=1):
        self.minSize = minSize
        self.numCols = dataset.shape[1]
        self.root = Node(dataset, labelset, minSize, math.ceil(math.sqrt(self.numCols)))

    # predicts a label given an instance
    def predict(self, instance):
        return self.helper(self.root, instance)

    # recursively go down the tree based on the attribute in each column
    def helper(self, node, instance):
        # base case 1: return a leaf label
        if node.isLeaf:
            return node.leafLabel

        attrValue = instance[node.focusColumn]
        
        # if numerical column, split based on the instance's value
        if node.isNumerical:
            if attrValue <= node.threshold:
                return self.helper(node.children["leq"], instance)
            return self.helper(node.children["gt"], instance)

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
    pass