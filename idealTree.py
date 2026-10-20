import numpy as np
import pandas as pd
# ----------------------------------------------------------------------------
# entropy / information gain helpers
# ----------------------------------------------------------------------------

def entropyRows(counts):
    """Entropy of each row of a (groups x classes) table of label counts."""
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / totals, 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1)


def findEntropy(labelCounts):
    """Entropy of a single vector of label counts."""
    return float(entropyRows(labelCounts.reshape(1, -1))[0])


def categoricalSplit(vals, labels, numClasses, parentEntropy):
    """
    Information gain of splitting on a categorical column (one child per value).
    Returns None if the column only has one distinct value here (nothing to split).
    """
    codes, uniques = pd.factorize(vals)
    if len(uniques) < 2:
        return None

    # table[v, c] = how many rows have attribute value v and label c
    table = np.zeros((len(uniques), numClasses))
    np.add.at(table, (codes, labels), 1)

    sizes = table.sum(axis=1)
    weightedEntropy = np.sum((sizes / len(vals)) * entropyRows(table))
    return parentEntropy - weightedEntropy, None


def numericalSplit(vals, labels, numClasses, parentEntropy):
    """
    Best threshold for a numerical column using one sorted pass with running counts.
    Candidate thresholds are midpoints between consecutive DISTINCT sorted values.
    Returns (gain, threshold), or None if all values are identical.
    """
    n = len(vals)
    order = np.argsort(vals, kind="stable")
    sortedVals = vals[order]
    sortedLabels = labels[order]

    # a boundary after sorted row i is only valid if the value actually changes
    valid = sortedVals[1:] != sortedVals[:-1]
    if not valid.any():
        return None

    # running label counts: leftCounts[i] = label counts of sorted rows 0..i
    oneHot = np.zeros((n, numClasses))
    oneHot[np.arange(n), sortedLabels] = 1
    cumulative = np.cumsum(oneHot, axis=0)
    leftCounts = cumulative[:-1]
    rightCounts = cumulative[-1] - leftCounts

    nLeft = np.arange(1, n)
    nRight = n - nLeft

    weightedEntropy = (nLeft * entropyRows(leftCounts) + nRight * entropyRows(rightCounts)) / n
    gains = parentEntropy - weightedEntropy
    gains[~valid] = -np.inf

    best = int(np.argmax(gains))
    threshold = (sortedVals[best] + sortedVals[best + 1]) / 2.0

    # float safety: make sure "<= threshold" still separates the two distinct values
    if threshold >= sortedVals[best + 1]:
        threshold = sortedVals[best]

    return float(gains[best]), float(threshold)


def isNumericalColumn(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


# ----------------------------------------------------------------------------
# tree structure
# ----------------------------------------------------------------------------

class Node():
    def __init__(self, majority):
        # the most common label among the rows that reached this node
        # (also used as the prediction when this node is a leaf)
        self.majority = majority

        self.isLeaf = True
        self.focusColumn = None
        self.isNumerical = False

        # categorical split: one child per attribute value
        self.children = {}

        # numerical split: rows <= threshold go left, rows > threshold go right
        self.threshold = None
        self.left = None
        self.right = None


class Tree():
    def __init__(self, dataset, labelset, minSizeForSplit=5, maxFeatures=None,
                 numericalColumns=None, rng=None):
        """
        dataset:          DataFrame of attributes
        labelset:         Series of labels
        minSizeForSplit:  a node with fewer rows than this becomes a leaf
        maxFeatures:      if set, only this many attributes are considered at each split
                          (chosen at random) -- this is the hook for random forests
        numericalColumns: optional list of column names to treat as numerical;
                          by default, numeric dtypes are numerical and the rest categorical
        rng:              optional np.random.Generator (only used when maxFeatures is set)
        """
        self.minSizeForSplit = minSizeForSplit
        self.maxFeatures = maxFeatures
        self.rng = rng if rng is not None else np.random.default_rng()

        self.columns = list(dataset.columns)
        if numericalColumns is None:
            self.numerical = {c: isNumericalColumn(dataset[c]) for c in self.columns}
        else:
            self.numerical = {c: (c in numericalColumns) for c in self.columns}

        # work on raw arrays and POSITIONAL indices (not DataFrame index labels), so this
        # keeps working when the training set has duplicate rows, e.g. bootstrap samples
        self.data = {c: dataset[c].to_numpy() for c in self.columns}
        for c in self.columns:
            if self.numerical[c]:
                self.data[c] = self.data[c].astype(float)

        # encode labels as 0..K-1 so counting is just bincount
        self.labelCodes, self.classes = pd.factorize(labelset)
        self.numClasses = len(self.classes)

        allRows = np.arange(len(self.labelCodes))
        self.root = self.build(allRows, self.columns)

    def build(self, rows, availableColumns):
        labels = self.labelCodes[rows]
        counts = np.bincount(labels, minlength=self.numClasses)
        node = Node(self.classes[int(np.argmax(counts))])

        # base cases: pure node, too few rows, or nothing left to split on
        if counts.max() == len(rows):
            return node
        if len(rows) < self.minSizeForSplit or len(availableColumns) == 0:
            return node

        parentEntropy = findEntropy(counts)

        # decide which columns to evaluate (all of them, or a random subset of size m)
        order = list(availableColumns)
        if self.maxFeatures is not None:
            perm = self.rng.permutation(len(order))
            order = [order[i] for i in perm]

        bestGain = float("-inf")
        bestColumn = None
        bestThreshold = None
        evaluated = 0

        for col in order:
            vals = self.data[col][rows]
            if self.numerical[col]:
                result = numericalSplit(vals, labels, self.numClasses, parentEntropy)
            else:
                result = categoricalSplit(vals, labels, self.numClasses, parentEntropy)

            if result is None:
                # column can't split these rows; doesn't count toward the m features
                continue

            gain, threshold = result
            if gain > bestGain:
                bestGain = gain
                bestColumn = col
                bestThreshold = threshold

            evaluated += 1
            if self.maxFeatures is not None and evaluated >= self.maxFeatures:
                break

        # no column could split this node
        if bestColumn is None:
            return node

        node.isLeaf = False
        node.focusColumn = bestColumn
        vals = self.data[bestColumn][rows]

        if self.numerical[bestColumn]:
            node.isNumerical = True
            node.threshold = bestThreshold
            leftMask = vals <= bestThreshold
            # numerical columns stay available: they can be split again with a new threshold
            node.left = self.build(rows[leftMask], availableColumns)
            node.right = self.build(rows[~leftMask], availableColumns)
        else:
            # a categorical column is constant within each child, so drop it
            remaining = [c for c in availableColumns if c != bestColumn]
            codes, uniques = pd.factorize(vals)
            for k, value in enumerate(uniques):
                node.children[value] = self.build(rows[codes == k], remaining)

        return node

    # predicts a label given a single instance (a Series or dict keyed by column name)
    def predict(self, instance):
        node = self.root
        while not node.isLeaf:
            value = instance[node.focusColumn]
            if node.isNumerical:
                node = node.left if value <= node.threshold else node.right
            else:
                # unseen category: best guess is this node's majority label
                if value not in node.children:
                    return node.majority
                node = node.children[value]
        return node.majority

    # predicts labels for every row of a DataFrame
    def predictAll(self, dataset):
        return [self.predict(row) for row in dataset.to_dict("records")]


# get the accuracy based on a constructed decision tree
def calculateAccuracy(tree, dataset, labelset):
    predictions = tree.predictAll(dataset)
    actual = labelset.to_numpy()
    correct = sum(1 for p, a in zip(predictions, actual) if p == a)
    return correct / len(dataset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # load data
    data = pd.read_csv("datasets/loan.csv")
    rows, cols = data.shape

    # split into attributes and label columns
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1]

    trainAccuracies = []
    testAccuracies = []

    # run experiment 100 times
    for iteration in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        tree = Tree(X_train, y_train)

        trainAccuracies.append(calculateAccuracy(tree, X_train, y_train))
        testAccuracies.append(calculateAccuracy(tree, X_test, y_test))

    trainMean = np.mean(trainAccuracies)
    trainDeviation = np.std(trainAccuracies)
    testMean = np.mean(testAccuracies)
    testDeviation = np.std(testAccuracies)

    plt.figure(figsize=(6, 5))
    plt.hist(trainAccuracies, bins=15, color='skyblue', edgecolor='black')
    plt.title(f"Training Set Accuracy\nMean: {trainMean:.4f} | Std: {trainDeviation:.4f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.hist(testAccuracies, bins=15, color='salmon', edgecolor='black')
    plt.title(f"Testing Set Accuracy\nMean: {testMean:.4f} | Std: {testDeviation:.4f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.show()