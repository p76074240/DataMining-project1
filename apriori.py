
import pandas as pd
import numpy as np
import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import operator

def subsets(arr):
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def Apriori(data_iter, datasetLen, minSupport, minConfidence):
    minSupport /= datasetLen
    
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()

    assocRules = dict()

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)*datasetLen)
                           for item in value])

    toRetRules = []
    if minConfidence > 0:
    #     for key, value in largeSet.items()[1:]:
        for key, value in list(largeSet.items())[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        # confidence = getSupport(item)/getSupport(element)
                        confidence = getSupport(item)/getSupport(element)
                        if confidence >= minConfidence:
                            toRetRules.append(((tuple(element), tuple(remain)),
                                            confidence))
    return toRetItems, toRetRules


def printResults(items, rules):
    print('\n------------------------ Frequent Item Set:')
    for item, support in sorted(items, key=operator.itemgetter(1)):
        print("item: %s , %.3f" % (str(item), support))
    print("\n------------------------ Rules:")
    for rule, confidence in sorted(rules, key=operator.itemgetter(1)):
        pre, post = rule
        print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))

def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = frozenset(line.split(','))
                yield record

def ReadData(dataset):
    for transation in dataset:
        record = frozenset(transation)
        yield record


def main():
    df = pd.read_csv(r'./dataset/data1_tab.data', sep='\t', header=None)
    dataset = []
    maxTransCount = np.max(df[1])
    for i in range(1, maxTransCount+1):
        dataset.append((df.loc[df[1] == i, 2]).values)

    print('\n------------------------ Transations:')
    print(dataset, '\n\n')

    minSupport = 9
    minConfidence = 0.8
    datasetLen = len(dataset)
    inFile = ReadData(dataset)
    items, rules = Apriori(inFile, datasetLen, minSupport, minConfidence)
    printResults(items, rules)

if __name__=='__main__':
    main()