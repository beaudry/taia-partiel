# coding=UTF-8
import numpy as np
import time

from matplotlib import pyplot

import load_datasets
from NeuralNet import NeuralNet
from DecisionTree import DecisionTree

# importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""


class BestCase:
    def __init__(self):
        self.error = 1
        self.nbNodes = 0
        self.nbFolds = 0
        self.nbLayers = 1
        self.epoch = 0


datasetsNames = ["Iris", "Monks 1", "Monks 2", "Monks 3", "Congressional"]

datasets = [load_datasets.load_iris_dataset(0.65),
            load_datasets.load_monks_dataset(1),
            load_datasets.load_monks_dataset(2),
            load_datasets.load_monks_dataset(3),
            load_datasets.load_congressional_dataset(0.5)]

best_cases = [BestCase(), BestCase(), BestCase(), BestCase(), BestCase()]

folds = 6

for datasetNo in range(len(datasets)):
    train, train_labels, test, test_labels = datasets[datasetNo]
    best_case = best_cases[datasetNo]
    train = train[:len(train) - len(train) % folds]
    train_labels = train_labels[:len(train_labels) - len(train_labels) % folds]

    train_fold = np.split(train, folds)
    train_labels_fold = np.split(train_labels, folds)

    errors = []
    for nodes in range(4, 51):
        classifierNeuralNet = NeuralNet(nbHiddenLayers=1, nbNodesInHiddenLayers=nodes)
        avgError = 0
        for fold in range(folds - 1):
            classifierNeuralNet.train(train_fold[fold], train_labels_fold[fold])
            avgError += classifierNeuralNet.test(train_fold[folds - 1], train_labels_fold[folds - 1], True)

        error = avgError / (folds - 1)
        errors.append(error)

        if error < best_case.error:
            best_case.error = error
            best_case.nbNodes = nodes

    pyplot.plot(range(4, 51), errors)

    print(
        "Meilleur cas pour dataset #{0}: {1:.2f}%, nbNodes: {2}".format(datasetNo, best_case.error * 100,
                                                                        best_case.nbNodes))

pyplot.title('Average error per number of neurons')
pyplot.legend(datasetsNames)
pyplot.savefig("images/neurons error.png")
pyplot.show()

for datasetNo in range(len(datasets)):
    train, train_labels, test, test_labels = datasets[datasetNo]
    best_case = best_cases[datasetNo]

    train = train[:len(train) - len(train) % folds]
    train_labels = train_labels[:len(train_labels) - len(train_labels) % folds]

    train_fold = np.split(train, folds)
    train_labels_fold = np.split(train_labels, folds)

    errors = []
    for layers in range(1, 6):
        classifierNeuralNet = NeuralNet(nbHiddenLayers=layers, nbNodesInHiddenLayers=best_case.nbNodes)

        avgError = 0
        for fold in range(folds - 1):
            classifierNeuralNet.train(train_fold[fold], train_labels_fold[fold])
            avgError += classifierNeuralNet.test(train_fold[folds - 1], train_labels_fold[folds - 1], True)

        error = avgError / (folds - 1)
        errors.append(error)

        if error < best_case.error:
            best_case.error = error
            best_case.nbLayers = layers

    pyplot.plot(range(3, 8), errors)

    print(
        "Meilleur cas pour dataset #{0}: {1:.2f}%, nbNodes: {2:2d},  nbLayers: {3}".format(datasetNo,
                                                                                           best_case.error * 100,
                                                                                           best_case.nbNodes,
                                                                                           best_case.nbLayers))

pyplot.legend(datasetsNames)
pyplot.title("Average error per number of layers")
pyplot.savefig("images/layers error.png")
pyplot.show()

nbEpochs = 32
for datasetNo in range(len(datasets)):
    train, train_labels, test, test_labels = datasets[datasetNo]
    best_case = best_cases[datasetNo]

    classifierNeuralNet = NeuralNet(nbHiddenLayers=best_case.nbLayers, nbNodesInHiddenLayers=best_case.nbNodes)

    errors = []
    for epoch in range(1, nbEpochs + 1):
        if epoch is 1:
            start_time = time.time()

        classifierNeuralNet.train(train, train_labels)

        if epoch is 1:
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

        error = classifierNeuralNet.test(test, test_labels, True)

        if epoch is 1:
            print("--- %s seconds ---" % (time.time() - start_time))

        errors.append(error)

        if best_case.epoch is 0 or error < best_case.error:
            best_case.error = error
            best_case.epoch = epoch

    pyplot.plot(range(nbEpochs), errors)
    print()
    print(datasetsNames[datasetNo])
    print("Entraînement:")
    classifierNeuralNet.test(train, train_labels)
    print("Test:")
    classifierNeuralNet.test(test, test_labels)

pyplot.title('Average error per epoch')
pyplot.legend(datasetsNames)
pyplot.savefig("images/" + datasetsNames[datasetNo] + " epochs error.png")
pyplot.show()

# Initializer vos paramètres

# Initializer/instanciez vos classifieurs avec leurs paramètres
classifierDT = DecisionTree()

# Charger/lire les datasets
train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.65)
# train, train_labels, test, test_labels = load_datasets.load_monks_dataset(1)
# train, train_labels, test, test_labels = load_datasets.load_monks_dataset(2)
# train, train_labels, test, test_labels = load_datasets.load_monks_dataset(3)
# train, train_labels, test, test_labels = load_datasets.load_congressional_dataset(0.5)

# Entrainez votre classifieur

start_time = time.time()
classifierDT.train(train, train_labels)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
classifierDT.test(test, test_labels)
print("--- %s seconds ---" % (time.time() - start_time))
