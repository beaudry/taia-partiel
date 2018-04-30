# coding=UTF-8
import numpy as np
import sys

from matplotlib import pyplot

import load_datasets
from BayesNaif import BayesNaif  # importer la classe du classifieur bayesien
from Knn import Knn  # importer la classe du Knn
from NeuralNet import NeuralNet  # importer la classe du Knn

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
    # nodes = 4

    errors = []
    for nodes in range(4, 51):
        classifierNeuralNet = NeuralNet(nbHiddenLayers=1, nbNodesInHiddenLayers=nodes)
        avgError = 0
        for fold in range(folds - 1):
            classifierNeuralNet.train(train_fold[fold], train_labels_fold[fold])
            avgError += classifierNeuralNet.test(train_fold[folds - 1], train_labels_fold[folds - 1], True)

        error = avgError / (folds - 1)
        errors.append(error)
        # print("Erreur: {0:.4f}%, nbNodes: {1:2d}".format(error * 100, nodes))

        if error < best_case.error:
            best_case.error = error
            best_case.nbNodes = nodes

    pyplot.plot(range(4, 51), errors)

    # print(
    #     "Meilleur cas pour dataset #{0}: {1:.2f}%, nbNodes: {2}".format(datasetNo, best_case.error * 100,
    #                                                                     best_case.nbNodes))

pyplot.title('Average error per number of neurons')
pyplot.legend(datasetsNames)
pyplot.savefig("images/" + datasetsNames[datasetNo] + "nodes error.png")
pyplot.show()
# print(
#     "Erreur moyenne: {0:.2f}%\n".format(np.sum([best_case.error for best_case in best_cases]) / len(best_cases) * 100))

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
        # print("Error: {0:.2f}%, nbLayers: {1}".format(error * 100, layers))

        if error < best_case.error:
            best_case.error = error
            best_case.nbLayers = layers

    pyplot.plot(range(3, 8), errors)

    # print(
    #     "Meilleur cas pour dataset #{0}: {1:.2f}%, nbNodes: {2:2d},  nbLayers: {3}".format(datasetNo,
    #                                                                                        best_case.error * 100,
    #                                                                                        best_case.nbNodes,
    #                                                                                        best_case.nbLayers))

pyplot.legend(datasetsNames)
pyplot.title("Learning curve")
pyplot.savefig(datasetsNames[datasetNo] + "layers error.png")
pyplot.show()

print(
    "Erreur moyenne: {0:.2f}%\n".format(
        np.sum([best_case.error for best_case in best_cases]) / len(best_cases) * 100))

nbEpochs = 32
for datasetNo in range(len(datasets)):
    train, train_labels, test, test_labels = datasets[datasetNo]
    best_case = best_cases[datasetNo]

    classifierNeuralNet = NeuralNet(nbHiddenLayers=best_case.nbLayers, nbNodesInHiddenLayers=best_case.nbNodes)

    for epoch in range(1, nbEpochs + 1):
        classifierNeuralNet.train(train, train_labels)
        # error = classifierNeuralNet.test(test, test_labels, True)
        #
        # # print("Error: {0:.2f}%".format(error * 100))
        #
        # if best_case.epoch is 0 or error < best_case.error:
        #     best_case.error = error
        #     best_case.epoch = epoch

    print()
    print(datasetsNames[datasetNo])
    classifierNeuralNet.test(test, test_labels)
    #
    # print(
    #     "Meilleur cas pour dataset #{0}: {1:.2f}%, nbNodes: {2:2d},  nbLayers: {3:2d}, epoch #{4:2d}".format(datasetNo,
    #                                                                                                          best_case.error * 100,
    #                                                                                                          best_case.nbNodes,
    #                                                                                                          best_case.nbLayers,
    #                                                                                                          best_case.epoch))
# print(
#     "Erreur moyenne: {0:.2f}%\n".format(np.sum([best_case.error for best_case in best_cases]) / len(best_cases) * 100))

# Initializer vos paramètres

# Initializer/instanciez vos classifieurs avec leurs paramètres
# classifierKnn = Knn()
# classifierBN = BayesNaif()

# Charger/lire les datasets
# train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.65)
# train, train_labels, test, test_labels = load_datasets.load_monks_dataset(1)
# train, train_labels, test, test_labels = load_datasets.load_monks_dataset(2)
# train, train_labels, test, test_labels = load_datasets.load_monks_dataset(3)
# train, train_labels, test, test_labels = load_datasets.load_congressional_dataset(0.5)

# Entrainez votre classifieur
# import time
# start_time = time.time()
# classifierNeuralNet.train(train, train_labels)
# classifierKnn.train(train, train_labels)
# print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# classifierBN.train(train, train_labels)
# print("--- %s seconds ---" % (time.time() - start_time))
#
#
# # Tester votre classifieur
# start_time = time.time()
# classifierKnn.test(test, test_labels)
# print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# classifierBN.test(test, test_labels)
# print("--- %s seconds ---" % (time.time() - start_time))
