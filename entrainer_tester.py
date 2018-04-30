# coding=UTF-8
import numpy as np
import sys
import load_datasets
from BayesNaif import BayesNaif # importer la classe du classifieur bayesien
from Knn import Knn # importer la classe du Knn
from DecisionTree import DecisionTree
#importer d'autres fichiers et classes si vous en avez développés


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initializer vos paramètres

# Initializer/instanciez vos classifieurs avec leurs paramètres
classifierKnn = Knn()
classifierBN = BayesNaif()
classifierDT = DecisionTree()

# Charger/lire les datasets
#train, train_labels, test, test_labels = load_datasets.load_iris_dataset(0.65)
train, train_labels, test, test_labels = load_datasets.load_monks_dataset(1)
# train, train_labels, test, test_labels = load_datasets.load_monks_dataset(2)
#train, train_labels, test, test_labels = load_datasets.load_monks_dataset(3)
# train, train_labels, test, test_labels = load_datasets.load_congressional_dataset(0.5)

# Entrainez votre classifieur
import time
start_time = time.time()
classifierDT.train(train, train_labels)
print("--- %s seconds ---" % (time.time() - start_time))

#start_time = time.time()
#classifierBN.train(train, train_labels)
#print("--- %s seconds ---" % (time.time() - start_time))



# Tester votre classifieur
#tart_time = time.time()
#classifierKnn.test(test, test_labels)
#print("--- %s seconds ---" % (time.time() - start_time))

#start_time = time.time()
#classifierBN.test(test, test_labels)
#print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
classifierDT.test(test, test_labels)
print("--- %s seconds ---" % (time.time() - start_time))

