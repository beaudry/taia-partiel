# coding=UTF-8
import numpy as np
import math
import heapq

class Knn: 

	def __init__(self, **kwargs):
		"""
		c'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.max_label = 0
		self.neighbors_number = 5

	def instance_distance(self, instance1, instance2):
		total_distance = 0

		for i in range(len(instance1)):
			total_distance += self.attr_distance(instance1[i], instance2[i])

		return math.sqrt(total_distance)
		
	def attr_distance(self, attr1, attr2):
		return (attr1 - attr2) ** 2

		
	def train(self, train, train_labels):
		"""
		c'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		

		
		
		------------
		Après avoir fait l'entrainement, faites maintenant le test sur 
		les données d'entrainement
		IMPORTANT : 
		Vous devez afficher ici avec la commande print() de python,
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)
		
		Bien entendu ces tests doivent etre faits sur les données d'entrainement
		nous allons faire d'autres tests sur les données de test dans la méthode test()
		"""
		self.data = train[:]
		self.labels = train_labels[:]
		self.max_label = train_labels.max()

		self.test(train, train_labels)

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""
		nearest_neighbors = []
		for i in range(self.neighbors_number):
			nearest_neighbors.append((-np.inf, None))

		for i in range(len(self.data)):
			distance = -self.instance_distance(self.data[i], exemple)
			if nearest_neighbors[0][0] < distance:
				heapq.heapreplace(nearest_neighbors, (distance, self.labels[i]))

		return int(round(np.average([neighbor[1] for neighbor in nearest_neighbors])))


	def test(self, test, test_labels):
		"""
		c'est la méthode qui va tester votre modèle sur les données de test
		l'argument test est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)

		test_labels : est une matrice numpy de taille nx1

		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire

		Faites le test sur les données de test, et afficher :
		- la matrice de confision (confusion matrix)
		- l'accuracy
		- la précision (precision)
		- le rappel (recall)

		Bien entendu ces tests doivent etre faits sur les données de test seulement

		"""
		matrix_size = max(self.max_label, test_labels.max()) + 1
		confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

		for i in range(len(test)):
			prediction = self.predict(test[i], test_labels[i])
			confusion_matrix[prediction][test_labels[i]] += 1

		print("–"*30)				
		print("Matrice de confusion :")
		print(confusion_matrix)
		print("–"*30)		
		print("Accuracy : {0}%".format(np.trace(confusion_matrix) / float(len(test)) * 100))
		print("–"*30)
		for i in range(matrix_size):
			print("LABEL {0}".format(i))
			print("Precision : {0}".format(confusion_matrix[i][i] / float(confusion_matrix.sum(axis=1)[i])))
			print("Recall : {0}".format(confusion_matrix[i][i] / float(confusion_matrix.sum(axis=0)[i])))
			print("–"*30)
