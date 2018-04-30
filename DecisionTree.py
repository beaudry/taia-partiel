"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 methodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement
	* predict 	: pour prédire la classe d'un exemple donné
	* test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les méthodes train, predict et test de votre code.
"""

import numpy as np


# le nom de votre classe
# NeuralNet pour le modèle Réseaux de Neurones
# DecisionTree le modèle des arbres de decision

class DecisionTree: #nom de la class à changer

	def __init__(self, **kwargs):
		"""
		c'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		
		
	def train(self, train, train_labels): #vous pouvez rajouter d'autres attribus au besoin
		"""
		c'est la méthode qui va entrainer votre modèle,
		train est une matrice de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		
		
		------------
		
		"""
		attributes = list(range(len(train[0])))
		self.tree = Node(train, train_labels, attributes)
		y = self.tree.predict(train)
		print("correct = {}".format(sum(np.asarray(y == train_labels, dtype=int))/len(y)*100))

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""

	def test(self, test, test_labels):
		"""
		c'est la méthode qui va tester votre modèle sur les données de test
		l'argument test est une matrice de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		test_labels : est une matrice taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		Faites le test sur les données de test, et afficher :
		- la matrice de confision (confusion matrix)
		- l'accuracy (ou le taux d'erreur)
		
		Bien entendu ces tests doivent etre faits sur les données de test seulement
		
		"""

class Node:
	def __init__(self, data, labels, attributes, parent=None, value_from_parent=None):
		if np.all(labels[:] == labels[0]):
			self.label = labels[0]
			return

		if parent is not None:
			self.parent = parent
		if value_from_parent is not None:
			self.value_from_parent = value_from_parent
		
		self.subtree(data, attributes, labels)
		return
		
	def subtree(self, data, attributes, labels):
		self.find_best_attribut(data, labels, attributes)
		column_best_attribute = attributes[self.attribute]
		attr_data = data[:, column_best_attribute]

		children_attr = attributes[:]
		children_attr.remove(self.attribute)

		self.children = []
		for value in np.unique(attr_data):
			children_labels = labels[attr_data == value]			
			children_data = np.delete(data[attr_data == value,:], column_best_attribute, 1)
			self.children.append(Node(children_data, children_attr, children_labels, parent=self, value_from_parent=value))
		return

	def predict(self, data):
		if self.children is None:
			labels = self.label * np.ones(len(data))
			return labels
		
		labels = np.zeros(len(data))

		for child in self.children:
			child_attribute_value = data[:, self.attribute] == child.value_from_parent
			labels[child_attribute_value] = child.predict(data[child_attribute_value])
		
		return labels


	def find_best_attribut(self, data, labels, attributes):
		higher_gain = 0.0
		for attr in attributes:
			attr_data = data[:, attributes[attr]]
			new_gain = self.gain(attr_data, labels)
			if (new_gain > higher_gain):
				higher_gain = new_gain
				self.attribute = attr
		return

	def entropy(self, attributes):
		frequency = np.unique(attributes, return_counts=True)[1]
		prob = frequency / len(attributes)
		return -prob.dot(np.log(prob))

	def gain(self, attributes, labels):
		values, frequency = np.unique(attributes, return_counts=True)
		attributes_count = dict(zip(values, frequency))
		total_label_count = len(labels)
		summation = 0.0
		
		for value, value_count in attributes_count.items():
			summation += value_count * self.entropy(labels[attributes == value])

		return self.entropy(labels) - summation / total_label_count 
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.

	
	