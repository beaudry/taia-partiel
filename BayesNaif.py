# coding=UTF-8
import numpy as np

class BayesNaif:

	def __init__(self, **kwargs):
		"""
		c'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		
		
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

		self.classes = set(train_labels)
		self.moyenne = np.mean(train, axis=0)
		self.ecart_type = np.std(train, axis=0)

		nb_classe = self.classes
		nb_attributs = train.shape[1]

		self.moyenne_classe = np.empty((nb_classe, nb_attributs))
		self.ecart_type_classe = np.empty((nb_classe, nb_attributs))
		self.p_c = np.empty((nb_classe,))

		print(len(self.classes))

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