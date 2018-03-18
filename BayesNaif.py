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

		self.classes = np.unique(train_labels)
		self.moyenne = np.mean(train, axis=0)
		self.ecart_type = np.std(train, axis=0)

		nb_classe = len(self.classes)
		nb_attributs = train.shape[1]

		self.moyenne_classe = np.empty((nb_classe, nb_attributs))
		self.ecart_type_classe = np.empty((nb_classe, nb_attributs))
		self.p_c = np.empty((nb_classe,))

		for classe in self.classes:
			indices = np.where(train_labels == classe)			
			self.moyenne_classe[int(classe)] = np.mean(train[indices], axis=0)
			self.ecart_type[int(classe)] = np.std(train[indices], axis=0)
			self.p_c[int(classe)] = indices[0].shape[0]/ float(train.shape[0])

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""
		#TODO: Understand how to predict
		evidence = self.pdf(exemple, self.moyenne, self.ecart_type)
		print(evidence)
		print(exemple)


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

	def pdf(self, x,  moyenne, ecart_type):
		return (1/(np.sqrt(2*np.pi) * ecart_type)) * np.exp(-((x-moyenne**2)/(2*ecart_type**2)))