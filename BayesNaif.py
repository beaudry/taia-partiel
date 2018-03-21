# coding=UTF-8
import numpy as np

class BayesNaif:

	def __init__(self, **kwargs):
		"""
		c'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.max_label = 0
		
		
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

		nb_classe = len(self.classes)
		nb_attributs = train.shape[1]

		self.moyenne_classe = np.empty((nb_classe, nb_attributs))
		self.ecart_type_classe = np.empty((nb_classe, nb_attributs))
		self.p_c = np.empty((nb_classe,))

		for classe in self.classes:
			indices = np.where(train_labels == classe)			
			self.moyenne_classe[classe] = np.mean(train[indices], axis=0)
			self.ecart_type_classe[classe] = np.std(train[indices], axis=0)
			self.p_c[classe] = indices[0].shape[0]/ float(train.shape[0])

		self.max_label = train_labels.max()

		self.test(train, train_labels)

	def predict(self, exemple, label):
		"""
		Prédire la classe d'un exemple donné en entrée
		exemple est de taille 1xm
		
		si la valeur retournée est la meme que la veleur dans label
		alors l'exemple est bien classifié, si non c'est une missclassification

		"""
		posterieures = np.zeros(len(self.classes))

		for classe in self.classes:	
			post_num = self.distribution_proba(exemple, self.moyenne_classe[classe], self.ecart_type_classe[classe])
			post_num = self.p_c[classe] * np.prod(post_num)
			posterieures[classe] = post_num

		max_post_num = np.unravel_index(np.argmax(posterieures, axis=None), posterieures.shape)[0]
		return max_post_num

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
		matrice_confusion = np.zeros((matrix_size, matrix_size), dtype=int)

		for i in range(len(test)):
			prediction = self.predict(test[i],test_labels[i])
			matrice_confusion[prediction][test_labels[i]] += 1

		print("–"*30)				
		print("Matrice de confusion : ")
		print(matrice_confusion)
		print("–"*30)		
		print("Accuracy : {0}%".format(np.trace(matrice_confusion)/float(len(test))*100))
		print("–"*30)		
		for i in range(len(self.classes)):
			print("LABEL {0}".format(i))
			print("Precision : {0}".format(matrice_confusion[i][i]/float(np.sum(matrice_confusion, axis=1)[i])))
			print("Recall : {0}".format(matrice_confusion[i][i]/float(np.sum(matrice_confusion, axis=0)[i])))
			print("–"*30)
			

	def distribution_proba(self, x,  moyenne, ecart_type):
		return (1/(np.sqrt(2*np.pi) * ecart_type**2)) * np.exp(-((x-moyenne)**2)/(2*ecart_type**2))