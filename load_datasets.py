# coding=UTF-8
import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribués à l'entrainement,
        le rest des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisé
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    f = open('datasets/bezdekIris.data', 'r')
        
    dataset = []
    
    for line in f:
        data = line.strip().split(',')
        if data != ['']:
            for i in range(4):
                data[i] = float(data[i])
            data[4] = conversion_labels[data[4]]
            dataset.append(data)

    random.shuffle(dataset)    
    data = np.array(dataset)
    ratio = round(len(data) * train_ratio)

    train = data[:ratio, :-1]
    train_labels = np.array([item[-1] for item in data[:ratio]])
    train_labels = train_labels.astype(int)    
    test = data[ratio:, :-1]
    test_labels = np.array([item[-1] for item in data[ratio:]])
    test_labels = test_labels.astype(int)        

    # REMARQUE très importante : 
	# remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
       
    
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return (train, train_labels, test, test_labels)
	
	
	
def load_congressional_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
		
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numériques 
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numérique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels = {'republican' : 0, 'democrat' : 1, 
                         'n' : 0, 'y' : 1, '?' : 2} 
    
    f = open('datasets/house-votes-84.data', 'r')

    dataset = []
    
    for line in f:
        data = line.strip().split(',')
        if data != ['']:
            for i in range(len(data)):
                data[i] = conversion_labels[data[i]]
            dataset.append(data)

    random.shuffle(dataset)
    data = np.array(dataset)
    ratio = round(len(data) * train_ratio)

    train = data[:ratio, 1:]
    train_labels = np.array([item[0] for item in data[:ratio]])
    test = data[ratio:, 1:]
    test_labels = np.array([item[0] for item in data[ratio:]])

    return (train, train_labels, test, test_labels)
	

def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks
    
    Notez bien que ce dataset est différent des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 différents sous problèmes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et 
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin 
    seulement du numéro du sous problème pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problèmes nous voulons charger (1, 2 ou 3 ?)
		par exemple, si numero_dataset=2, vous devez lire :
			le fichier monks-2.train contenant les exemples pour l'entrainement
			et le fichier monks-2.test contenant les exemples pour le test
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels
        
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les étiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
	
	
    train_dataset = open('datasets/monks-{0}.train'.format(numero_dataset))
    test_dataset = open('datasets/monks-{0}.test'.format(numero_dataset))

    datasets = []

    for f in [train_dataset, test_dataset]:
        dataset = []
        for line in f:
            data = line.strip().split(' ')

            if data != ['']:
                dataset.append(data[:-1])

        random.shuffle(dataset)
        datasets.append(np.array(dataset, dtype=int))

    train, train_labels, test, test_labels = datasets[0][:, 1:], datasets[0][:, :1], datasets[1][:, 1:], datasets[1][:, :1]

    return (train, train_labels, test, test_labels)
