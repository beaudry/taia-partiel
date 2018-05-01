# coding=UTF-8
import numpy as np
import math
import heapq

ALPHA = 0.01


def ReLU(value):
    # return value
    return max(0, value)


def ReLU_prime(value):
    if value > 0:
        return 1
    return 0


class Layer:
    def __init__(self, size, previous_layer_size, weightsAtZero=False):
        self.nodes = [Node(previous_layer_size, weightsAtZero) for i in range(size)]

    def setValues(self, values):
        for i in range(min(len(self), len(values))):
            self.nodes[i].value = values[i]

    def setThetas(self, expectedValues):
        for i in range(len(self)):
            self.nodes[i].setTheta(expectedValues[i])

    def forward(self, previous_layer):
        values = previous_layer.getValues()
        for i in range(len(self)):
            self.nodes[i].forward(values)

    def backward(self, next_layer):
        weights, thetas = next_layer.getWeights(), next_layer.getThetas()
        for i in range(len(self)):
            self.nodes[i].backward(weights[i], thetas)

    def updateWeights(self, previous_layer):
        # values = previous_layer.getValues()
        for i in range(len(previous_layer)):
            for j in range(len(self)):
                # print(self.getThetas()[j])
                self.nodes[j].weights[i] += ALPHA * previous_layer.getValues()[i] * self.nodes[j].theta
        self.nodes[j].bias += ALPHA * self.nodes[j].theta

    def getValues(self):
        return np.array([self.nodes[i].value for i in range(len(self))])

    def getWeights(self):
        return np.transpose([self.nodes[i].weights for i in range(len(self))])

    def getThetas(self):
        return np.array([self.nodes[i].theta for i in range(len(self))])

    def __len__(self):
        return len(self.nodes)


class Node:
    def __init__(self, nb_inputs, weightsAtZero):
        if weightsAtZero:
            self.bias = 0
            self.weights = np.zeros(nb_inputs)
        else:
            self.bias = np.random.random()
            self.weights = np.random.random(nb_inputs) / 8.1
        self.inputs = 0
        self.value = 0
        self.theta = 0

    def forward(self, *aN):
        # print(self.weights, aN)
        self.inputs = np.sum(self.weights * aN) + self.bias
        self.value = ReLU(self.inputs)

    def backward(self, weights, thetas):
        self.theta = ReLU_prime(self.inputs) * np.sum(weights * thetas)
        # print(self.inputs, weights, thetas)

    def setTheta(self, expectedValue):
        # print(self.inputs)
        self.theta = ReLU_prime(self.inputs) * (expectedValue - self.value)


class NeuralNet:
    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        np.random.seed(0)
        nb_layers = 2 + kwargs.get("nbHiddenLayers")
        nb_nodes = kwargs.get("nbNodesInHiddenLayers")
        self.weightsAtZero = kwargs.get("weightsAtZero")
        self.layers = [Layer(16, 1)]

        for hiddenLayer in range(nb_layers - 2):
            self.layers.append(Layer(nb_nodes, len(self.layers[hiddenLayer]), kwargs.get("weightsAtZero")))

        self.layers.append(Layer(1, nb_nodes))

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[len(self) - 1]

    def train(self, train, train_labels):
        """
        c'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (l e nombre de caractéristiques)

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

        # print(np.array([self.layers[i].getWeights() for i in range(len(self))]))

        for i in range(len(train)):
            # print(self.output_layer.getWeights())
            self.predict(train[i], train_labels[i])
            # print(self.predict(train[i], train_labels[i]), train_labels[i])

            self.output_layer.setThetas([train_labels[i]])

            for l in reversed(range(len(self) - 2)):
                self.layers[l].backward(self.layers[l + 1])

            for l in range(1, len(self)):
                self.layers[l].updateWeights(self.layers[l - 1])

        # print(np.array([self.layers[i].getWeights() for i in range(len(self))]))

    def predict(self, exemple, label):
        """
        Prédire la classe d'un exemple donné en entrée
        exemple est de taille 1xm

        si la valeur retournée est la meme que la veleur dans label
        alors l'exemple est bien classifié, si non c'est une missclassification

        """
        self.input_layer.setValues(exemple)
        for l in range(1, len(self)):
            self.layers[l].forward(self.layers[l - 1])

        return int(self.output_layer.getValues()[0])

    def test(self, test, test_labels, returnAverageError=False):
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
        matrix_size = np.max(test_labels) + 100
        confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        error_sum = 0

        for i in range(len(test)):
            prediction = self.predict(test[i], test_labels[i])
            if not self.weightsAtZero:
                confusion_matrix[prediction][test_labels[i]] += 1
            error_sum += (prediction - int(test_labels[i])) ** 2

        average_error = error_sum / len(test)

        if returnAverageError:
            return average_error

        print("–" * 30)
        print("Matrice de confusion :")
        print(confusion_matrix)
        print("–" * 30)
        print("Accuracy : {0:.2f}%".format(np.trace(confusion_matrix) / float(len(test)) * 100))

    def printWeights(self):
        print(np.array([layer.getWeights() for layer in self.layers]))

    def __len__(self):
        return len(self.layers)
