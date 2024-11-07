import numpy as np

# Données d'entrée (x) et de sortie (y)
x_entrener = np.array([[3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [1.5, 1.5]], dtype=float)
y = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [0]], dtype=float)  # données de sortie 1 = Rouge / 0 = Bleu

# Normalisation des données d'entrée
x_entrener /= np.amax(x_entrener, axis=0)
X, xPrediction = np.split(x_entrener, [8])  # Séparation des données d'entrée pour l'entraînement et la prédiction
y, yPrediction = np.split(y, [8])  # Séparation des étiquettes correspondantes

# Définition de la classe du réseau de neurones
class Neural_Network(object):
    def __init__(self):
        # Paramètres
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # Poids (les matrices de poids W1 et W2)
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # Matrice 2x3
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # Matrice 3x1

    def forward(self, X):
        # Propagation avant à travers le réseau
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        # Fonction d'activation sigmoid
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # Dérivée de sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # Rétropropagation
        self.o_error = y - o  # Erreur en sortie
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # Application de la dérivée de sigmoid à l'erreur

        self.z2_error = self.o_delta.dot(self.W2.T)  # Erreur à z2
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # Application de la dérivée de sigmoid à z2_error

        self.W1 += X.T.dot(self.z2_delta)  # Ajustement des premiers poids (entrée --> caché)
        self.W2 += self.z2.T.dot(self.o_delta)  # Ajustement des seconds poids (caché --> sortie)

    def train(self, X, y):
        # Entraînement du NN
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        # Fonction de prédiction après l'entraînement
        print("Donnée prédite après entraînement : ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))
        if (self.forward(xPrediction) < 0.5):
            print("La fleur est BLEUE ! \n")
        else:
            print("La fleur est ROUGE ! \n")

# Création et entraînement du réseau de neurones
NN = Neural_Network()

# Entraînement du réseau de neurones (par exemple, 15000 itérations)
for i in range(15000):
    print("#" + str(i) + "\n")
    print("Valeurs d'entrées : \n" + str(X))
    print("Sortie actuelle : \n" + str(y))
    print("Sortie prédite : \n" + str(np.round(NN.forward(X), 2)))
    print("\n")
    NN.train(X, y)

# Prédiction avec le réseau de neurones après l'entraînement
NN.predict()
