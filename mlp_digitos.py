import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection


numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)

scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_neuronas = np.arange(1,21)
loss = np.zeros([1,20])
F1 = np.zeros([2,20])

for n in n_neuronas:
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                           hidden_layer_sizes=(n), 
                                           max_iter=3000)
    mlp2 = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                           hidden_layer_sizes=(n), 
                                           max_iter=3000)
    mlp.fit(X_train, Y_train)
    loss[:,n-1] = mlp.loss_
    F1[:,n-1] = sklearn.metrics.f1_score(Y_train, mlp.predict(X_train), 
      average='macro') ,sklearn.metrics.f1_score(Y_test, mlp.predict(X_test),
                                                average='macro')
    


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(n_neuronas, loss.T, label = 'train', c = 'g')
plt.legend()
plt.subplot(122)
plt.plot(n_neuronas, F1.T[:,0], label = 'train', c = 'g')
plt.plot(n_neuronas, F1.T[:,1], label = 'test', c = 'r')
plt.legend()

plt.savefig('loss_f1.png')

mejor_n_neuronas = 8

mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                       hidden_layer_sizes=(mejor_n_neuronas), 
                                       max_iter=3000)
mlp.fit(X_train, Y_train)

neuronas = mlp.coefs_[0]

fig, axes = plt.subplots(2, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(8, 8), cmap=plt.cm.RdBu)
    ax.set_xticks(())
    ax.set_yticks(())

plt.savefig('neuronas.png')