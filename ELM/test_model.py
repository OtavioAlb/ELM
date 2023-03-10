import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from elm import ELM

# Load Iris dataset from sklearn.datasets
iris = datasets.load_iris()
y = np.zeros((len(iris.target), 3))
y[:,0] = np.where(iris.target == 0, 1, 0)
y[:,1] = np.where(iris.target == 1, 1, 0)
y[:,2] = np.where(iris.target == 2, 1, 0)

# Create instance of ELM object
# ELM(input_size, output_size, hidden_size)
elm = ELM(iris.data.shape[1], 1, 10)

# Train test split 80:20
#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.2)

X_train = np.vstack((iris.data[0::3], iris.data[1::3]))
X_test  = iris.data[2::3]
y_train = np.vstack((y[0::3], y[1::3]))
y_test  = y[2::3]

# Train data
#elm.train(X_train, y_train.reshape(-1, 1))
elm.train(X_train, y_train)

# Make prediction from training process
y_pred = elm.predict(X_test)
y_pred = (y_pred == np.max(y_pred,axis = 1, keepdims=True))*1

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Accuracy: ', np.mean(y_test == y_pred))

