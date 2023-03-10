import numpy as np


class ELM(object):

    def __init__(self, input_size, output_size, hidden_size):
        """
        Initialize weight and bias between input layer and hidden layer
        Parameters:
        inputSize: int
            The number of input layer dimensions or features in the training data
        outputSize: int
            The number of output layer dimensions
        hiddenSize: int
            The number of hidden layer dimensions        
        """

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Initialize random weight with range [-0.5, 0.5]
        self.weight = np.array(np.random.uniform(-0.5, 0.5, (self.hidden_size, self.input_size)))

        # Initialize random bias with range [0, 1]
        self.bias = np.array(np.random.uniform(0, 1, (1, self.hidden_size)))

        self.H = 0
        self.beta = 0

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function

        Parameters:
        x: array-like or matrix
            The value that the activation output will look for
        Returns:      
            The results of activation using sigmoid function
        """
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    @staticmethod
    def relu(x):
        """
        ReLu activation function

        Parameters:
        x: array-like or matrix
            The value that the activation output will look for
        Returns:
            The results of activation using ReLu function
        """
        return np.maximum(x, 0, x)

    def predict(self, X):
        """
        Predict the results of the training process using test data
        Parameters:
        X: array-like or matrix
            Test data that will be used to determine output using ELM
        Returns:
            Predicted results or outputs from test data
        """
        X = np.array(X)

        # Calculating the activation function to target
        # ŷ = H * β
        y = self.sigmoid((X @ self.weight.T) + self.bias) @ self.beta

        return y

    def train(self, X, y):
        """
        Extreme Learning Machine training process
        Parameters:
        X: array-like or matrix
            Training data that contains the value of each feature
        y: array-like or matrix
            Training data that contains the value of the target (class)
        Returns:
            The results of the training process
        """
        if type(X) != np.ndarray:
            X = np.array(X)
        if type(y) != np.array:
            y = np.array(y)

        """
        Calculate hidden layer output matrix (Hinit)
        
        The initial hidden layer output matrix is calculated by multiplying 
        X which is training data with transpose of weight matrix.
        """
        # H is output from the hidden layer
        # H  = g(Ax + b)
        self.H = (X @ self.weight.T) + self.bias

        # Sigmoid activation function
        self.H = self.sigmoid(self.H)

        # Calculate the Moore-Penrose Pseudoinverse matrix
        # H.T is the transpose of H
        # Moore-Penrose PseusoInverse H₀ = (H.Hᵗ)⁻¹.Hᵗ
#        if np.linalg.det(self.H.T @ self.H) != 0:
#            h_moore_penrose = np.linalg.inv(self.H.T @ self.H) @ self.H.T
#        elif np.linalg.det(self.H @ self.H.T) != 0:
#            h_moore_penrose = self.H.T @ np.linalg.inv(self.H @ self.H.T)
#        else:
#            h_moore_penrose = np.linalg.pinv(self.H)
        # Calculate the output weight matrix beta
        # β is bias of output layer
        # ŷ = y is target
        # β = H₀.y
#        self.beta = h_moore_penrose @ y
        self.beta = np.linalg.pinv(self.H) @ y

        # For the testing dataset, creating a new H matrix ŷ
        # return ŷ = H * β
        return self.H @ self.beta

#+==============================
# +==============================
# +==============================
# +==============================
# +==============================
    def __init__(self, input_size, output_size, hidden_size):
        """
        Initialize weight and bias between input layer and hidden layer
        Parameters:
        inputSize: int
            The number of input layer dimensions or features in the training data
        outputSize: int
            The number of output layer dimensions
        hiddenSize: int
            The number of hidden layer dimensions
        """
    
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
    
        # Initialize random weight with range [-0.5, 0.5]
        self.weight = np.array(
            np.random.uniform(-0.5, 0.5, (self.hidden_size, self.input_size)))
    
        # Initialize random bias with range [0, 1]
        self.bias = np.array(np.random.uniform(0, 1, (1, self.hidden_size)))
    
        self.H = 0
        self.beta = 0

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function

        Parameters:
        x: array-like or matrix
            The value that the activation output will look for
        Returns:
            The results of activation using sigmoid function
        """
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    @staticmethod
    def relu(x):
        """
        ReLu activation function

        Parameters:
        x: array-like or matrix
            The value that the activation output will look for
        Returns:
            The results of activation using ReLu function
        """
        return np.maximum(x, 0, x)

    def rbs_predict(X_test, weight, bias, beta):
        """
        Predict the results of the training process using test data
        Parameters:
        X: array-like or matrix
            Test data that will be used to determine output using ELM
        Returns:
            Predicted results or outputs from test data
        """
        X = np.array(X_test)
    
        # Calculating the activation function to target
        # ŷ = H * β
        y = sigmoid((X_test @ weight.T) + bias) @ beta
    
        return y

    def rbs_train(X, y, input_size, hidden_size, **kwargs):
        """
        Extreme Learning Machine training process
        Parameters:
        X: array-like or matrix
            Training data that contains the value of each feature
        y: array-like or matrix
            Training data that contains the value of the target (class)
        Returns:
            The results of the training process
        """
        if type(X) != np.ndarray:
            X = np.array(X)
        if type(y) != np.array:
            y = np.array(y)
    
        """
        Calculate hidden layer output matrix (Hinit)

        The initial hidden layer output matrix is calculated by multiplying
        X which is training data with transpose of weight matrix.
        """
        # Initialize random weight with range [-0.5, 0.5]
        weight = np.array(
            np.random.uniform(-0.5, 0.5, (hidden_size, input_size)))

        # Initialize random bias with range [0, 1]
        bias = np.array(np.random.uniform(0, 1, (1, hidden_size)))

        # H is output from the hidden layer
        # H  = g(Ax + b)
        H = (X @ weight.T) + bias
    
        # Sigmoid activation function
        H = sigmoid(H)
    
        # Calculate the Moore-Penrose Pseudoinverse matrix
        # H.T is the transpose of H
        # Moore-Penrose PseusoInverse H₀ = (H.Hᵗ)⁻¹.Hᵗ
        #        if np.linalg.det(self.H.T @ self.H) != 0:
        #            h_moore_penrose = np.linalg.inv(self.H.T @ self.H) @ self.H.T
        #        elif np.linalg.det(self.H @ self.H.T) != 0:
        #            h_moore_penrose = self.H.T @ np.linalg.inv(self.H @ self.H.T)
        #        else:
        #            h_moore_penrose = np.linalg.pinv(self.H)
        # Calculate the output weight matrix beta
        # β is bias of output layer
        # ŷ = y is target
        # β = H₀.y
        #        self.beta = h_moore_penrose @ y
        beta = np.linalg.pinv(H) @ y
    
        # For the testing dataset, creating a new H matrix ŷ
        # return ŷ = H * β
        return H @ beta
