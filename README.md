# Extreme Learning Machines

Proposed by Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2004) in the entitled arcticle "Extreme learning machine- a new learning scheme of feedforward neural networks". It is a learning algorithm for single-hidden layer feedforward neural networks (SLFNs) that randomly chooses input weights and analytically determines the output weights of SLFNs.

The authors proved in their paper that SLFNs with randomly generated hidden neurons and the output weights adjusted by regularized least squares maintain their universal approximation capacity, even without updating the hidden layer parameters, and could have more learning speed compared to the Perceptron algorithm in applications such as face classification, image segmentation and human action recognition.

In this project, we developed an ELM algorithm using different activation functions. To evaluate the algorithm, we compared with MLP regarding to the the accuracy and processing time metrics, selected neural network models on some open binary and multi-class datasets.

For the cross validation, we used the Holdout validation spliting by 80/20% in train/test datasets, as well use k-fold cross validation (k=5) computing the true error and confidence interval.

Neural network models:
- Extreme Learning Machine (ELM)
- Single Layer Perceptron (SLP)
- Multi Layer Perceptron (MLP)
- Radial-Basis-Function Network (RBF)

Datasets:
- MNIST
- NORB
- Small NORB
- Breast cancer
- Diabetes
- DNA
- Iris
- Wine
