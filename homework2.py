# Homework №2
# Group 23
# Andrii Sherstiuk
import numpy as np
import matplotlib.pyplot as plt

SIZE = 100
EPOCHS = 1000
LEARNING_RATE = 0.02


# Create dataset class
class Dataset:
    # Initialize data 
    def __init__(self):
        self.x = np.random.rand(SIZE)
        self.t = self.x**3 - self.x**2 + 1
        # Plot to show how the function is supposed to look like
        plt.scatter(self.x, self.t)
        plt.ylabel('Target')
        plt.xlabel('Data')
        plt.show()

    def get_xt(self):
        return self.x, self.t

# Implement the ReLu activation function
def relu(preactivation):
    return np.maximum(0, preactivation)

def relu_derivative(preactivation):
    if preactivation > 0:  
        return 1
    elif preactivation <= 0:
        return 0

# Shuffle dataset
def shuffle_data(x,t):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    return x, t

# Create neural network layer class
class Layer:

    def __init__(self, n_units, input_units, learning_rate):
        np.random.seed(1)
        self.n_units = n_units
        self.input_units = input_units
        self.learning_rate = learning_rate
        self.weights = 2 * np.random.rand(self.input_units, self.n_units) - 1
        self.biases = np.zeros(n_units)
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    def forward_step(self,input):
        self.layer_input = input
        self.layer_preactivation = np.matmul(self.layer_input, self.weights) + self.biases
        self.layer_activation = relu(self.layer_preactivation)

        return self.layer_activation
        
    # Updates each unit’s parameters
    def backward_step(self, grad_activation):
        d_relu_preactivation = np.asarray(relu_derivative(self.layer_preactivation)) 
        d_preact_gradient_activation = np.multiply(d_relu_preactivation, grad_activation)

        layer_input_T = np.transpose(self.layer_input)

        grad_weights = np.matmul(layer_input_T, d_preact_gradient_activation)
        grad_bias = d_preact_gradient_activation
        weight_T = np.transpose(self.weights)
        grad_input = np.matmul(d_preact_gradient_activation, weight_T)

        # Updating weights and biases
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_bias

        return grad_input

# Create multiple layer perceptron class 
class MLP:

    def __init__(self, learning_rate):
    
        self.hidden_layer = Layer(10, 1, learning_rate)
        self.output_layer = Layer(1, 10, learning_rate)

    def forward_step(self, input):
        hidden_layer_output = self.hidden_layer.forward_step(input)
        out = self.output_layer.forward_step(hidden_layer_output)

        return out

    def backpropagation(self, grad_activation):
        grad_input = self.output_layer.backward_step(grad_activation)
        grad_input = self.hidden_layer.backward_step(grad_input)
    

# Training neural network
def train(mlp, x, t):
    l = np.zeros(shape=x.shape)

    for elem in range(x.shape[0]):
        y = mlp.forward_step(np.expand_dims(np.asarray([x[elem]]), axis=0))
        l[elem] = 0.5 * ((y[0][0] - t[elem])**2)
        grad_activation = y[0][0] - t[elem]
        mlp.backpropagation(grad_activation)

    return l

def main():

    # Create dataset object
    numbers_dataset = Dataset()
    # Neural netowrk data
    x, t = numbers_dataset.get_xt()
    
    mlp = MLP(LEARNING_RATE)

    mean_l = []

    for elem in range(EPOCHS):
        x, t = shuffle_data(x, t)
        losses = train(mlp, x, t)
        mean_l.append(np.mean(losses))

    # Plot the mean loss
    plt.plot(range(EPOCHS), mean_l)
    plt.xlabel('Epoch')
    plt.ylabel('Mean loss')
    plt.title('Mean loss per epoch')
    plt.show()

    # Plot the predicted y-values
    y = np.ndarray(shape = x.shape)
    for elem in range(x.shape[0]):
        y[elem] = mlp.forward_step(np.expand_dims(np.asarray([x[elem]]), axis=0))

    plt.scatter(x, y)
    plt.ylabel('Predicted y-value')
    plt.xlabel('Data')
    plt.title('Learned function')
    plt.show()

if __name__ == '__main__':
    main()

