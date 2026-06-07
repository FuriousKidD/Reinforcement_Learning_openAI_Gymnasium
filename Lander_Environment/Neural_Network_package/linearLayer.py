import numpy as np

class LinearLayer:
    """A class representing a linear layer in a neural network, which performs a linear transformation on the input data using weights and biases. 
    The class includes methods for initializing the weights and biases, performing the forward pass, and applying an activation function to the output of the linear transformation.
    Attributes:
        input_size (int): 
        output_size (int):
        weights (numpy.ndarray):
        biases (numpy.ndarray):
        
    Methods:
        __init__(self, input_size, output_size, wb_initialiser='xavier'):
            Initializes the LinearLayer with the specified input size (Layer), output size (Layer), activation function,
              and weight initialization method. The weights and biases are initialized based on the chosen weight initialization method.
        """
    def __init__(self, input_size, output_size, wb_initialiser='xavier'):
        self.input_size = input_size
        self.output_size = output_size
        self.wb_initialiser = wb_initialiser
        self.weights = self.initialise_weights()
        self.biases = np.zeros(output_size)
        self.cached_input = None # Cached input for use in the backward pass, which is necessary for computing gradients during backpropagation.

        self.weights_gradient = None # Gradient with respect to the weights, which will be computed during the backward pass.
        self.biases_gradient = None # Gradient with respect to the biases, which will be computed during the backward pass.

    def initialise_weights(self):
        if self.wb_initialiser == 'xavier':
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            # Initializing the weights of the linear layer using Xavier initialization, which sets the weights to random values drawn from a uniform distribution within a specific range determined by the input and output sizes of the layer.
            # This initialization method helps to maintain a healthy variance in the activations throughout the network, which can improve training performance and convergence.  
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif self.wb_initialiser == 'he':
            stddev = np.sqrt(2 / self.input_size)
            # Initializing the weights of the linear layer using He initialization, which sets the weights to random values drawn from a normal distribution with a mean of zero and a standard deviation determined by the input size of the layer.
            # This initialization method is particularly effective for layers that use ReLU activation functions, as it helps to maintain a healthy variance in the activations and can improve training performance and convergence.
            return np.random.normal(0, stddev, (self.input_size, self.output_size))
        elif self.wb_initialiser == 'random':
            # Initializing the weights of the linear layer using a simple random initialization, where the weights are set to random values drawn from a uniform distribution between -0.5 and 0.5.
            # This method is less sophisticated than Xavier or He initialization and may not perform as well in deep networks, but it can still be used for simple models or as a baseline for comparison.
            return np.random.uniform(-0.5, 0.5, (self.input_size, self.output_size))
        else:
            raise ValueError(f"Invalid weight initialization method: {self.wb_initialiser}. Choose from 'xavier', 'he', or 'random'.")

        
    def forward_pass(self, input_data):
        # caching the input data for use in the backward pass, which is necessary for computing gradients during backpropagation.
        self.cached_input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward_pass(self, incoming_gradient):
        """ Performs the backward pass through the linear layer, computing the gradients with respect to the weights, biases, and input of the layer based on the incoming gradient from the next layer in the network.
        The method takes the incoming gradient from the next layer, computes the gradients with respect to the weights and biases of the current layer, and then computes the gradient with respect to the input of the current layer, 
        which will be passed back to the previous layer during backpropagation. 
        Args:
            incoming_gradient (numpy.ndarray): The gradient of the loss with respect to the output of the current layer, which is received from the next layer in the network during backpropagation.
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the current layer, which will be passed back to the previous layer during backpropagation.
        """
        # Computing the gradient with respect to the weights of the current layer. 
        self.weights_gradient = np.dot(np.array(self.cached_input).T, incoming_gradient) # This results in a matrix of the same shape as the weights, where each element represents the gradient of the loss with respect to the corresponding weight in the linear layer.
        
        # Computing the gradient with respect to the biases of the current layer.
        self.biases_gradient = np.sum(incoming_gradient, axis=0) # This results in a vector of the same shape as the biases, where each element represents the gradient of the loss with respect to the corresponding bias in the linear layer.

        # Computing the gradient with respect to the input of the current layer, which will be passed back to the previous layer during
        # backpropagation.
        gradient_to_previous_layer = np.dot(incoming_gradient, self.weights.T) # This results in a matrix of the same shape as the input to the current layer, where each element represents the gradient of the loss with respect to the corresponding input feature for each sample in the batch.

        return gradient_to_previous_layer 

    def update_weights(self, learning_rate):
        """Update weights and biases using stored gradients."""
        if self.weights_gradient is None or self.biases_gradient is None:
            return
        # Optionally normalize by batch size if gradients were computed as sums
        # If weights_gradient was computed as sum over batch, dividing by batch size stabilizes updates.
        self.weights = self.weights - learning_rate * self.weights_gradient
        self.biases = self.biases - learning_rate * self.biases_gradient

