import numpy as np
from abc import ABC as ABC, abstractmethod

class ActivationFunction(ABC):
    """
    A class representing an activation function in a neural network, which applies a non-linear transformation to the output of a linear layer.
    This class makes use of the factory design pattern to allow for easy instantiation of different types of activation functions based on a specified function type.
    Attributes:
        function_type (str): The type of activation function, such as 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax', or 'identity'.
    Methods:
        __init__(self, function_type='identity'): Initializes the ActivationFunction with the specified function type, which determines the behavior of the forward and backward pass methods.
        forward_pass(self, input_data) -> np.ndarray: An abstract method that defines the forward pass of the activation function, which applies the non-linear transformation to the input data and returns the output as a NumPy array.
        backward_pass(self, input_data) -> np.ndarray: An abstract method that defines the backward pass of the activation function, which computes the gradient of the loss with respect to the input data based on the output of the forward pass and returns the gradient as a NumPy array.
        get_activation_function(function_type): A static factory method that returns an instance of the specified activation function class based on the provided function type.
    
    """
    def __init__(self, function_type='identity'):
        self.function_type = function_type

    @abstractmethod
    def forward_pass(self, input_data) -> np.ndarray:
        pass

    @abstractmethod
    def backward_pass(self, incoming_gradient) -> np.ndarray:
        pass

    @staticmethod
    def get_activation_function(function_type):
        """ Factory method that returns an instance of the specified activation function class based on the provided function type. """
        lower_function_type = function_type.lower()
        if lower_function_type == 'relu':
            return Relu() # Returning an instance of the ReLU activation function class when the specified function type is 'relu'.
        elif lower_function_type == 'sigmoid':
            return Sigmoid() # Returning an instance of the Sigmoid activation function class when the specified function type is 'sigmoid'.
        elif lower_function_type == 'tanh':
            return Tanh()  # Returning an instance of the Tanh activation function class when the specified function type is 'tanh'.
        elif lower_function_type == 'leaky_relu':
            return LeakyRelu() # Returning an instance of the Leaky ReLU activation function class when the specified function type is 'leaky_relu'.
        elif lower_function_type == 'softmax':
            return Softmax() # Returning an instance of the Softmax activation function class when the specified function type is 'softmax'.
        elif lower_function_type == 'identity':
            return Identity() # Returning an instance of the Identity activation function class when the specified function type is 'identity'.
        else:
            raise ValueError(f"Invalid activation function type: {function_type}. Choose from 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'softmax', or 'identity'.")

class Relu(ActivationFunction):
    """" A class representing the ReLU activation function, which is a non-linear activation function that sets all negative input values to zero and leaves positive values unchanged."""
    def __init__(self):
        super().__init__('relu')
        self.cached_input = np.array([]) # Cached input for use in the backward pass, which is necessary for computing gradients during backpropagation.

    
    def forward_pass(self, input_data):
        self.cached_input = input_data # Caching the input data for use in the backward pass, which is necessary for computing gradients during backpropagation.
        return np.maximum(0, input_data)
    
    def backward_pass(self, incoming_gradient):
        return incoming_gradient * (self.cached_input > 0)

class Sigmoid(ActivationFunction):
    """" A class representing the Sigmoid activation function, which is a non-linear activation function that maps input values to a range between 0 and 1. 
        The Sigmoid function is commonly used in binary classification problems, as it can output probabilities that can be interpreted as class memberships."""
    def __init__(self):
        super().__init__('sigmoid')
        self.cached_input = np.array([]) # Cached input for use in the backward pass, which is necessary for computing gradients during backpropagation.

    def forward_pass(self, input_data):
        self.cached_input = input_data # Caching the input data for use in the backward pass, which is necessary for computing gradients during backpropagation.
        return 1 / (1 + np.exp(-input_data))
    
    def backward_pass(self, incoming_gradient):
        return incoming_gradient * self.forward_pass(self.cached_input) * (1 - self.forward_pass(self.cached_input))

class Tanh(ActivationFunction):
    """A class representing the Tanh activation function, which is a non-linear activation function that maps input values to a range between -1 and 1."""
    def __init__(self):
        super().__init__('tanh')
        self.cached_input = np.array([]) # Cached input for use in the backward pass, which is necessary for computing gradients during backpropagation.

    def forward_pass(self, input_data):
        self.cached_input = input_data # Caching the input data for use in the backward pass, which is necessary for computing gradients during backpropagation.
        return np.tanh(input_data)
    
    def backward_pass(self, incoming_gradient):
        return incoming_gradient * (1 - np.tanh(self.cached_input) ** 2)

class LeakyRelu(ActivationFunction):
    """A class representing the Leaky ReLU activation function, which is a variant of the ReLU activation function that allows for a small, non-zero gradient when the input is negative.
      This can help to mitigate the "dying ReLU" problem, where neurons can become inactive and stop learning if they consistently receive negative inputs."""
    def __init__(self, alpha=0.01):
        super().__init__('leaky_relu')
        self.alpha = alpha
        self.cached_input = np.array([]) # Cached input for use in the backward pass, which is necessary for computing gradients during backpropagation.

    def forward_pass(self, input_data):
        self.cached_input = input_data # Caching the input data for use in the backward pass, which is necessary for computing gradients during backpropagation.
        return np.where(input_data > 0, input_data, self.alpha * input_data)

    def backward_pass(self, incoming_gradient):
        return np.where(self.cached_input > 0, incoming_gradient, self.alpha * incoming_gradient) # Returning the gradient of the loss with respect to the input data for the Leaky ReLU activation function, which is computed based on the cached input and the incoming gradient from the next layer in the network during backpropagation. The gradient is computed using a piecewise function that applies different transformations to the incoming gradient based on whether the cached input is positive or negative.

class Softmax(ActivationFunction):
    """A class representing the Softmax activation function, which is a non-linear activation function that maps input values to a probability distribution over multiple classes. 
       The Softmax function is commonly used in multi-class classification problems, as it can output probabilities that can be interpreted as class membershipsv. """
    def __init__(self):
        super().__init__('softmax')
        self.cached_input = np.array([]) # Cached input for use in the backward pass, which is necessary for computing gradients during backpropagation.

    def forward_pass(self, input_data):
        self.cached_input = input_data # Caching the input data for use in the backward pass, which is necessary for computing gradients during backpropagation.
        exp_values = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def backward_pass(self, incoming_gradient):
        softmax_output = self.forward_pass(self.cached_input)
        sum_of_gradients = np.sum(incoming_gradient * softmax_output, axis=-1, keepdims=True)
        return softmax_output * (incoming_gradient - sum_of_gradients) # Returning the gradient of the loss with respect to the input data for the Softmax activation function, which is computed based on the cached input and the incoming gradient from the next layer in the network during backpropagation. The gradient is computed using the Jacobian matrix of the Softmax function, which accounts for the interactions between the different classes in the output probability distribution.

class Identity(ActivationFunction):
    """A class representing the Identity activation function, which is a linear activation function that outputs the input value unchanged. 
       The Identity function is typically used in the output layer of regression models, where the goal is to predict a continuous value."""
    def __init__(self):
        super().__init__('identity')
        self.cached_input = np.array([]) # Cached input for use in the backward pass, which is necessary for computing gradients during backpropagation. 


    def forward_pass(self, input_data):
        self.cached_input = input_data # Caching the input data for use in the backward pass, which is necessary for computing gradients during backpropagation.
        return input_data
    
    def backward_pass(self, incoming_gradient):
        return incoming_gradient # Returning the gradient of the loss with respect to the input data for the Identity activation function, which is simply the incoming gradient from the next layer in the network during backpropagation, as the Identity function does not apply any non-linear transformation to the input data.