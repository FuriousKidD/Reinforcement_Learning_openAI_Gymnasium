from Neural_Network_package import linearLayer as ll
from Neural_Network_package import activation_functions as af
import copy
import pickle

class MLP:
    """A Multi-Layer Perceptron (MLP) neural network. 
    The MLP consists of multiple layers of linear transformations followed by non-linear activation functions.
    Attributes:
        layers_config (list): A list of dictionaries, where each dictionary contains the configuration for a layer in the MLP, including the input size, output size, activation function, and weight initialization method.
        list_config -> [(layer1_size, layer2_size, activation1, wb_initialiser1), (layer2_size, layer3_size, activation2, wb_initialiser2), ...]    
    Methods:
    """
    def __init__(self, layers_configuration: list):
        self.layers_configuration = layers_configuration
        self.layers = [] # list of layers, where each layer is a tuple containing a LinearLayer instance and its corresponding activation function instance.

    def copy(self):
        """ Creates a deep copy of the MLP instance, including its layers and their configurations. 
        This method is useful for creating independent instances of the MLP that can be modified without affecting the original instance. 
        """
        return copy.deepcopy(self) 

    def layer_assembly(self):
        """ Assembles the layers of the MLP based on the provided configuration, creating instances of LinearLayer and corresponding activation functions for each layer in the network. 
        The method iterates through the layers_config list, extracting the input size, output size, activation function type, and weight initialization method for each layer. """

        for current_layer_config in self.layers_configuration:
            input_size, output_size, activation_function_type, wb_initialiser = current_layer_config

            # Create LinearLayer instance
            linear_layer = ll.LinearLayer(input_size, output_size, wb_initialiser)
            # Create activation function instance and store it so cached inputs survive across forward/backward calls
            activation_instance = af.ActivationFunction.get_activation_function(activation_function_type)
            self.layers.append((linear_layer, activation_instance))

    
    def forward_pass(self, input_data):
        """ Performs the forward pass through the MLP, applying the linear transformations and activation functions for each layer in the network.
        The method takes the input data from the current layer, applies the linear transformation using the weights and biases of the LinearLayer instance, and 
        then applies the specified activation function to the output of the linear transformation.
        The output of the activation function then becomes the input for the next later in the network. This process continues until the final output of the MLP is 
        produced."""

        current_input = input_data
        for current_linear_layer, current_activation in self.layers:
            linear_output = current_linear_layer.forward_pass(current_input)
            current_input = current_activation.forward_pass(linear_output)
        return current_input
    
    def backward_pass(self, loss_gradient):
        """ Performs the backward pass through the MLP, computing the gradients with respect to the weights, biases, and inputs of each layer based on the incoming gradient from the loss function. 
        The method iterates through the layers of the MLP in reverse order, starting from the output layer and moving back towards the input layer. For each layer, it computes the gradients with respect to the weights and biases of the LinearLayer instance, 
        as well as the gradient with respect to the input of the current layer, which will be passed back to the previous layer during backpropagation. 
        Args:
            loss_gradient (numpy.ndarray): The gradient of the loss with respect to the output of the MLP, which is received from the loss function during backpropagation.
        """
        
        current_gradient = loss_gradient
        # Loop backwards through layers and apply activation.backward_pass then linear.backward_pass
        for current_linear_layer, current_activation in reversed(self.layers):
            grad_after_activation = current_activation.backward_pass(current_gradient)
            current_gradient = current_linear_layer.backward_pass(grad_after_activation)

        return current_gradient

    def update_weights(self, learning_rate):
        """ Updates the weights and biases of all layers in the MLP using the computed gradients and a specified learning rate. 
        This method is typically called during the training process after the backward pass has been performed and the gradients have been computed for each layer. 
        The weights and biases of each LinearLayer instance are updated by subtracting the product of the learning rate and the corresponding gradients from their current values, which helps to minimize the loss function and improve the performance of the neural network over time.
        Args:
            learning_rate (float): The learning rate to use for updating the weights and biases, which determines the step size for each update during training.
        """
        for current_linear_layer, _ in self.layers:
            current_linear_layer.update_weights(learning_rate) # Updating the weights and biases of each LinearLayer instance in the MLP using the computed gradients and the specified learning rate.

    
    def save(self, filepath: str):
        """ Saves the MLP weights and configuration to disk """

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        """ Loads an MLP instance from disk"""

        with open(filepath, 'rb') as f:
            return pickle.load(f)