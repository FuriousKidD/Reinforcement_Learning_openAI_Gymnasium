import numpy as np
import modular_mlp as mlp

# Shape: (1, 8)
fake_state = np.random.randn(1, 8)
print(f"State: {fake_state}")

# input_layer -> hidden_layer1 -> hidden_layer2 -> output_layer
configuration_list = [(8, 64, 'relu', 'xavier'), (64, 64, 'relu', 'xavier'), (64, 4, 'identity', 'xavier')]
main_network = mlp.MLP(configuration_list)
target_network = main_network.copy()

main_network.layer_assembly()
output = main_network.forward_pass(fake_state)
print(f"Output: {output}")