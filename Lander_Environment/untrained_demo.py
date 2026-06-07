# untrained_demo.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Lander_Environment"))
from Neural_Network_package import modular_mlp as mlp
import pickle

config = [
    (8, 128, 'relu', 'xavier'),
    (128, 128, 'relu', 'xavier'),
    (128, 4, 'identity', 'xavier')
]
network = mlp.MLP(config)
network.layer_assembly()

save_path = os.path.join(os.path.dirname(__file__), 
                         "Data_lr0005", "untrained_model.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(network, f)
print("Untrained model saved.")