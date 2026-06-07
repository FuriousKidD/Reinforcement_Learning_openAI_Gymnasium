import gymnasium as gym
import numpy as np

class Lunar_Environment:
    def __init__(self):
        self.agent_environment = gym.make('LunarLander-v3', render_mode='rgb_array')
    
    def reset_environment(self):
        return self.agent_environment.reset()
    
    def get_environment(self):
        return self.agent_environment
