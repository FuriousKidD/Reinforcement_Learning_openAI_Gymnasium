import Lunar_Environment as le
import Replay_Buffer as rb
import numpy as np
from Neural_Network_package import modular_mlp as mlp
from Metrics import logger as lg
from Metrics import plots as plt_tools
from gymnasium.spaces.utils import flatdim
from gymnasium.spaces import Discrete
import os

class DQNAgent:
    """ A Deep Q-Network (DQN) agent for reinforcement learning, which interacts with an environment to learn optimal policies for action selection based on the Q-learning algorithm.
    The DQNAgent class includes methods for initializing the Q-networks, selecting actions using an epsilon-greedy strategy, and performing the training loop where the agent learns from its interactions with the environment by sampling experiences from a replay buffer and updating the Q-networks based on the Bellman equation.
    Attributes:
        replay_buffer (ReplayBuffer): An instance of the ReplayBuffer class to store and sample experiences for training the Q-networks.
        environment (Lunar_Environment): An instance of the Lunar_Environment class that represents the environment in which the agent interacts and learns.
        main_network (MLP): The main Q-network used for action selection during training, which is updated based on the sampled experiences from the replay buffer.
        target_network (MLP): The target Q-network used to provide stable target values for learning updates, which is periodically synchronized with the main network to ensure that the target values are based on a stable set of parameters.
        sampled_experiences (numpy.ndarray): An array to store the batch of experiences sampled from the replay buffer during training, which includes states, actions, rewards, next states, and done flags for each experience in the batch.
        synchronization_interval (int): The number of steps between synchronizing the target network with the main  network, which helps to stabilize the learning process by providing consistent target values for updates.
        learning_rate (float): The learning rate for updating the weights of the Q-networks during training, which determines the step size for each update and can affect the convergence and performance of the learning process.
        gamma (float): The discount factor for future rewards, which determines the importance of future rewards compared to immediate rewards during the training of the Q-networks, and can influence the agent's ability to learn long-term strategies.
        batch_size (int): The batch size for sampling experiences from the replay buffer during training, which determines the number of experiences used for each update to the Q-networks and can affect the stability and efficiency of the learning process.
    Methods:

    
    """
    def __init__(self, synchronization_interval=2500, learning_rate=0.001, gamma=0.98, batch_size=64, seed=42):
        self.replay_buffer = rb.ReplayBuffer()
        self.environment = le.Lunar_Environment()
        self.main_network = None
        self.target_network = None
        self.sampled_experiences = np.array([])
        self.synchronization_interval = synchronization_interval # Number of steps between synchronizing the target network with the main network.
        self.learning_rate = learning_rate # Learning rate for updating the weights of the Q-Networks during training.
        self.gamma = gamma # Discount factor for future rewards, which determines the importance of future rewards compared to immediate rewards during the training of the Q-Networks.
        self.batch_size = batch_size # Batch size for sampling experiences from the replay buffer during training

        self.batch_states = np.array([])
        self.batch_actions = np.array([])
        self.batch_rewards = np.array([])
        self.batch_next_states = np.array([])
        self.batch_terminated = np.array([])
        self.batch_truncated = np.array([])
        self.Q_current_predictions = np.array([])
        self.actual_actions_Q_values = np.array([])

        self.logger = lg.AgentLogger("Iteration", "./Data", seed)

        # self.environment.get_environment().reset()

    def Q_network_initialization(self, input_size=8, output_size=4 , num_hidden_layers=2, hidden_layer_size=(128,128)):
        """ Initializes the main and target Q-networks for the agent. 
            The networks are created based on a specified configuration of layers. 
            The main network is used for action selection during training,
            while the target network is used to provide stable target values for learning updates. 
            Both networks are initialized with the same architecture and weights to ensure that they start with the same parameters before training begins. 
            """
        configuration_list = [] # Start with the input layer configuration in the list of layer configurations for the MLP.  
        input_layer = (input_size, hidden_layer_size[0], 'relu', 'xavier') 
        configuration_list.append(input_layer)
        # Adding the hidden layer configurations to the list of layer configurations for the MLP, which includes the input size, output size, activation function type (ReLU), and weight initialization method (Xavier) for each hidden layer in the network.
        for i in range(1,num_hidden_layers):
            configuration_list.append((hidden_layer_size[i-1], hidden_layer_size[i], 'relu', 'xavier'))
            
        # Adding the output layer configuration to the list of layer configurations for the MLP, which includes the output size, activation function type (identity), and weight initialization method (Xavier).
        configuration_list.append((hidden_layer_size[num_hidden_layers-1], output_size, 'identity', 'xavier'))
        
        self.main_network = mlp.MLP(configuration_list)
        self.main_network.layer_assembly()
        self.target_network = self.main_network.copy()
        #self.target_network.layer_assembly()
        return self.main_network, self.target_network

    def track_steps_and_synchronize_networks(self, step_count):
        """ Tracks the number of steps taken during training and synchronizes the target network with the main network at specified intervals. 
            This method is called during the training loop to ensure that the target network is periodically updated with the parameters of the main network, which helps to stabilize the learning process by providing consistent target values for updates based on a stable set of parameters.
            Args:
                step_count (int): The current step count during training, which is used to determine when to synchronize the target network with the main network based on the specified synchronization interval.
        """
        if step_count % self.synchronization_interval == 0: # Check if the current step count is a multiple of the synchronization interval.
            
            if self.main_network is not None and self.target_network is not None: # Ensure that both the main network and target network have been initialized before attempting to synchronize.
                self.target_network = self.main_network.copy() # Synchronize the target network with the main network by creating a copy of the main network's parameters and architecture.
            else:
                raise ValueError("Main network and target network must be initialized before synchronization.") # Raise an error if either the main network or target network has not been initialized, as synchronization cannot occur without both networks being properly set up.
            

    def action_selection(self, current_state, network=None,epsilon=0.1):
        random_number = np.random.rand() # Generate a random number between 0 and 1 to determine whether to take a random action or an action based on the network's output.
        if random_number < epsilon: # If the random number is less than epsilon, take a random action.
            action = self.environment.get_environment().action_space.sample()
        else: # If the random number is greater than or equal to epsilon, take an action based on the network's output.
            if network is None:
                action = self.environment.get_environment().action_space.sample() # If no network is provided, take a random action.
            else:
                
                # Adding an extra dimension to the current state to match the expected input shape of the network. 
                # The network is expected to take a batch of states as input, so the current state is expanded to create a batch of one state. 
                batched_state = np.expand_dims(current_state, axis=0)
                
                # Use the Neural Network to select an action based on the current state. The network's output is expected to be a vector of action values,
                #  and the action with the highest value is selected as the action to take.
                network_output = network.forward_pass(batched_state)
                action = np.argmax(network_output) # Action with the highest Q-value from the network's output as the action to take. 
        return action

    def rolling_episode_average(self, current_episode, loss_log, episode_rewards, window_size=10):
        """ Calculates the rolling average of episode rewards and loss values over a specified window size, which can be used for monitoring the agent's learning progress and performance over time.
            Args:
                current_episode (int): The current episode number during training, which is used to determine the range of episodes to include in the rolling average calculation based on the specified window size.
                loss_log (list): A list of loss values recorded during training, which is used to calculate the rolling average of loss values over the specified window size.
                episode_rewards (list): A list of total rewards for each episode recorded during training, which is used to calculate the rolling average of episode rewards over the specified window size.
                window_size (int): The number of episodes to include in the rolling average calculation.
        """
        if (current_episode + 1) % window_size == 0: 
            
            average_reward = np.mean(episode_rewards[-window_size:]) # Calculate the rolling average of episode rewards over the last window_size episodes.
            
            # Calculate the rolling average of loss values over the last 100 updates, if there are any loss values recorded in the log; 
            # otherwise, set the average loss to 0.
            average_loss = np.mean(loss_log[-100:]) if loss_log else 0
            
            # Print the current episode number along with the calculated average reward and average loss for monitoring the agent's learning progress and performance over time.
            print(f"Episode {current_episode + 1} | Average Reward (last {window_size} episodes): {average_reward:.2f}) | Average Loss (last 100 updates): {average_loss:.4f}") 
    
    def evaluate(self, num_episodes=10):
        """ Evaluates the performance of the trained agent by running a specified number of episodes in the environment and calculating the average reward obtained across those episodes. 
            This method can be used to assess the effectiveness of the learned policy and the overall performance of the agent after training.
            Args:
                num_episodes (int): The number of episodes to run for evaluation, which determines how many times the agent will interact with the environment to calculate the average reward.
        """
        rewards = []
        for i in range(num_episodes):
            state, info = self.environment.reset_environment()
            done = False
            total_reward = 0
            while not done:
                # Selecting the actions based on the main network's output without exploration (epsilon=0; very greedy) during evaluation.
                action = self.action_selection(state, self.main_network, epsilon=0.0)
                state, reward, terminated, truncated, info = self.environment.get_environment().step(action)
                total_reward += float(reward)
                done = terminated or truncated
            
            rewards.append(total_reward)
        average_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        print(f"Average_Reward: {average_reward} | Max_Reward:{max_reward} | Min_Reward:{min_reward}" )  
        return (average_reward, max_reward, min_reward)

    
    def scheduled_decay_calculation(self, num_start, num_min=0.01, num_episode=1000, exploration_fraction=0.7):
        """ Calculates the decay rate for the epsilon value used in the epsilon-greedy action selection strategy, 
        which determines how quickly the exploration rate decreases over time during training. 
        The decay rate is calculated based on the initial epsilon value, the minimum epsilon value, the total number of episodes for training, 
        and the fraction of episodes during which exploration should occur.
        Args:
            epsilon_start (float): The initial exploration rate for epsilon-greedy action selection at the start of training.
            epsilon_min (float): The minimum exploration rate to ensure some level of exploration throughout training.
            num_episode (int): The total number of episodes for training.
            exploration_fraction (float): The fraction of episodes during which exploration should occur.  
            Returns:
                float: The calculated decay rate for epsilon.
        """
        T = int(exploration_fraction * num_episode)
        decay_rate = (num_min / num_start) ** (1 / T)
        print(f"Decay_Rate: {decay_rate:.6f}")
        return decay_rate

    
    def training_loop(self, num_episodes=1000):
        epsilon_start = 1.0 # Initial exploration rate for epsilon-greedy action selection.
        epsilon = epsilon_start
        epsilon_min = 0.05 # Minimum exploration rate to ensure some level of exploration throughout training.
        epsilon_decay = self.scheduled_decay_calculation(num_start=epsilon_start, num_min=epsilon_min, num_episode=num_episodes, exploration_fraction=0.6)
        
        lr = self.learning_rate
        lr_min = 0.00001
        # Learning rate decays slower than epsilon
        lr_decay =  self.scheduled_decay_calculation(lr, lr_min, num_episodes, exploration_fraction=0.9) 

        
        self.main_network, self.target_network = self.Q_network_initialization() # Initialize the main and target Q-networks for the agent before starting the training loop.
        total_steps = 0
        episode_rewards = [] # List to track total rewards for each episode during training, which can be used for monitoring the agent's learning progress and performance over time.  
        loss_log_training = [] # List to track the loss values during training, which can be used for monitoring the convergence of the learning process and diagnosing potential issues with the training of the Q-networks.

        best_reward = float('-inf')

        for episode in range(num_episodes):
            current_state, info = self.environment.reset_environment()
            #print(current_state)
            action_space = self.environment.get_environment().action_space
            action_space_size = action_space.n if isinstance(action_space, Discrete) else flatdim(action_space)
            action_counts = np.zeros(action_space_size, dtype=int)
            
            total_reward = 0.0
            steps_per_episode = 0
            done = False
            loss_log_episode = []
            
            # Simulating a single episode of the interaction between the agent and the environment, where the agent takes random actions until the episode ends. During each step of the episode, the experience of the current step is stored into the replay buffer as a tuple.
            while not done:
                action = self.action_selection(current_state,self.main_network, epsilon)
                #print(f"Action: {action}")
                action_counts[action] += 1
                transition = self.environment.get_environment().step(action)
                next_state, reward, terminated, truncated, info = transition # unpacking the transition tuple into its components
                self.replay_buffer.push(current_state, action, float(reward), next_state, terminated, truncated) # Storing the experience of the current step into the replay buffer as a tuple.
                #print(f"Next State: {next_state}")
                #print(f"Reward: {reward}")

                self.track_steps_and_synchronize_networks(total_steps) # Tracking the number of steps taken during training and synchronizing the target network with the main network at specified intervals to ensure stable learning updates based on a consistent set of parameters.
                # Sampling a batch of experiences from the replay buffer for the purpose of training the Q-Network, 
                # but only if the replay buffer has accumulated atleast 1000 experiences, to ensure that there is sufficient data for training updates.
                if len(self.replay_buffer.buffer) > 5000:
                    self.sampled_experiences = self.replay_buffer.sample(self.batch_size) # Sampling a batch of experiences from the replay buffer for training the Q-Network.
                    self.batch_states, self.batch_actions, self.batch_rewards, self.batch_next_states, self.batch_terminated, self.batch_truncated = self.sampled_experiences
                
                    # Predicting the current Q-values for the batch of states using the main network
                    Q_current_predictions = np.array(self.main_network.forward_pass(self.batch_states)) # Forward pass through the main network to get the current Q-values for the batch of states as a numpy array.

                    # Isolating the Q-values corresponding to the actions taken in the batch of experiences.
                    batch_len = len(self.batch_actions)
                    actions_idx = np.array(self.batch_actions, dtype=int)
                    actual_actions_Q_values = Q_current_predictions[np.arange(batch_len), actions_idx] # Extracting the Q-values corresponding to the actions taken in the batch of experiences.
                    
                    # Predicting the next Q-values for the batch of next states using the target network.
                    Q_future_predictions = self.target_network.forward_pass(self.batch_next_states) # Forward pass through the target network to get the next Q-values for the batch of next states.
                    max_Q_future_predictions_per_experience = np.max(Q_future_predictions, axis=1) # Extracting the maximum Q-value for each experience in the batch of next states, which represents the best possible future reward according to the target network.

                    # Bellman equation to calculate the target Q-values for the batch of experiences using the rewards and the maximum next Q-values, while accounting for terminal states where the future reward is zero.
                    Q_targets = self.batch_rewards + ( (np.array(self.batch_terminated) == 0) * self.gamma * max_Q_future_predictions_per_experience )
                    
                    td_errors = np.clip(actual_actions_Q_values - Q_targets, -2.0,2.0) # Calculating the temporal difference (TD) errors for the batch of experiences, which represent the difference between the predicted Q-values for the actions taken and the target Q-values calculated using the Bellman equation.
                    loss = np.mean(td_errors ** 2) # Calculating the mean squared error loss
                    loss_log_training.append(loss) # Logging the loss value for monitoring the convergence of the learning process.
                    loss_log_episode.append(loss)

                    # gradient matrix shape: (batch_size, num_actions)
                    num_actions = Q_current_predictions.shape[1]
                    gradient_matrix = np.zeros((batch_len, num_actions))
                    gradient_matrix[np.arange(batch_len), actions_idx] = td_errors # Populate gradient matrix for chosen actions only.
                    #gradient_matrix/= batch_len # Normalize by batch size to stabilize updates.
                    self.main_network.backward_pass(gradient_matrix) # Performing the backward pass through the main network using the computed gradient matrix to calculate the gradients for updating the weights of the main network.
                    self.main_network.update_weights(lr) # Updating the weights of the main network using the computed gradients and the specified learning rate.


                
                total_reward += float(reward)
                done = terminated or truncated
                #print(f"Done: {done}")
                current_state = next_state
                steps_per_episode += 1
                total_steps += 1
            
            
            # Decaying the exploration rate epsilon after each episode to encourage the agent to take more actions based on the network's output and less random actions as training progresses, 
            # while ensuring that epsilon does not fall below a specified minimum value.
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            lr = max(lr * lr_decay, lr_min)
            episode_rewards.append(total_reward) # Logging the total reward for the episode to track the agent's learning progress and performance over time.

            print(f"Episode {episode + 1} completed. | Epsilon: {epsilon:.4f}. | Total Steps(eps): {steps_per_episode}. Total Reward: {total_reward:.2f}")
            self.rolling_episode_average(episode, loss_log_training, episode_rewards) # Calculating the rolling average of episode rewards and loss values for monitoring the agent's learning progress and performance over time.
            #print(f"Replay Buffer Size: {len(self.replay_buffer.buffer)}")
            #sampled_experiences = self.replay_buffer.sample(32) # Sampling a batch of experiences from the replay buffer.

            if episode % 50 == 0:
                average_reward, max_reward, min_reward = self.evaluate(num_episodes=10) # Evaluating the performance of the trained agent every 50 episodes by running a specified number of episodes in the environment and calculating the average reward obtained across those episodes.
                self.logger.log_evaluation(episode, average_reward, max_reward, min_reward)
                best_reward = self.save_best_model(average_reward, best_reward)

            action_percentages = (action_counts / steps_per_episode).tolist()
            self.logger.log_episode(episode, total_reward, steps_per_episode, epsilon, lr, loss_log_episode, action_percentages)



            # Unpacking the sampled experiences into separate arrays for states, actions, rewards, next states, and done flags, and printing their shapes to verify the sampling process.
            #sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = sampled_experiences
            #print(f"Sampled States: {np.array(sampled_states).shape}")
            #print(f"Sampled Actions: {np.array(sampled_actions).shape}")
            #print(f"Sampled Rewards: {np.array(sampled_rewards).shape}")
            #print(f"Sampled Next States: {np.array(sampled_next_states).shape}")
            #print(f"Sampled Dones: {np.array(sampled_dones).shape}")
            #self.environment.get_environment().close()

    def save_best_model(self, evaluated_reward, best_reward, save_dir="./Data"):
        if evaluated_reward > best_reward:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"best_model_{self.logger.run_id}.pkl")
            if self.main_network is None:
                raise ValueError("Main network is not initialized")
            self.main_network.save(path)
            print(f"NEW BEST MODEL SAVED WITH EVALUATED REWARD: {evaluated_reward:.4f}")
            return evaluated_reward # Updates the current best_reward to this value
        return best_reward
    

seed = int(os.environ.get("TRAINING_SEED", 42))
num_episodes = int(os.environ.get("NUM_EPISODES", 1000))    
np.random.seed(seed)

training_agent = DQNAgent(seed=seed,)
training_agent.training_loop(num_episodes)

average, max, min = training_agent.evaluate(num_episodes=10)
print(f"Final Evaluation | Avg: {average:.2f} | Max: {max:.2f} | Min: {min:.2f}")

training_agent.logger.save()
plt_tools.generate_all_plots("./Data", training_agent.logger.run_id)