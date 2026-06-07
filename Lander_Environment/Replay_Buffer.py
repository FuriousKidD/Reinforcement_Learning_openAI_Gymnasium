import collections
import random


class ReplayBuffer:
    """A class to store and sample experiences for reinforcement learning."""
    def __init__(self, capacity: int = 250000):
        """Initialize the replay buffer."""
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, terminated, truncated):
        """Add an experience to the replay buffer. Takes in the experience of a single current step, and stores it into the buffer as a tuple.
        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The next state of the environment after taking the action.
            terminated: A boolean indicating whether the episode has terminated.
            truncated: A boolean indicating whether the episode has been truncated.
        """
        experience_step = (state, action, reward, next_state, terminated, truncated)
        self.buffer.append(experience_step)

    def sample(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer.
        Args:
            batch_size: The number of experiences to sample. must be less than or equal to the current size of the buffer.
                Returns: A tuple containing batches of states, actions, rewards, next states, and done flags."""
        random_experiences = random.sample(self.buffer, batch_size) # Randomly sample a batch of experiences from the buffer.
        states_array, actions_array, rewards_array, next_states_array, terminated_array, truncated_array = zip(*random_experiences) # Unpacking the random experiences into separate arrays for states, actions, rewards, next states, and done flags.
        grouped_experiences = states_array, actions_array, rewards_array, next_states_array, terminated_array, truncated_array # Group the separate arrays into a tuple to return as a batch of experiences.
        return grouped_experiences