import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, observation_size, action_size, goal_size):
        self.buffer_size = buffer_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.observations = np.zeros((buffer_size, observation_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, observation_size), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.goals = np.zeros((buffer_size, goal_size), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, next_obs, done, goal):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.goals[self.ptr] = goal
        self.ptr = (self.ptr + 1) % self.buffer_size

    def sample(self, batch_size):
        idxs = np.random.randint(0, min(self.ptr, self.buffer_size), size=batch_size)
        return dict(
            observations=self.observations[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_observations=self.next_observations[idxs],
            dones=self.dones[idxs],
            goals=self.goals[idxs]
        )

    