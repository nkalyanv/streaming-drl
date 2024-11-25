import numpy as np
import torch
from typing import Dict, Optional, Union
from enum import Enum
import warnings

class GoalSelectionStrategy(Enum):
    FINAL = "final"
    EPISODE = "episode" 
    FUTURE = "future"

class HerReplayBuffer:
    """
    Hindsight Experience Replay (HER) buffer compatible with SAC implementation.
    Adapted from the original HER paper: https://arxiv.org/abs/1707.01495
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_size: int,
        action_size: int,
        goal_size: int,
        reward_fn,
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.buffer_size = buffer_size
        self.observation_size = observation_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.reward_fn = reward_fn
        self.n_sampled_goal = n_sampled_goal
        self.device = device
        
        # Convert goal_selection_strategy to enum
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = GoalSelectionStrategy(goal_selection_strategy.lower())
        else:
            self.goal_selection_strategy = goal_selection_strategy
            
        # Initialize buffers
        self.observations = np.zeros((buffer_size, observation_size), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, observation_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.goals = np.zeros((buffer_size, goal_size), dtype=np.float32)
        self.achieved_goals = np.zeros((buffer_size, goal_size), dtype=np.float32)
        self.next_achieved_goals = np.zeros((buffer_size, goal_size), dtype=np.float32)
        
        # Episode tracking
        self.ep_start_indices = []
        self.pos = 0
        self.full = False
        self.current_ep_start = 0
        
    def add(self, obs: Dict, next_obs: Dict, action: np.ndarray, reward: float, done: bool, info: Dict) -> None:
        """Add a new transition to the buffer."""
        self.observations[self.pos] = obs["observation"]
        self.next_observations[self.pos] = next_obs["observation"]
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.goals[self.pos] = obs["desired_goal"]
        self.achieved_goals[self.pos] = obs["achieved_goal"]
        self.next_achieved_goals[self.pos] = next_obs["achieved_goal"]
        
        if done:
            self.ep_start_indices.append(self.current_ep_start)
            self.current_ep_start = (self.pos + 1) % self.buffer_size
            
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
            
    def sample(self, batch_size: int, env=None) -> Dict:
        """Sample a batch of transitions with HER."""
        if self.full:
            max_pos = self.buffer_size
        else:
            max_pos = self.pos
            
        # Calculate number of virtual transitions
        n_virtual = int(batch_size * self.n_sampled_goal / (self.n_sampled_goal + 1))
        n_real = batch_size - n_virtual
        
        # Sample real transitions
        real_batch_inds = np.random.randint(0, max_pos, size=n_real)
        
        # Sample virtual transitions
        virtual_batch_inds = np.random.randint(0, max_pos, size=n_virtual)
        
        observations = np.concatenate([
            self.observations[real_batch_inds],
            self.observations[virtual_batch_inds]
        ])
        
        actions = np.concatenate([
            self.actions[real_batch_inds],
            self.actions[virtual_batch_inds]
        ])
        
        next_observations = np.concatenate([
            self.next_observations[real_batch_inds],
            self.next_observations[virtual_batch_inds]
        ])
        
        # For real transitions, use original goals and rewards
        goals = list(self.goals[real_batch_inds])
        rewards = list(self.rewards[real_batch_inds])
        dones = list(self.dones[real_batch_inds])
        
        # For virtual transitions, sample new goals and compute new rewards
        for idx in virtual_batch_inds:
            ep_start = self._get_episode_start(idx)
            ep_end = self._get_episode_end(ep_start)
            
            if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
                # Use the final achieved goal from the episode
                new_goal = self.next_achieved_goals[ep_end - 1]
            elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
                # Sample a future state from the same episode
                future_idx = np.random.randint(idx, ep_end)
                new_goal = self.next_achieved_goals[future_idx]
            else:  # EPISODE strategy
                # Sample any state from the same episode
                future_idx = np.random.randint(ep_start, ep_end)
                new_goal = self.next_achieved_goals[future_idx]
                
            goals.append(new_goal)
            
            # Compute new reward using the environment's reward function
            new_reward = self.reward_fn(
                self.next_achieved_goals[idx].copy(),
                new_goal.copy(),
                {"is_success": False}
            )
            rewards.append(new_reward)
            dones.append(self.dones[idx])
            
        return {
            "observations": observations,
            "actions": actions,
            "rewards": np.array(rewards).reshape(-1, 1),
            "next_observations": next_observations,
            "dones": np.array(dones).reshape(-1, 1),
            "goals": np.array(goals)
        }
        
    def _get_episode_start(self, pos: int) -> int:
        """Get the starting position of the episode containing pos."""
        for start_idx in reversed(self.ep_start_indices):
            if start_idx <= pos:
                return start_idx
        return 0
        
    def _get_episode_end(self, ep_start: int) -> int:
        """Get the end position of the episode starting at ep_start."""
        for start_idx in self.ep_start_indices:
            if start_idx > ep_start:
                return start_idx
        return self.pos if not self.full else self.buffer_size