import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, List
import panda_gym
from stable_baselines3.common.logger import configure


class CustomSACTrainer:
    def __init__(
        self,
        env,
        policy_kwargs: Dict = None,
        learning_starts: int = 100,
        batch_size: int = 256,
        gradient_steps: int = 50,
        replay_buffer_kwargs: Dict = None,
        goal_conditioned: bool = False,
    ):
        """
        Initialize custom SAC trainer with HER
        """
        self.env = env
        # Initialize SAC agent
        if goal_conditioned:
            self.agent = SAC(
                "MultiInputPolicy",
                env,
                learning_starts=learning_starts,
                batch_size=batch_size,
                policy_kwargs=policy_kwargs or {},
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=replay_buffer_kwargs or {
                "n_sampled_goal": 4,
                "goal_selection_strategy": "future",
            },
            tensorboard_log="./sac_tensorboard/",
            verbose=1,
            )
        else:
            self.agent = SAC(policy="MlpPolicy", env=env,
                learning_starts=learning_starts,
                batch_size=batch_size,
                policy_kwargs=policy_kwargs or {},
            )
        new_logger = configure(folder="./sac_tensorboard/", format_strings=["stdout", "csv", "tensorboard"])
        self.agent.set_logger(new_logger)

        self.batch_size = batch_size
        self.gradient_steps = gradient_steps #Episode length
        
    def collect_rollouts(self, n_episodes: int = 1) -> List[Dict]:
        """
        Collect experience from environment
        """
        episodes = []
        
        for _ in range(n_episodes):
            episode = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "next_observations": [],
                "dones": [],
                "infos": []
            }
            
            obs = self.env.reset()
            done = False
            ep_reward = 0
            ep_rewards = []
            
            while not done:
                # Get action from policy
                action, _ = self.agent.predict(obs, deterministic=False)
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Store transition
                episode["observations"].append(obs)
                episode["actions"].append(action)
                episode["rewards"].append(reward)
                episode["next_observations"].append(next_obs)
                episode["dones"].append(done)
                episode["infos"].append(info)
                ep_reward += reward
                obs = next_obs
                
            episodes.append(episode)
            ep_rewards.append(ep_reward)
            print(f"Episode Reward: {ep_reward}")
            
        return episodes, ep_rewards
    
    def add_to_buffer(self, episodes: List[Dict]):
        """
        Add episodes to replay buffer
        """
        for episode in episodes:
            for t in range(len(episode["observations"])):
                self.agent.replay_buffer.add(
                    episode["observations"][t],
                    episode["next_observations"][t],
                    episode["actions"][t],
                    episode["rewards"][t],
                    episode["dones"][t],
                    episode["infos"][t]
                )

    
    def collect_train(
        self,
        total_timesteps: int,
        n_episodes_per_collect: int = 1,
    ):
        """
        Main training loop
        """
        timesteps = 0
        ep_rewards = []
        while timesteps < total_timesteps:
            # Collect rollouts
            episodes, ep_rewards_temp = self.collect_rollouts(n_episodes_per_collect)
            ep_rewards.extend(ep_rewards_temp)
            # Add to buffer
            self.add_to_buffer(episodes)
            
            # Update timesteps
            timesteps += sum(len(episode["observations"]) for episode in episodes)
            
            # Perform gradient steps using SB3's train_step
            if self.agent.replay_buffer.size() > self.agent.learning_starts:
                self.agent.train(self.batch_size, self.gradient_steps)  # This calls _n_updates training steps internally
                    
                print(f"Timesteps: {timesteps}")
        return ep_rewards
    
# Example usage
if __name__ == "__main__":
    # Create environment (example with FetchReach)
    env = gym.make('PandaReach-v3')
    env = DummyVecEnv([lambda: env])
    # Initialize trainer
    trainer = CustomSACTrainer(
        env,
        policy_kwargs={"net_arch": [256, 256]},
        learning_starts=1000,
        batch_size=256,
        gradient_steps=50 #Length of Episode
    )
    
    # Train
    ep_rewards = trainer.collect_train(
        total_timesteps=10000,
        n_episodes_per_collect=1,
    )

    import matplotlib.pyplot as plt
    plt.plot(ep_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.savefig('reward_plot_custom.png')
    plt.show()