import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import panda_gym
# Create environment (example with FetchReach)
env = gym.make('Pendulum-v1')

# Initialize SAC agent with HER
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./sac_tensorboard/"
)

# Train the agent
model.learn(total_timesteps=100_000)