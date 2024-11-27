import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import panda_gym
from normalization_wrappers import ScaleReward, NormalizeObservation, NormalizeGoalConditionedObservation
import wandb
from wandb.integration.sb3 import WandbCallback
# Create environment (example with FetchReach)
env = gym.make('PandaReach-v3')
# env = ScaleReward(env, gamma=0.99)
env = NormalizeGoalConditionedObservation(env)

# Initialize wandb
wandb.init(project="sac_panda_reach", config={"env_name": "PandaReach-v3"}, sync_tensorboard=True)

# Initialize SAC agent with HER
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    verbose=1,
    tensorboard_log="./sac_tensorboard/",
)

# Train the agent
model.learn(total_timesteps=25_000, callback=WandbCallback())