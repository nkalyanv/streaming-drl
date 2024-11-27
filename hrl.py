from sb3_sac import CustomSACTrainer
from stream_ac_continuous import StreamAC
import gymnasium as gym
import argparse
import torch
import numpy as np
from normalization_wrappers import NormalizeObservation, ScaleReward
from time_wrapper import AddTimeInfo
from utils import store_transition
from gymnasium.spaces import Box, Dict
import utils
import wandb
from normalization_wrappers import SampleMeanStd

def train(env, manager, worker, args, subgoal_normalizer_stats):
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="streaming_hrl", config=args)
    else:
        wandb.init(project="streaming_hrl", config=args, mode='disabled')

    def create_obs_for_worker(obs, sub_goal):
        return {'observation': obs, 'achieved_goal': obs, 'desired_goal': sub_goal}
    
    def dist(obs, sub_goal):
        return np.linalg.norm(obs - sub_goal)

    steps = 0
    def rollout_one_episode(env, manager, worker, args, steps, subgoal_normalizer_stats):

        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
            "infos": []
        }
        obs, _ = env.reset()
        sub_goal_unnormalized = manager.sample_action(obs)
        if args.normalize_subgoal:
            sub_goal, subgoal_normalizer_stats = normalize_subgoal(subgoal_normalizer_stats, sub_goal_unnormalized)
        else:
            sub_goal = sub_goal_unnormalized
        done = False
        ep_reward = 0
        start_state, lower_steps, reward_agg = obs, 0, 0
        worker_reward_fn = utils.sparse_reward if args.sparse_reward_worker else utils.dense_reward
        while not done:
            obs_worker = create_obs_for_worker(obs, sub_goal)
            action, _ = worker.agent.predict(obs_worker, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            lower_steps += 1
            reward_agg += reward
            info['TimeLimit.truncated'] = truncated
            worker_reward = worker_reward_fn(next_obs, sub_goal, info, args.goal_tol)
            next_obs_worker = create_obs_for_worker(next_obs, sub_goal)
            subgoal_reached = dist(next_obs, sub_goal) < args.goal_tol
            episode = store_transition(episode, obs_worker, action, worker_reward, next_obs_worker, subgoal_reached or truncated, info)
            done = terminated or truncated
            if done or lower_steps > args.lower_horizon or subgoal_reached:
                if steps > args.learning_starts:    
                    manager.update_params(start_state, sub_goal_unnormalized, reward_agg, next_obs, terminated, args.entropy_coeff_manager)
                sub_goal_unnormalized = manager.sample_action(next_obs)
                if args.normalize_subgoal:
                    sub_goal, subgoal_normalizer_stats = normalize_subgoal(subgoal_normalizer_stats, sub_goal_unnormalized)
                else:
                    sub_goal = sub_goal_unnormalized
                reward_agg, lower_steps = 0, 0
                start_state = next_obs
            ep_reward += reward
            obs = next_obs
            steps += 1
        return episode, ep_reward, steps

    ep_rewards = []
    while steps < args.total_steps:
        episode, ep_reward, steps = rollout_one_episode(env, manager, worker, args, steps, subgoal_normalizer_stats)
        worker.add_to_buffer([episode])
        if steps > args.learning_starts:
            worker.agent.train(gradient_steps=args.gradient_steps, batch_size=args.batch_size)
        print("Time Step: {}, Episodic Reward: {}".format(steps, ep_reward))
        ep_rewards.append(ep_reward)

        # Log the episode reward to wandb
        wandb.log({"episode_reward": ep_reward, "steps": steps})

    # Finish the wandb run
    wandb.finish()

#This is done for the worker since we're using SB3, it needs to be initialized with the right goal conditioned env.
def create_goal_conditioned_env(env, sparse_reward=True, goal_tol=0.1):

    class GoalConditionedEnv(gym.Wrapper):
        def __init__(self, env):
            super(GoalConditionedEnv, self).__init__(env)
            obs_space = env.observation_space
            self.observation_space = Dict({
                'observation': obs_space,
                'achieved_goal': obs_space,
                'desired_goal': obs_space
            })
            self.action_space = env.action_space
            self.sparse_reward = sparse_reward
            self.goal_tol = goal_tol

        def reset(self, **kwargs):
            obs, _ = self.env.reset(**kwargs)
            return {
                'observation': obs,
                'achieved_goal': obs,
                'desired_goal': obs
            }

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward = self.compute_reward(obs, action, info)
            return {
                'observation': obs,
                'achieved_goal': obs,
                'desired_goal': self.subgoal
            }, reward, terminated, truncated, info
        
        def set_goal(self, goal):
            self.subgoal = goal
        
        def compute_reward(self, achieved_goal, desired_goal, info):
            pass
        
    class GoalConditionedEnvSparse(GoalConditionedEnv):
        def compute_reward(self, achieved_goal, desired_goal, info):
            return utils.sparse_reward(achieved_goal, desired_goal, info)

    class GoalConditionedEnvDense(GoalConditionedEnv):
        def compute_reward(self, achieved_goal, desired_goal, info):
            return utils.dense_reward(achieved_goal, desired_goal, info)

    return GoalConditionedEnvSparse(env) if sparse_reward else GoalConditionedEnvDense(env)

def normalize_subgoal(obs_stats, obs):
    obs_stats.update(obs)
    return (obs - obs_stats.mean) / np.sqrt(obs_stats.var + 1e-8), obs_stats

def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = gym.make(args.env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    #Clip actions to be done for higher level. 
    env = ScaleReward(env, gamma=args.gamma)
    env = NormalizeObservation(env)
    if args.add_time_info:
        env = AddTimeInfo(env)
    subgoal_normalizer_stats = SampleMeanStd(shape=env.observation_space.shape)
    if args.gradient_steps is None:
        args.gradient_steps = env.spec.max_episode_steps
    state_dim = env.observation_space.shape[0]
    gc_env = create_goal_conditioned_env(env, sparse_reward=args.sparse_reward_worker)
    worker = CustomSACTrainer(gc_env, goal_conditioned=True)
    manager = StreamAC(n_obs=state_dim, n_actions=state_dim, hidden_size=128, lr=args.stream_lr, gamma=args.gamma, lamda=args.lamda, kappa_policy=args.kappa_policy, kappa_value=args.kappa_value)
    train(env, manager, worker, args, subgoal_normalizer_stats)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HRL')
    parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stream_lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=200_000)
    parser.add_argument('--entropy_coeff_manager', type=float, default=0.4)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gradient_steps', type=int, default=None) #If None, then it's set to the episode length
    parser.add_argument('--learning_starts', type=int, default=5000)
    parser.add_argument('--lower_horizon', type=int, default=20)
    parser.add_argument('--goal_tol', type=float, default=0.1)
    parser.add_argument('--sparse_reward_worker', default=True)
    parser.add_argument('--use_wandb', default=True)
    parser.add_argument('--normalize_subgoal', default=False)
    parser.add_argument('--add_time_info', default=False)
    args = parser.parse_args()
    main(args)

#TODO: 
# Setting add_time_info as true breaks the environment, since subgoals are given in this space, done, reward, etc needs to be modified.