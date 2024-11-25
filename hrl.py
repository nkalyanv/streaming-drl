from sb3_sac import CustomSACTrainer
from stream_ac_continuous import StreamAC
import gymnasium as gym
import argparse
import torch
import numpy as np
from normalization_wrappers import NormalizeObservation, ScaleReward
from time_wrapper import AddTimeInfo
from utils import store_transition

def train(env, manager, worker, args):

    steps = 0
    def rollout_one_episode(env, manager, worker, args, steps):
        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
            "infos": []
        }
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = worker.agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            info['TimeLimit.truncated'] = truncated
            store_transition(episode, obs, action, reward, next_obs, terminated or truncated, [info])
            done = terminated or truncated
            ep_reward += reward
            obs = next_obs
            steps += 1
        return episode, ep_reward, steps

    ep_rewards = []
    while steps < args.total_steps:
        episode, ep_reward, steps = rollout_one_episode(env, manager, worker, args, steps)
        worker.add_to_buffer([episode])
        if steps > args.learning_starts:
            worker.agent.train(args.batch_size, args.gradient_steps)
        print("Time Step: {}, Episodic Reward: {}".format(steps, ep_reward))
        ep_rewards.append(ep_reward)
    

def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = gym.make(args.env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    #Clip actions to be done for higher level. 
    env = ScaleReward(env, gamma=args.gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    args.gradient_steps = env.spec.max_episode_steps
    manager = StreamAC(n_obs=2, n_actions=3, hidden_size=128, lr=args.stream_lr, gamma=args.gamma, lamda=args.lamda, kappa_policy=args.kappa_policy, kappa_value=args.kappa_value)
    worker = CustomSACTrainer(env, goal_conditioned=False)
    train(env, manager, worker, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HRL')
    parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stream_lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--entropy_coeff_manager', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gradient_steps', type=int, default=50)
    parser.add_argument('--learning_starts', type=int, default=1000)
    args = parser.parse_args()
    main(args)

#TODO: 
# Check what action scale this stream ac is using