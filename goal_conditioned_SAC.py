import gymnasium as gym
import gymnasium_robotics
import torch
import torch.nn as nn
from stream_ac_continuous import initialize_weights
import torch.nn.functional as F
import numpy as np
from replay_buffers import ReplayBuffer
import argparse
from torch.distributions import Normal
import dotmap
from her_replay_buffer import HerReplayBuffer

LOG_STD_MAX = 2
LOG_STD_MIN = -5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sg_representation(obs):
    obs_observation = torch.tensor(obs['observation'], dtype=torch.float32)
    obs_desired_goal = torch.tensor(obs['desired_goal'], dtype=torch.float32)
    return torch.cat([obs_observation, obs_desired_goal], dim=-1)

class SoftQNetwork(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, init_sparse_weights=False):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        if init_sparse_weights:
            self.apply(initialize_weights)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, action_space_range=(-1.0, 1.0)):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, n_actions)
        self.fc_logstd = nn.Linear(hidden_size, n_actions)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space_range[1] - action_space_range[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space_range[1] + action_space_range[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, autotune=True, tau=0.005, alpha=0.2, action_space_range=(-1.0, 1.0)):
        super(SAC, self).__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size, action_space_range=action_space_range).to(device)
        self.qf1 = SoftQNetwork(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size).to(device)
        self.qf2 = SoftQNetwork(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size).to(device)
        self.qf1_target = SoftQNetwork(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size).to(device)
        self.qf2_target = SoftQNetwork(n_obs=n_obs, n_actions=n_actions, hidden_size=hidden_size).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.tau = tau
        if autotune:
            self.target_entropy = -n_actions
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    def sample_action_sg(self, obs):
        x = sg_representation(obs)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        action, log_prob, mean = self.policy_net.get_action(x)
        return action.squeeze().detach().cpu().numpy()

    def sample_action(self, obs):
        action, log_prob, mean = self.policy_net.get_action(torch.tensor(obs, dtype=torch.float32, device=device))
        return action, log_prob, mean

    def update_targets(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_critic(self, data):
        data = dotmap.DotMap(data)
        with torch.no_grad():
            data.next_observations = torch.tensor(data.next_observations, dtype=torch.float32, device=device)
            data.goals = torch.tensor(data.goals, dtype=torch.float32, device=device)
            data.observations = torch.tensor(data.observations, dtype=torch.float32, device=device)
            data.actions = torch.tensor(data.actions, dtype=torch.float32, device=device)
            data.rewards = torch.tensor(data.rewards, dtype=torch.float32, device=device)
            data.dones = torch.tensor(data.dones, dtype=torch.float32, device=device)
            next_obs_goal = torch.cat([data.next_observations, data.goals], dim=1)
            obs_goal = torch.cat([data.observations, data.goals], dim=1)
            next_state_actions, next_state_log_pi, _ = self.sample_action(next_obs_goal)
            qf1_next_target = self.qf1_target(next_obs_goal, next_state_actions)
            qf2_next_target = self.qf2_target(next_obs_goal, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(obs_goal, data.actions).view(-1)
        qf2_a_values = self.qf2(obs_goal, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

    def train_actor(self, data, args):
        data = dotmap.DotMap(data)
        for _ in range(
            args.policy_freq
        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
            data.observations = torch.tensor(data.observations, dtype=torch.float32, device=device)
            data.goals = torch.tensor(data.goals, dtype=torch.float32, device=device)
            obs_goal = torch.cat([data.observations, data.goals], dim=1)
            pi, log_pi, _ = self.sample_action(obs_goal)
            qf1_pi = self.qf1(obs_goal, pi)
            qf2_pi = self.qf2(obs_goal, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if args.autotune:
                with torch.no_grad():
                    _, log_pi, _ = self.sample_action(obs_goal)
                alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()


def main(args):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_size = env.observation_space['observation'].shape[0]
    goal_size = env.observation_space['desired_goal'].shape[0]
    action_size = env.action_space.shape[0]
    agent = SAC(n_obs=obs_size + goal_size, n_actions=action_size, hidden_size=128, lr=args.policy_lr, gamma=args.gamma, autotune=args.autotune, tau=args.tau, alpha=args.alpha, 
                action_space_range=(env.action_space.low, env.action_space.high))
    if args.use_her:
        rb = HerReplayBuffer(buffer_size=args.buffer_size, observation_size=obs_size, action_size=action_size, goal_size=goal_size, reward_fn=env.unwrapped.compute_reward)
    else:
        rb = ReplayBuffer(buffer_size=args.buffer_size, observation_size=obs_size, action_size=action_size, goal_size=goal_size)
    print("HERE")
    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0  # Initialize episode reward
    episode_number = 0  # Initialize episode number
    for step in range(args.total_steps):
        if step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.sample_action_sg(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            info['TimeLimit.truncated'] = True
        real_next_obs = next_obs.copy()
        if args.use_her:
            rb.add(obs, real_next_obs, action, reward, terminated or truncated, [info])
        else:
            rb.add(obs['observation'], action, reward, real_next_obs['observation'], terminated, obs['desired_goal'])
        obs = next_obs
        episode_reward += reward  # Accumulate reward

        if terminated or truncated:
            episode_number += 1  # Increment episode number
            print(f"Episode {episode_number} finished with reward: {episode_reward} at timestep {step}")  # Print episode reward and timestep
            episode_reward = 0  # Reset episode reward
            obs, _ = env.reset(seed=args.seed)  # Reset environment

        if step > args.learning_starts:
            if args.use_her:
                data = rb.sample(args.batch_size, env)
            else:
                data = rb.sample(args.batch_size)
            agent.train_critic(data)
            if step % args.policy_freq == 0:
                agent.train_actor(data, args)
            if step % args.target_update_freq == 0:
                agent.update_targets()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Goal Conditioned SAC')
    parser.add_argument('--env_name', type=str, default='FetchReachDense-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--autotune', default=True)
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--q_lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--target_update_freq', type=int, default=1)
    parser.add_argument('--learning_starts', type=int, default=10_000)
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--use_her', default=False)
    args = parser.parse_args()
    main(args)


#TODO: Add HER