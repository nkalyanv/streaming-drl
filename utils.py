import numpy as np

#The arrays and lists are done to make sure that the data is stored in the right format for SB3. (Akin to having used a dummy vec env)
def store_transition(episode, obs, action, reward, next_obs, done, info):
    assert isinstance(obs, dict)
    assert isinstance(next_obs, dict)
        # Ensure each observation key is converted to a numpy array
    obs_arrays = {k: np.array(v) for k, v in obs.items()}
    next_obs_arrays = {k: np.array(v) for k, v in next_obs.items()}
    
    episode["observations"].append(obs_arrays)
    episode["actions"].append(np.array(action))
    episode["rewards"].append(np.array(reward))
    episode["next_observations"].append(next_obs_arrays)
    episode["dones"].append(np.array([done]))
    episode["infos"].append([info])
    return episode

def sparse_reward(achieved_goal, desired_goal, info, goal_tol=0.01):
    return np.array(float(np.linalg.norm(achieved_goal - desired_goal) < goal_tol))

def dense_reward(achieved_goal, desired_goal, info, goal_tol=None):
    return np.array(-np.linalg.norm(achieved_goal - desired_goal))