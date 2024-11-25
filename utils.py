def store_transition(episode, obs, action, reward, next_obs, done, info):
    episode["observations"].append(obs)
    episode["actions"].append(action)
    episode["rewards"].append(reward)
    episode["next_observations"].append(next_obs)
    episode["dones"].append(done)
    episode["infos"].append(info)