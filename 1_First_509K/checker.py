
from stable_baselines3.common import env_checker
from feature_extractor import CarlaSequentialFeatureExtractor
from environment import TrainingEnvironment
env = TrainingEnvironment()
obs = env.reset()
images = obs[0]['image']      # shape: (num_envs, H, W, C)
#print('images: ', images)
print("Image batch shape:", images.shape, images.dtype)

states = obs[0]["state"]      # shape: (num_envs, state_dim)
#print('states: ', states)
print("State batch shape:", states.shape, states.dtype)

# # For single environment
# print("Single image shape:", images[0].shape)
# print("Single state shape:", states[0].shape)
env_checker.check_env(env, warn=True, skip_render_check=True)


