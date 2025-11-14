# custom environment
from environment import TrainingEnvironment
# custom feature extractor
from feature_extractor import CarlaSequentialFeatureExtractor
# imports
import time
import gymnasium as gym
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
# from stable_baselines3.common.utils import get_schedule_fn
import signal
import sys
# import matplotlib.pyplot as plt # used to visualize first couple images from training
# import torch.nn as nn # used to debug training data shape/format etc.
import numpy as np

# handle exit by ctrl + C
def handle_exit(sig, frame):
    print("\n[INFO] Caught termination signal — cleaning up safely.")
    try:
        if 'env' in globals() and env is not None:
            # due to vectorized, call first one
            env.envs[0].clean_env()
    except Exception as e:
        print(f"[WARN] Cleanup during signal failed: {e}")
    sys.exit(0)

# Register handlers once at startup
signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_exit)  # kill <pid>


# class for info logs
class CarlaMetricsCallback(BaseCallback):
    def __init__(self, verbose=0, print_every=50):
        super().__init__(verbose)
        self.episode_count = 0
        
        # Termination statistics
        self.terminate_counts = {}
        self.truncate_counts = {}
        self.failure_counts = {}

        # Rolling episode stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        self.current_length = 0

        self.episode_lane_violations = 0
        self.episode_conflicting_actions = 0
        self.accident_caused_by_us = 0
        self.traffic_light_violation = 0        
        self.collided_walker = 0
        self.collided_static = 0
        self.collided_other = 0
        
        self.print_every = print_every
        self.recent = []

    def _on_step(self):

        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        # Track per-step reward
        if len(rewards) > 0:
            self.current_rewards.append(float(rewards[0]))
            self.current_length += 1

        # Take info of first environment (single env case)
        info = infos[0] if len(infos) > 0 else {}

        # count and track lane_violation and conflicting action
        for info in infos:
            # increment counters if events occur
            if "lane_violation" in info:
                self.episode_lane_violations += 1
            if "conflicting_action" in info:
                self.episode_conflicting_actions += 1
            if "accident_caused_by_us" in info:
                self.accident_caused_by_us += 1
            if "traffic_light_violation" in info:
                self.traffic_light_violation += 1
            if "collided_walker" in info:
                self.collided_walker += 1
            if "collided_static" in info:
                self.collided_static += 1
            if "collided_other" in info:
                self.collided_other += 1


        # Log step-level signals (if available)
        keys_to_log = [
            "a_progress", "a_speed", "lane_violation", "traffic_light_violation", "traffic_light_warning",
            "collision_impact", "success_route_total_waypoints", "a_overspeed", "collided_type",
            "a_brake", "a_throttle"
        ]
        for k in keys_to_log:
            if k in info:
                self.logger.record(f"carla/{k}", info[k])

        # Episode ended
        if any(dones):
                        
            ep_reward = sum(self.current_rewards)
            ep_length = self.current_length
            
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)

            # track termination/truncation causes
            term = info.get("terminated_reason", None)
            trunc = info.get("truncated_reason", None)
            failure = info.get("failed_reason", None)


            if term:
                self.terminate_counts[term] = self.terminate_counts.get(term, 0) + 1
            if trunc:
                self.truncate_counts[trunc] = self.truncate_counts.get(trunc, 0) + 1
            if failure:
                self.failure_counts [failure] = self.failure_counts.get(failure, 0) + 1


            # Log episode aggregates
            self.logger.record("carla/episode_reward", ep_reward)
            self.logger.record("carla/episode_length", ep_length)

            mean100 = np.mean(self.episode_rewards[-100:])
            # mean of last 100 rewards
            self.logger.record("carla/mean_reward_100", mean100)

            self.logger.record("carla/lane_violation_count", self.episode_lane_violations)
            self.logger.record("carla/conflicting_action_count", self.episode_conflicting_actions)
            self.logger.record("carla/accident_caused_by_us", self.accident_caused_by_us)


            # Log counters for visual debugging in TensorBoard
            for k, v in self.terminate_counts.items():
                self.logger.record(f"terminated/{k}", v)
            for k, v in self.truncate_counts.items():
                self.logger.record(f"truncated/{k}", v)
            for k, v in self.failure_counts.items():
                self.logger.record(f"failed/zero_obs/{k}", v)

            self.recent.append({
                "episode": self.episode_count,
                "terminated": term,
                "truncated": trunc,
                "speed": info.get("speed"),
                "progress": info.get("progress"),
                "impact": info.get("impact_force"),
            })

            # Debug summary every X episodes
            if self.episode_count % self.print_every == 50:
                print(f"\n===== Training Diagnostics (Episode {self.episode_count}) =====")
                print("Termination reasons:", dict(self.terminate_counts))
                print("Truncation reasons:", dict(self.truncate_counts))
                print("Failure reasons:", dict(self.failure_counts))
                print("Last 3 episodes:")
                for r in self.recent[-3:]:
                    print(r)
                print("=============================================================")

            # Reset episode buffers
            self.current_rewards = []
            self.current_length = 0
            self.episode_lane_violations = 0
            self.episode_conflicting_actions = 0
            self.accident_caused_by_us = 0
            self.traffic_light_violation = 0
            self.collided_walker = 0
            self.collided_static = 0
            self.collided_other = 0
        
            self.episode_count += 1

        return True


# Adjust path so we can import local env module
HERE = os.path.dirname(__file__)
sys_path = os.path.abspath(os.path.join(HERE))
if sys_path not in os.sys.path:
    os.sys.path.append(sys_path)

# if paralled training modify rank and pass
# def make_env(rank, seed=0):
def make_env(seed=0):
    def _init():
        env = TrainingEnvironment(show_pygame=False)
        env = Monitor(env)
        # env.seed(seed + rank) # if paralled training modify rank

        return env
    return _init

## for debugging and sanity check of first couple image visualization
# def register_activation_hooks(model):
#     activations = []

#     def hook_fn(module, input, output):
#         activations.append(output.detach().cpu())

#     for layer in model.cnn:
#         if isinstance(layer, nn.Conv2d):
#             layer.register_forward_hook(hook_fn)

#     return activations


if __name__ == "__main__":

    # create models directory
    models_dir = f"models/{int(time.time())}"
    # log_dir = f"logs/{int(time.time())}" # modified using another folder
    os.makedirs(models_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True) # modified using another folder

    model = None
    ####################### if parallel training ######################
    # never tried, might be issues with below
    # Create multiple parallel CARLA envs (optional for faster training)
    # num_envs = 1  # use 1 while debugging; increase to 4–8 later
    # env_fns = [make_env(i, route) for i in range(num_envs)]
    # env = SubprocVecEnv(env_fns)
    # env = VecMonitor(env)
    ####################### if parallel training ######################

    # without vectorization and make_env()
    # env = env = TrainingEnvironment(show_pygame=True)
    # env = Monitor(env)

    # Single env vectorized wrapper
    env = DummyVecEnv([make_env()])
    # vectorized env with VecNormalize (observations and returns)
    # This stabilizes learning when reward scales change
    # and image should not be touched...
    # our reward mechanism is also balanced for PPO between -10 to +10 range, no need to normalize reward either
    # above was true upto some point, later reward shape changed extremely, so normalization depends on the case
    # we already normalized states in observation, no need to normalize here again | depends on the case 
    # modify accordingly
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10., norm_obs_keys=["state"])


    # to sanity check for observation shapes
    # make sure image and states are reaching model as expected
    sample_obs = env.reset()
    print("Image shape before the model training:", sample_obs['image'].shape)
    print("Image before the model training mean/std: ", sample_obs['image'].mean(), sample_obs['image'].std())
    print("State shape before the model training: ", sample_obs['state'].shape)
    assert sample_obs['image'].dtype == np.uint8
    assert sample_obs['image'].max() <= 255 and sample_obs['image'].min() >= 0
    print("State dtype:", sample_obs['state'].dtype)
    # Image --train mean/std: 87.47014361300076 49.318698767582305
    # State --train  shape: (1, 8, 11)

    image_space = env.observation_space["image"]
    print("Is image space -- from train.py:", is_image_space(image_space))
    print("Is image space First -- from train.py:", is_image_space_channels_first(image_space))


    # vec_env = DummyVecEnv([lambda: env])
    env.seed(0)
    """
    If you want to use CnnPolicy or MultiInputPolicy with image-like observation (3D tensor) that are already normalized, 
    you must pass normalize_images=False to the policy (using policy_kwargs parameter, policy_kwargs=dict(normalize_images=False)) 
    and make sure your image is in the channel-first format.
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """
    # Define policy arguments
    policy_kwargs = dict(
    # encouraging it to explore throttle–brake dynamics
    log_std_init = 0.0,
    normalize_images=False,
    features_extractor_class=CarlaSequentialFeatureExtractor,
    features_extractor_kwargs=dict(
        features_dim=512,    # latent dim after combining CNN + state branch
        seq_len=8,           # last 8 states
        state_hidden=64      # hidden dim for MLP/LSTM on state sequence
    ),
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)   

    # Initialize PPO model
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        tensorboard_log="./ppo_carla_final/",
        learning_rate=1e-4, 
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99, 
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.05, 
        vf_coef=0.25, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    try:
        # checkpoint callback
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path="./checkpoints/", name_prefix="ppo_carla")
        metrics_callback = CarlaMetricsCallback(print_every=50)
        print("[DEBUG] Starting training...")
        # Train
        model.learn(total_timesteps=2_000_000, callback=[checkpoint_callback, metrics_callback])
        # started with 2M but crashed around 509K steps due to bug in info logging

        # Save final model
        model.save(os.path.join(models_dir, "ppo_carla_final"))
        env.save(os.path.join(models_dir, "ppo_carla_vecnormalize.pkl"))
        print("[INFO] VecNormalize statistics saved.")

    # try to catch and save the model and vecnormalize stats according to crash type
    except KeyboardInterrupt:
        if model is not None:
            print("Keyboard Interrupted. Saving model...")
            model.save(os.path.join(models_dir, "ppo_carla_keyboard_interrupt"))
            env.save(os.path.join(models_dir, "ppo_carla_vecnormalize_interrupt.pkl"))

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        if model is not None:
            model.save(os.path.join(models_dir, "ppo_carla_crash_backup"))
            env.save(os.path.join(models_dir, "ppo_carla_vecnormalize_backup.pkl"))
            print(f"[INFO] Backup model saved to {models_dir}")

    except RuntimeError as e:
        if model is not None:
            print(f"Runtime Error {e}. Saving model...")
            model.save(os.path.join(models_dir, "ppo_carla_runtime_interrupt"))
            env.save(os.path.join(models_dir, "ppo_carla_vecnormalize_runtime_backup.pkl"))

    finally:
        # clean environment once training finishes
        env.envs[0].clean_env()
        print('Environment Closed after training ended.')

# tensorboard --logdir=C:\Users\Path\to\ppo_carla_final\PPO_20