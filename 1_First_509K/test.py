from stable_baselines3 import PPO
from environment import TrainingEnvironment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import signal
import sys

# handle exit by ctrl + C
def handle_exit(sig, frame):
    print("\n[INFO] Caught termination signal — cleaning up safely.")
    try:
        if 'env' in globals() and env is not None:
            env.envs[0].clean_env()
    except Exception as e:
        print(f"[WARN] Cleanup during signal failed: {e}")
    sys.exit(0)

# Register handlers once at startup
signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_exit)  # kill <pid>

# def make_env(rank, seed=0): # if paralled training modify rank and pass
def make_env(seed=0):
    def _init():
        env = TrainingEnvironment(show_pygame=True)
        env = Monitor(env)
        # env.seed(seed + rank) # if paralled training modify rank
        return env
    return _init

# Single env vectorized wrapper
env = DummyVecEnv([make_env()])
env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10.)
env.training = False
env.norm_reward = False
env.seed(0)

# model_path = "/home/ubuntu-user/carlaProject/SelfDrive/Thesis/models/1761648520/ppo_carla_final.zip"
model_path = "/home/ubuntu-user/carlaProject/SelfDrive/Thesis/checkpoints/ppo_carla_1000000_steps.zip"
model = PPO.load(model_path, env=env)

episodes = 3
for episode in range(episodes):
    print(f'\n[---Testing Agent---] Episode: {episode+1}\n')
    obs = env.reset()

    try:
        
        while True:
            # use model to predict the action
            action, _ = model.predict(obs)
            #obs, reward, terminated, truncated, info = env.step(action)
            obs, reward, done, info = env.step(action)

            if done[0]:
                print('*************** Done! ***************')
                break

            #done = terminated or truncated  # SB3 new API
    except Exception as e:
        print(f'Error while testing: {e}')
    
    finally:
        try:
            env.envs[0].clean_env()
        except Exception as e:
            print(f'Error while cleaning env as finally: {e}')
