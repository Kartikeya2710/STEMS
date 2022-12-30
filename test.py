import os
import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import DQN
from sumo_rl import TrafficSignal
from stable_baselines3.common.callbacks import ProgressBarCallback

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment

def my_reward_fn(traffic_signal):
    return traffic_signal._queue_reward()

if __name__ == '__main__':

    SEED = 42
    NUM_SECONDS = 1_00_000
    
    # checkpoint_callback = CheckpointCallback(
    #     save_freq = 1000,
    #     save_path = "./saved_model",
    #     name_prefix="PPO",
    # )

    env = SumoEnvironment(
        net_file='simple.net.xml',
        route_file='routes.rou.xml',
        single_agent=True,
        use_gui=True,
        num_seconds=NUM_SECONDS,
        # additional_sumo_cmd="-a detectors.add.xml",
        reward_fn=my_reward_fn,
        observation_fn='custom',
        sumo_seed=SEED,
        yellow_time=3,
        min_green=5,
        max_green=60
    )

    
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
        seed=SEED
    )
    
    model.learn(
        total_timesteps=NUM_SECONDS,
        progress_bar=True,
        log_interval=1
    )
