from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from DQN.training_simulation import Simulation
from helper.generator import TrafficGenerator
from DQN.memory import ReplayMemory
from DQN.Agent import DuelingDoubleDQNAgent, DoubleDQNAgent
from helper.visualization import Visualization
from helper.utils import *


if __name__ == "__main__":
    dir_config = load_config_file('dir_settings.yaml')
    train_settings_path = os.path.join(dir_config['agent_dir'], 'train_settings.yaml')

    train_config = load_config_file(train_settings_path)

    sumo_cmd = set_sumo(train_config['gui'], dir_config['intersection_dir'], dir_config['sumocfg_file_name'], train_config['max_steps'])
    path = set_train_path(dir_config['agent_dir'], dir_config['models_path_name'])

    Model = DoubleDQNAgent(
        train_config['num_states'], 
        train_config['num_actions'],
        train_config['fc_dims'], 
        train_config['gamma'],
        train_config['learning_rate'],
        train_config['batch_size'], 
        train_config['update_every'], 
    )

    Memory = ReplayMemory(
        train_config['capacity']
    )

    TrafficGen = TrafficGenerator(
        train_config,
        dir_config
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        train_config['gamma'],
        train_config['max_steps'],
        train_config['green_duration'],
        train_config['yellow_duration'],
        train_config['num_states'],
        train_config['num_actions'],
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < train_config['total_episodes']:
        print('\n----- Episode', str(episode), 'of', str(train_config['total_episodes']))
        epsilon = 1.0 - (episode / train_config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        training_time = Simulation.run(episode, epsilon)  # run the simulation
        if episode % train_config['save_every'] == 0:
            Model.save_model(path, episode)
        print('Training time:', training_time, round(training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    copyfile(src=train_settings_path, dst=os.path.join(path, 'training_settings.yaml'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.avg_speed_store, filename='avg_speed', xlabel='Episode', ylabel='Average speed (m/s)')
