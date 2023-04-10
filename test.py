from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
from DQN.testing_simulation import Simulation
from helper.generator import TrafficGenerator
from DQN.Agent import DuelingDoubleDQNAgent, DoubleDQNAgent
from helper.visualization import Visualization
from helper.utils import *


if __name__ == "__main__":
    dir_config = load_config_file('dir_settings.yaml')
    test_settings_path = os.path.join(dir_config['agent_dir'], 'test_settings.yaml')

    test_config = load_config_file(test_settings_path)
    sumo_cmd = set_sumo(test_config['gui'], dir_config['intersection_dir'], dir_config['sumocfg_file_name'], test_config['max_steps'])
    model_path, plot_path = set_test_path(dir_config['agent_dir'], dir_config['models_path_name'], dir_config['model_to_test'])
    checkpoint_number = dir_config['checkpoint_number']

    # Load the pretrained model

    Model = DoubleDQNAgent(
        test_config['num_states'], 
        test_config['num_actions'],
        test_config['fc_dims']
    )
    
    Model = Model.load_model(model_path, checkpoint_number, Model)
        

    TrafficGen = TrafficGenerator(
        test_config,
        dir_config
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        test_config['max_steps'],
        test_config['green_duration'],
        test_config['yellow_duration'],
        test_config['num_states'],
        test_config['num_actions']
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(test_config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src=test_settings_path, dst=os.path.join(plot_path, 'test_settings.yaml'))

    Simulation.test_ttl()

    Visualization.save_data_and_plot(data=Simulation.wait_time_store, ttl_data=Simulation.ttl_wait_times,filename='wait times', xlabel='Action step', ylabel='Wait Time')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, ttl_data=Simulation.ttl_queue_lengths, filename='queue', xlabel='Step', ylabel='Queue Length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.avg_speed_store, ttl_data=Simulation.ttl_avg_speeds, filename='avg_speed', xlabel='Step', ylabel='Average Speed (m/s)')