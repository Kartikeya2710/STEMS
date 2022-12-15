import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def get_metrics(num_lanes, time_steps):
    tree = ET.parse("e2output.xml")
    root = tree.getroot()

    freq = int(float(root[0].get('end')))
    throughput = 0
    throughputs = []
    queue_lengths = []
    cumulative_delay = []
    time = []
    
    for step in range(time_steps//freq):
        time.append(step*freq)
        
        queue_length = 0
        mean_delay = 0
        for lane in range(num_lanes):
            index = step*num_lanes + lane
            throughput += int(root[index].get('nVehLeft'))
            queue_length += int(root[index].get('jamLengthInVehiclesSum'))
            mean_delay += float(root[index].get('meanHaltingDuration'))

        throughputs.append(throughput)
        queue_lengths.append(queue_length)
        cumulative_delay.append(mean_delay)

    fig, ax = plt.subplots(3, 1)

    ax[0].plot(time, queue_lengths, label="TTL")
    ax[0].set_ylabel("Queue Length (No.)")

    ax[1].plot(time, throughputs, label="TTL")
    ax[1].set_ylabel("Throughtput (No.)")

    ax[2].plot(time, cumulative_delay, label="TTL")
    ax[2].set_ylabel("Lane Delay (sec)")

    plt.xlabel("Steps")

    plt.show()