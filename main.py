import os, sys
import traci
import matplotlib.pyplot as plt
from QueueLength import getAllQueueLengths
from Throughput import get_throughput
from CumulativeDelay import cummulative_delay_per_lane

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.exit("please declare environment variable 'SUMO_HOME'")

# Hyperparameters
sumoBinary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe"
sumoCmd = [sumoBinary, "-c", "simple.sumocfg"]
NUM_STEPS = 1000

traci.start(sumoCmd)
step = 0
queue_lens = []
total_throughput = 0
throughputs = []
time_step = []
avg_delays = []

while step < NUM_STEPS:
	traci.simulationStep() # Executes the next simulation step

	queue_len = getAllQueueLengths() # Sum total of all the queue lengths in our network
	queue_lens.append(queue_len)

	total_throughput += get_throughput()
	throughputs.append(total_throughput)

	avg_delay = cummulative_delay_per_lane()
	avg_delays.append(avg_delay)

	time_step.append(step)

	step += 1

traci.close(False)


# Plotting the metrics
fig, ax = plt.subplots(3, 1)

ax[0].plot(time_step, queue_lens, label="TTL")
ax[0].set_ylabel("Queue Length (No.)")

ax[1].plot(time_step, throughputs, label="TTL")
ax[1].set_ylabel("Throughtput (No.)")

ax[2].plot(time_step, avg_delays, label="TTL")
ax[2].set_ylabel("Lane Delay (sec)")

plt.xlabel("Steps")

plt.show()

