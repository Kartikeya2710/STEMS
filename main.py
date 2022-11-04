import os, sys
import traci
import matplotlib.pyplot as plt
from QueueLength import getAllQueueLengths
from Throughput import get_throughput

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

while step < NUM_STEPS:
	traci.simulationStep() # Executes the next simulation step
	queue_len = getAllQueueLengths() # Sum total of all the queue lengths in our network
	queue_lens.append(queue_len)
	time_step.append(step)	
	total_throughput += get_throughput()
	throughputs.append(total_throughput)
	step += 1

traci.close(False)

fig, ax = plt.subplots(2, 1)

ax[0].plot(time_step, queue_lens, label="TTL")
ax[0].set_title(f"Queue Length v/s {NUM_STEPS} SUMO Simulation Steps")

ax[1].plot(time_step, throughputs, label="TTL")
ax[1].set_title(f"Throughput v/s {NUM_STEPS} SUMO Simulation Steps")

plt.show()

