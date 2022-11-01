import os, sys
import traci
import matplotlib.pyplot as plt
from QueueLength import getAllQueueLengths

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
time_step = []

while step < NUM_STEPS:
	traci.simulationStep() # Executes the next simulation step
	queue_len = getAllQueueLengths() # Sum total of all the queue lengths in our network
	queue_lens.append(queue_len)
	time_step.append(step)	
	step += 1

plt.plot(time_step, queue_lens, label="TTL")
plt.title(f"Queue Length v/s {NUM_STEPS} SUMO Simulation Steps")
plt.xlabel("Step")
plt.ylabel("Queue Length")
plt.legend(loc = "upper right")
plt.show()

traci.close(False)