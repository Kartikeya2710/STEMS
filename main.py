import os
import sys
import argparse
import traci
from metricsCalculation import get_metrics

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

parser = argparse.ArgumentParser()
parser.add_argument("-gm", "--gui-mode", help="GUI Mode = ['enabled', 'disabled']", default='enabled')
parser.add_argument("--steps", type=int, help="Number of simulation steps", default=1000)

args = parser.parse_args()

if args.gui_mode is None or args.gui_mode == "enabled":
    sumoBinary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe"
elif args.gui_mode == "disabled":
    sumoBinary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo.exe"

sumoCmd = [sumoBinary, "-c", "simple.sumocfg", "--quit-on-end"]

traci.start(sumoCmd)

NUM_STEPS = args.steps
traffic_light_ids = traci.trafficlight.getIDList()
lanes = []
for tl_id in traffic_light_ids:
    lanes += ( list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id))) )


for step in range(NUM_STEPS):
    traci.simulationStep()  # Executes the next simulation step

traci.close()

get_metrics(num_lanes=len(lanes), time_steps=NUM_STEPS)


