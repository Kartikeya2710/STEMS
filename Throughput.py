import traci

def get_throughput():
	detectors = traci.multientryexit.getIDList()
	throughput = 0

	for detector in detectors:
		throughput += traci.multientryexit.getLastStepVehicleNumber(detector)
	
	return throughput
