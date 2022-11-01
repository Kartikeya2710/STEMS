import traci

def getQueueLength(lane, length_threshold=200):
	# num = 0
	# lane_length = traci.lane.getLength(lane)

	# for k in traci.lane.getLastStepVehicleIDs(lane):
	# 	if traci.vehicle.getLanePosition(k) > lane_length - length_threshold:
	# 		num += 1

	# return num
    return traci.lane.getLastStepHaltingNumber(lane)


def getAllQueueLengths(length_threshold=200):
    lanes = traci.lane.getIDList() # Returns a list of all lane IDs in the network
    total_queue_length = 0

    for lane in lanes:
        queue_length = getQueueLength(lane)
        total_queue_length += queue_length

    return total_queue_length

