import traci

def getAllQueueLengths():
    lanes = traci.lane.getIDList() # Returns a list of all lane IDs in the network
    total_queue_length = 0

    for lane in lanes:
        queue_length = traci.lane.getLastStepHaltingNumber(lane) # Gives the number of vehicles travelling at speeds < 0.1 m/s in the lane
        total_queue_length += queue_length

    return total_queue_length

