import traci

def cummulative_delay_per_lane():
    #retriving all the lanes by their IDs
    lanes = traci.lane.getIDList()
    cumulative_wait_time = 0

    #running through the loop of the lane ID
    for lane in lanes:
        #retriving the waiting time for each lane 
        wait_time = traci.lane.getWaitingTime(lane)
        #appending the waiting time for each lane
        cumulative_wait_time += wait_time

    num_lanes = len(lanes)


    return (cumulative_wait_time/num_lanes)
