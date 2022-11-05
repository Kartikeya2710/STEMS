import traci

def avgdelay():
    #retriving the ID list 
    lanes = traci.lane.getIDList()

    time=0.0

    #running through the loop of the lane ID
    for lane in lanes:
        #retriving the waiting time for each lane 
        t = traci.lane.getWaitingTime(lane)
        #appending the waiting time for each lane
        time=time+t;

    leng=len(lanes)


    return (time/leng)
