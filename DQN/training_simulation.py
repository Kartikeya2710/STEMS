import traci
import numpy as np
import timeit

PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

class Simulation:
    MIN_GAP = 1
    def __init__(self, Agent, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Agent = Agent
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._avg_speed_store = []

    def run(self, episode, epsilon):
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        
        traci.start(self._sumo_cmd)
        self.id = traci.trafficlight.getIDList()[0]
        self._incoming_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))
        self._outgoing_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
        self._outgoing_lanes = list(set(self._outgoing_lanes))
        self.lanes_length = {lane: traci.lane.getLength(lane) for lane in self._incoming_lanes + self._outgoing_lanes}

        print("Training...")
        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._old_total_wait = 0
        self._old_queue_length = 0
        self._avg_speed = 0

        old_state = -1
        old_action = -1

        while self._step < self._max_steps:
            
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            current_queue_length = self._get_queue_length()
            reward = self._get_reward()

            if self._step != 0:
                self._Memory.push(old_state, old_action, reward, current_state)

                if len(self._Memory) >= self._Agent.batch_size: 
                    experiences = self._Memory.sample(self._Agent.batch_size)
                    self._Agent.learn(experiences, self._step)

            action = self._choose_action(current_state, epsilon)

            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_state = current_state
            old_action = action
            self._old_queue_length = current_queue_length
            self._old_total_wait = current_total_wait

            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 3))

        traci.close()

        training_time = round(timeit.default_timer() - start_time, 1)

        return training_time


    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_length == waited_seconds
            self._avg_speed += self._get_average_speed()

    def _collect_waiting_times(self):
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)  # get the road id where the car is located
            if lane_id in self._incoming_lanes:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state, epsilon):
        return self._Agent.act(state, epsilon)

    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_per_lane_vehicles(self):
        return [(traci.lane.getLastStepVehicleNumber(lane)) for lane in self._incoming_lanes]

    def _get_normalized_per_lane_vehicles(self):
        return [(traci.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / (self.MIN_GAP + traci.lane.getLastStepLength(lane)))) for lane in self._incoming_lanes]


    def _get_per_lane_waiting_times(self):
        return [(traci.lane.getWaitingTime(lane)) for lane in self._incoming_lanes]

    def _get_normalized_per_lane_waiting_times(self):
        return [(traci.lane.getWaitingTime(lane) / max(1, traci.lane.getLastStepVehicleNumber(lane))) for lane in self._incoming_lanes]

    def _get_queue_length(self):
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self._incoming_lanes)

    def _get_per_lane_queue_lengths(self):
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in self._incoming_lanes]

    def _get_normalized_per_lane_queue_lengths(self):
        return [(traci.lane.getLastStepHaltingNumber(lane) / (self.lanes_length[lane] / (self.MIN_GAP + traci.lane.getLastStepLength(lane)))) for lane in self._incoming_lanes]

    def _get_average_speed_per_lane(self):
        return [(traci.lane.getLastStepMeanSpeed(lane)) for lane in self._incoming_lanes]

    def _get_normalized_average_speed_per_lane(self):
        return [(traci.lane.getLastStepMeanSpeed(lane) / traci.lane.getMaxSpeed(lane)) for lane in self._incoming_lanes]

    def _get_average_speed(self):
        avg_speed_per_lane = self._get_average_speed_per_lane()
        return sum(avg_speed_per_lane) / len(avg_speed_per_lane)

    def _get_pressure(self):
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self._outgoing_lanes) - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self._incoming_lanes)

    def _get_outgoing_lanes_density(self):
        return [traci.lane.getLastStepVehicleNumber(lane) for lane in self._outgoing_lanes]

    def _get_incoming_lanes_density(self):
        return [traci.lane.getLastStepVehicleNumber(lane) for lane in self._incoming_lanes]

    # def _get_per_lane_waiting_times(self):
    #     """
    #     Retrieve the number of cars individually for every lane
    #     """
    #     per_lane_waiting_times = []
    #     for lane in self._incoming_lanes:
    #         max_vehicles = self.lanes_length[lane] / (self.MIN_GAP + traci.lane.getLastStepLength(lane))
    #         per_lane_waiting_times.append(traci.lane.getWaitingTime(lane) / (max_vehicles * 14))
    #     return per_lane_waiting_times

    # def _get_emergency_vehicle_wait_times(self):
    #     total_wait_time = 0
    #     veh_list = traci.vehicle.getIDList()
    #     for veh_id in veh_list:
    #         lane_id = traci.vehicle.getLaneID(veh_id)
    #         if traci.vehicle.getTypeID(veh_id) == "vTypeEmergency" and lane_id in self._incoming_lanes:
    #             total_wait_time += traci.vehicle.getAccumulatedWaitingTime(veh_id)
        
    #     return total_wait_time
    
    # def _get_emergency_vehicle_distances(self):
    #     emergency_vehs = {lane: 0 for lane in self._incoming_lanes}
    #     veh_list = traci.vehicle.getIDList()
    #     for veh_id in veh_list:
    #         lane_id = traci.vehicle.getLaneID(veh_id)
    #         if traci.vehicle.getTypeID(veh_id) == "vTypeEmergency" and lane_id in self._incoming_lanes:
    #             lane_length = traci.lane.getLength(lane_id)
    #             dist_from_intersection = (lane_length - traci.vehicle.getLanePosition(veh_id))/lane_length
    #             emergency_vehs[lane_id] += dist_from_intersection
        
    #     return emergency_vehs

    # def _get_emergency_vehicle_count_per_lane(self):
    #     emergency_vehs = {lane: 0 for lane in self._incoming_lanes}
    #     veh_list = traci.vehicle.getIDList()
    #     for veh_id in veh_list:
    #         lane_id = traci.vehicle.getLaneID(veh_id)
    #         if traci.vehicle.getTypeID(veh_id) == "vTypeEmergency" and lane_id in self._incoming_lanes:
    #             emergency_vehs[lane_id] += 1
        
    #     return emergency_vehs

    # def _emergency_vehicle_priority(self):
    #     emergency_vehicle_count = self._get_emergency_vehicle_count_per_lane()
    #     lane_to_activate = max(emergency_vehicle_count)
    #     if emergency_vehicle_count[lane_to_activate] == 0:
    #         print("no lane")
    #         return
    #     info = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
    #     phases = info.getPhases()
    #     print(phases)
    #     # Loop through the phases to find the one controlling the given lane
    #     for i, phase in enumerate(phases):
    #         if lane_to_activate in phase:
    #             signal_phase_index = i
    #             break
    #     print(signal_phase_index)
    #     # print(lane_to_activate)

    def _get_signal_phase(self, tl_id):
        num_phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].getPhases().__len__()
        current_phase = traci.trafficlight.getPhase(tl_id)
        one_hot_phase = [0] * num_phases
        one_hot_phase[current_phase] = 1
        return np.array(one_hot_phase)

    def _get_state(self):
        state = []
        # per_lane_queue_lengths = [x / 28 for x in self._get_per_lane_queue_lengths()]
        # per_lane_vehicles = [x / 28 for x in self._get_per_lane_vehicles()]
        # per_lane_waiting_times = [x / 28 for x in self._get_per_lane_waiting_times()]
        signal_phase = self._get_signal_phase(self.id)
        per_lane_queue_lengths = self._get_normalized_per_lane_queue_lengths()
        per_lane_vehicles = self._get_normalized_per_lane_vehicles()
        per_lane_waiting_times = self._get_normalized_per_lane_waiting_times()
        # per_lane_avg_speed = self._get_normalized_average_speed_per_lane()

        # signal_phase = [traci.trafficlight.getPhase(self.id)]
        # emergency_vehs = list(self._get_emergency_vehicle_count_per_lane().values())
        
        state.extend(signal_phase)
        state.extend(per_lane_queue_lengths)
        state.extend(per_lane_vehicles)
        state.extend(per_lane_waiting_times)
        # state.extend(per_lane_avg_speed)

        return np.array(state)

    def _get_reward(self):
        current_wait_time = self._collect_waiting_times()
        # current_queue_length = self._get_queue_length()
        return (self._old_total_wait - current_wait_time)
        # return (self._old_queue_length - current_queue_length)

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._avg_speed_store.append(self._avg_speed / self._max_steps)

    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store


    @property
    def avg_speed_store(self):
        return self._avg_speed_store
    