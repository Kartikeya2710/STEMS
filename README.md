# STEMS
STEMS (Smart Traffic Engagement and Management System) is aimed towards optimizing traffic flow at intersections with the help of Reinforcement Learning

### Generating random trips having custom vehicles
```powershell
$ python randomTrips.py -n simple.net.xml -o trips.trips.xml -r routes.rou.xml --additional-file vehicles.add.vtype.xml --fringe-factor 100 --trip-attributes="type=\"typedist1\"" -e 3600 --validate
```

### Traditional Traffic Light (TTL) Performance

**Note:** Please refer to `simple.net.xml` for traffic signal duration and phase related information.

<img src="TTL performance.PNG" alt="TTL performance" width="500" height="350"/>


### Implementation Details

We changed the `TrafficSignal` implementation of `sumo_rl` to include a observation space consisting of the phase information and the pressure of the intersection. 

We did this by redefining the `self.observation_space` and `self.discrete_observation_space` variables

For the queue length metric, we did not know what will be maximum queue length.

1. Pre-define an upper bound
2. Assume the minimum vehicle length to be x meters (`self.MIN_VEH_LEN`) and the minimum gap between two vehicles to be y meters (`self.MIN_GAP`). In this case: `self.max_queue_length = int(max(self.lanes_lenght.values())) // (self.MIN_VEH_LEN + self.MIN_GAP)`

```python
self.max_lane_length = int(max(self.lanes_lenght.values()))
        
self.observation_space = spaces.Box(
    low=np.zeros(self.num_green_phases+len(self.lanes), dtype=np.float32), 
    high=np.ones(self.num_green_phases+len(self.lanes), dtype=np.float32)
)

self.discrete_observation_space = spaces.Tuple((
    spaces.Discrete(self.num_green_phases),
    *(spaces.Discrete(self.max_lane_length) for _ in range(len(self.lanes)))
))
```

We also needed to define a function for calculating the observation at each time step

```python
def _custom_observation_fn(self):
    phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]
    lane_queues = self.get_abs_lane_queue()
    observation = np.array(phase_id + lane_queues, dtype = np.float32)
    return observation
```

And finally included this function in the dictionary `observation_fns`

```python
observation_fns = {
    'default': _observation_fn_default,
    # Included custom observation function
    'custom': _custom_observation_fn
}
```
