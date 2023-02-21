import subprocess
import os
import sys

class TrafficGenerator:
    def __init__(self, model_config, dir_config):
        self._model_config = model_config
        self._dir_config = dir_config
        self._net_file = os.path.join(dir_config['intersection_dir'], "environment.net.xml")
        self._route_file = os.path.join(dir_config['intersection_dir'], "episode_routes.rou.xml")
        self._additional_file = os.path.join(dir_config['intersection_dir'], dir_config['additional_file'])
        self._trips_file = os.path.join(dir_config['intersection_dir'], "trips.trips.xml")
        
    def generate_routefile(self, seed):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        random_trips = os.path.join(tools, "randomTrips.py")

        if self._dir_config.get("additional_file"):
            subprocess.call(["python", random_trips, "-n", self._net_file, "-r", self._route_file, "-o", self._trips_file, "--additional-file", self._additional_file, "--trip-attributes", f"type=\"{self._dir_config['trip_attributes']}\"", "-e", str(self._model_config['max_steps']), "--validate", "--seed", str(seed), "--random-depart"])

        else:
            subprocess.call(["python", random_trips, "-n", self._net_file, "-r", self._route_file, "-o", self._trips_file, "-e", str(self._model_config['max_steps']), "--validate", "--seed", str(seed), "--random-depart"])


        # python randomTrips.py -n environment.net.xml -r episode_routes.rou.xml --additional-file vehicles.add.vtype.xml --trip-attributes="type=\"typedist1\"" -e 5400 --validate --random-depart