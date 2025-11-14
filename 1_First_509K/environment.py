# imports
import carla
import gymnasium as gym
import numpy as np
import threading
import pygame
from gymnasium import spaces
import os
import sys
import cv2
import random
import time
from collections import deque
import math
# from gymnasium.utils import seeding

# this was necessary once we worked other than PythonAPI/examples folder, PythonAPI/carla/agents/navigation folder needed
# Add PythonAPI path so we can import navigation modules (adjust if your path differs)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "PythonAPI"))
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import LocalPlanner

# custom training environment used by gymnasium during the training
class TrainingEnvironment(gym.Env):
    SAFE_DISTANCE = 20.0 # started with 40, reduced to 30 after training reduced to 20

    # default parameters
    def __init__(self,
                 host='localhost', port=2000, 
                 sync_mode=False, 
                 fixed_delta_seconds=0.05, 
                 show_pygame=False
                 ):
        
        super().__init__()
        # pygame needed during the testing, can be used also in training if set True in train.py
        self.SHOW_PYGAME = show_pygame
        # CARLA client/world,host,port etc.
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # set synchronous mode if you control the server and want determinism.
        # for quick testing, running in asynchronous mode is often safer (less strict) -- sensor frame id not matching
        # sync is problematic causes issues
        # a lot of try/exceptions implemented due to many crashes
        try:
            self.settings = self.world.get_settings()
            # toggle sync via constructor argument; leaving async by default to avoid issues
            if sync_mode:
                self.settings.synchronous_mode = True
                self.settings.fixed_delta_seconds = fixed_delta_seconds
            else:
                self.settings.synchronous_mode = False
                self.settings.fixed_delta_seconds = None

            # no_rendering_mode may break spectator/visuals
            # heavy task is going on, set it off to save GPU
            self.settings.no_rendering_mode = True
            self.world.apply_settings(self.settings)
        except Exception as e:
            print(f"[WARN] Could not apply world settings: {e}")

        # blueprints and spawn points
        self.blueprint_library = self.world.get_blueprint_library()
        # we picked just a vehicle from available ones
        # before experimented with Charger, camera locations etc needs to be adjusted accordingly
        vehicle_list = self.blueprint_library.filter('vehicle.*mercedes*')
        self.vehicle_bp = vehicle_list[0] if vehicle_list else self.blueprint_library.filter('vehicle.*')[0]
        self.spawn_points = self.map.get_spawn_points()

        # store the route info
        self.route = []
        # local planner
        self.lp = None

        # actor lists (ensure exist even before spawn)
        self.actor_vehicles = []
        self.walkers = []
        self.walker_controllers = []
        self.sensor_actors = []
        
        # for image processing
        self.latest_image = None
        # Thread-safe image sharing
        self.image_lock = threading.Lock()

        # other telemetry
        # imu used in data debugging via autopilot but not used in training
        # self.latest_imu_data = ""
        # collision tracker
        self.collision_happened = False
        # for debugging
        # self.collision_event = None
        # Actor against whom the parent collided.
        self.collision_other_actor = None
        # Normal impulse result of the collision, to use the defining guilty party from accident
        self.collision_normal_impulse = None
        # [Debugging]
        # self.collision_info = {}
        # to use the defining guilty party from accident
        self.collision_impact = None

        # Obstacle: Actor detected as an obstacle.
        self.detected_actor = None
        # max distance of the obstacle detector sensor - limit
        self.detector_distance = self.SAFE_DISTANCE
        # detected object distance, by default is max of the detector
        self.front_distance = self.SAFE_DISTANCE
        # collision sensor: if we are the reason of accident
        self.accident_caused_by_us = False

        # track lane data - solid lane
        self.latest_lane_data = None
        # self.latest_detector_data = None *****
        # to control display time
        # self.collision_timer = 0 *****
        # to control display time
        # self.lane_invasion_timer = 0 *****
        # to control display time
        # self.detection_timer = 0 *****

        # to calculate remaining distance to goal
        self.prev_distance = None
        # to track the progress of route
        # self.progress_buffer = [] *****

        # to track traffic lights in respect to ego vehicle
        self.last_traffic_light = None
        # to track the light state
        self.last_traffic_light_state = "None"
        # traffic light distance
        self.tl_distance = None
        # to normalize the distance
        self.initial_tl_distance = None
        # self.current_throttle = 0.0 *****
        # to store steering value that comes from local planner, used to detect sharp turns
        self.current_steer = 0.0
        # to calculate ego vehicle speed
        self.speed = 0.0
        # reward mechanism, compare situation of unnecessary breaking
        self.last_action = (0.0, 0.0)  # (throttle, brake)
        # pygame display — create only once when needed
        # init pygame while testing
        self.display = None
        # track pygame event to close pygame, render, used while testing
        self.close_pygame = False
     
        # length of sequential data from state vector
        self.state_buffer_len = 8
        # store sequential data
        self.state_buffer = deque(maxlen=self.state_buffer_len)
        # length of state vector components; to supply to the agent, to use during training
        self.current_state_dim = 11  

        # Fill initial state buffer with zeros
        for _ in range(self.state_buffer_len):
            self.state_buffer.append(np.zeros(self.current_state_dim, dtype=np.float32))

        # [Debuging] 
        # frame numbers to compare observations from sensors
        # it seems they do not match
        # self.image_frame = None
        # self.lane_frame = None
        # self.obstacle_frame = None
        # self.collision_frame = None
        # self.current_frame = None

        # add difficulty after some steps for curriculum teaching
        self.total_steps = 0 
        # to track moved steps and avoid stuck
        self.step_moved = 0
        # set limit for max step in each episode
        self.max_steps = 5000
        # Action space: throttle [-1,1], brake [0,1]
        # limit throttle to [0,1] due to model expects all possible actions checker.py raised error
        # we do not need reverse, so limit throttle also with [0,1] and brake already has this boundaries
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # sequential observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8),
            "state": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.state_buffer_len, self.current_state_dim), 
                dtype=np.float32)
        })

    # RESET function, set all variables to default after each episode; when terminate or truncate
    def reset(self, seed=None, options=None, spawn_vehicles=20, spawn_walkers=10):
        # Initialize random seed (important for reproducibility)
        super().reset(seed=seed)

        # check any remaining actor from previous episode
        self.debug_list_actors('before')
        
        # Clean up old actors/sensors if any
        self.clean_env()
        # to allow carla do the necessary cleanings
        time.sleep(0.5)

        # reset to default values as initialized
        self.last_traffic_light_state = "None"
        self.tl_distance = None
        self.speed = 0.0
        self.collision_happened = False
        self.collision_normal_impulse = None
        self.accident_caused_by_us = False
        self.detected_actor = None
        self.front_distance = self.SAFE_DISTANCE
        self.latest_lane_data = None
        self.close_pygame = False
        self.step_moved = 0
        # self.progress_buffer = [] *****
        # empty the spawned actors during reset TODO uncomment after test ********************
        self.sensor_actors = []                
        self.actor_vehicles = []
        self.walkers = []
        self.walker_controllers = []
        self.vehicle = None
        self.lp = None
        self.route = []
        # self.collision_info = {}
        self.collision_impact = None
        self.current_steer = 0.0                    
        # store initial distance to goal to be able to compare
        self.initial_distance = self.calculate_remaining_route_distance()
        # to normalize the distance
        self.initial_tl_distance = None
        self.prev_distance = self.initial_distance
        # clear stuck detection
        self.no_progress_steps = 0

        # curriculum difficulty
        # spawn vehicle and pedestrian after some time
        if self.total_steps < 500_001:
            spawn_vehicles = 0
            spawn_walkers = 0

        elif self.total_steps < 1_000_001:
            spawn_vehicles = 7
            spawn_walkers = 5

        elif self.total_steps < 1_200_001:
            spawn_vehicles = 14
            spawn_walkers = 8
        else:
            spawn_vehicles = 20
            spawn_walkers = 10

        # used while testing
        if self.SHOW_PYGAME:
            # Ensure previous display is closed
            if pygame.get_init():
                pygame.quit()

            # init pygame display
            if self.display is None:
                pygame.init()
                # use HWSURFACE | DOUBLEBUF for faster flipping
                self.display = pygame.display.set_mode((400, 300), pygame.HWSURFACE | pygame.DOUBLEBUF)
                pygame.display.set_caption("CARLA Semantic Camera RL Agent Test View")
        
        # if fails return zero observations        
        try:
            # Spawn ego vehicle, to avoid crash try_spawn_actor
            spawn_point = random.choice(self.spawn_points) if self.spawn_points else carla.Transform()
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn_point)
            # if fails to spawn ego vehicle, give warning
            if self.vehicle is None:                
                print("[WARN] Ego vehicle failed to spawn. Returning fallback observation.")
            # brake for Carla to do spawning
            time.sleep(0.5)
            
            # Spawn the traffic / designed by curriculum / total time steps
            try:
                self.spawn_traffic(num_vehicles=spawn_vehicles, num_walkers=spawn_walkers)

            # if fails return zero observations
            except Exception as e:
                print(f"[WARN] spawn_traffic had issues: {e} \n _safe_zero_obs returned")
                obs = self._safe_zero_obs()
                info = {"failed_reason": str(e)}
                return obs, info
            # brake for Carla to do spawning
            time.sleep(2)

            # # Spectator above vehicle, incase used
            # try:
            #     self.spectator = self.world.get_spectator()
            #     self.spectator.set_transform(
            #         carla.Transform(self.vehicle.get_transform().location + carla.Location(z=30),
            #                         carla.Rotation(pitch=-90))
            #     )
            # except Exception as e:
            #     print(f"[WARN] spectator set failed: {e}")

            # random route planning
            self.start_location = self.vehicle.get_location()
            self.goal_location = random.choice(self.spawn_points).location if self.spawn_points else self.start_location
            self.sampling_resolution = 2.0

            try:
                self.grp = GlobalRoutePlanner(self.map, self.sampling_resolution)
                self.route = self.grp.trace_route(self.start_location, self.goal_location)
                print(f"Route has {len(self.route)} waypoints from start to goal")

            # if fails return zero observations
            except Exception as e:
                print(f"[WARN] GlobalRoutePlanner failed: {e} \n_safe_zero_obs returned")
                self.route = []
                obs = self._safe_zero_obs()
                info = {"failed_reason": str(e)}
                return obs, info

            # Local planner
            # even though speed set as parameter, has no effect, used via autopilot while developing sensor data debuging
            self.lp = LocalPlanner(self.vehicle, opt_dict={"target_speed": 30})

            # set the route to local planner
            if self.route:
                self.lp.set_global_plan(self.route)
            
            # give planner a moment
            time.sleep(1)  

            # camera transfor varies from our packed version and released packaged version
            # earlier version of package -- ours            
            # camera_transform = carla.Transform(carla.Location(x=0.6, z=1.6))
            # released version
            camera_transform = carla.Transform(carla.Location(x=1.2, z=1.85))

            # use semantic segmentation camera
            try:                
                self.camera = self.attach_sensor('sensor.camera.semantic_segmentation', camera_transform, {
                    'image_size_x': '400',
                    'image_size_y': '300',
                    'fov': '90',
                })
                # before training RGB camera used for sensor data debugging
                """
                self.camera = self.attach_sensor('sensor.camera.rgb', camera_transform, {
                    'image_size_x': '400',
                    'image_size_y': '300',
                    'fov': '90',
                    'exposure_mode': 'manual',
                    'exposure_compensation': '2.0',     # Brightness control (-5 to 5)
                    'iso': '200',                       # ISO sensitivity (default is 100)
                    'shutter_speed': '100'             # In microseconds; try 100–200
                })
                """
            # if fails return zero observations
            except Exception as e:
                print(f"[ERROR] Camera spawn failed: {e} \n_safe_zero_obs returned")
                obs = self._safe_zero_obs()
                info = {"failed_reason": str(e)}
                return obs, info


            # IMU - not used in trainig
            # imu_transform = carla.Transform(carla.Location(z=2.0))
            # try:
            #     self.imu = self.attach_sensor('sensor.other.imu', imu_transform)
            # except Exception as e:
            #     print(f"[WARN] IMU spawn failed: {e}")
            #     self.imu = None

            # use collision sensor
            col_transform = carla.Transform(carla.Location(z=2.0))
            try:
                self.col = self.attach_sensor('sensor.other.collision', col_transform)
            except Exception as e:
                print(f"[WARN] collision sensor spawn failed: {e}")
                self.col = None

            # use lane invasion sensor / only solid line detection implemented
            lane_transform = carla.Transform(carla.Location(z=0.30))
            try:
                self.lane = self.attach_sensor('sensor.other.lane_invasion', lane_transform)
            except Exception as e:
                print(f"[WARN] lane sensor spawn failed: {e}")
                self.lane = None
 

            # use obstacle detector - transform location important, if close to road, detects road as obstacle
            obj_transform = carla.Transform(carla.Location(z=1.5))
            try:
                self.obstacle_detector = self.attach_sensor('sensor.other.obstacle', obj_transform, {
                    'distance': self.detector_distance,
                    'hit_radius': 0.3, # later changed
                    'only_dynamics': False,
                    'debug_linetrace': False,
                })
            except Exception as e:
                print(f"[WARN] obstacle detector spawn failed: {e}")
                self.obstacle_detector = None

            # register listeners: callbacks
            # The camera callback will convert and store a *numpy* copy of image bytes.
            self.camera.listen(self.camera_callback)
            if self.imu:
                self.imu.listen(self.imu_callback)
            if self.col:
                self.col.listen(self.collision_handler)
            if self.lane:
                self.lane.listen(self.lane_handler)
            if self.obstacle_detector:
                self.obstacle_detector.listen(self.obstacle_detector_handler)

            # wait a few ticks to stabilize sensors
            for _ in range(5):
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                time.sleep(0.1)
        
            obs = self.get_observation()
            #print(f'Obs: {obs}')

            # include route info
            # info = {f"Route has {len(self.route)} waypoints from start to goal"}  
            info = {"route_info": f"Route has {len(self.route)} waypoints from start to goal"}

            # return obs and info from reset function, expected so...
            return obs, info
        
        # if any failure of spawning ego vehicle return zero observations
        except Exception as e:
                print(f"[ERROR] Reset failed: {e} \n_safe_zero_obs returned")
                obs = self._safe_zero_obs()
                info = {"failed_reason": str(e)}
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                return obs, info

    # STEP function; each step in simulation decided by agent's action, 
    # collects and returns [observations, rewards, terminate, truncate, info]
    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        try:
            # --- APPLY ACTION from the Agent ---
            # agent has right only on throttle and brake, steering data comes from planner
            throttle, brake = action
            # for security limit to boundaries
            throttle = float(np.clip(throttle, 0.0, 1.0))
            brake    = float(np.clip(brake, 0.0, 1.0))
            # store action for comparison in reward mechanism
            self.last_action = (throttle, brake)
            # self.current_throttle = throttle *****

            # check vehicle again, if no vehicle terminate and return zeros
            if self.vehicle is None or not self.vehicle.is_alive:
                print("[WARN] Vehicle invalid, skipping planner step. _safe_zero_obs returned")
                info = {"failed_reason": "Vehicle is None"}
                terminated = True
                obs = self._safe_zero_obs()
                return obs, 0.0, True, False, {"terminated_reason": "vehicle_invalid"}
            
            # Local planner handles steering
            control = self.lp.run_step(debug=False)
            # set agent's throttle value to control object
            control.throttle = float(throttle)
            # set agent's brake value to control object
            control.brake = float(brake)
            # store steering to detect sharp turns
            self.current_steer = control.steer
            # and keep in boundaries
            self.current_steer    = float(np.clip(self.current_steer, -1.0, 1.0))
            # print('control.steering', self.current_steer)
            # control the vehicle with control object
            self.vehicle.apply_control(control)
            # increment step movement in episode
            self.step_moved += 1
            # increment total steps in whole training
            self.total_steps += 1

            # # [Debug] how step, throttle, brake and speed is going along with training ####################### [Debug]
            # if self.step_moved % 10 == 0:
            #     print(f"[DEBUG] step {self.step_moved}: steer={self.current_steer} throttle={throttle:.2f}, brake={brake:.2f}, speed={self.speed:.2f}")
            # ####################### [Debug]

            # simulation tick according to sync or async
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()

            # wait camera to start
            wait_start = time.time()
            while True:
                with self.image_lock:
                    if self.latest_image is not None:
                        break
                if time.time() - wait_start > 1.0:  # 1-second timeout
                    print("[WARN] No camera image this tick.")
                    break            

            # --- GET OBSERVATIONS [IMAGE, STATE_VECTOR] ---
            obs = self.get_observation()

            # --- CALCULATE THE REWARDS-TERMINATE-TRUNCATE-INFO ---
            reward, terminated, truncated, info = self.compute_reward_done_info()

            # [DEBUG] follow training
            if terminated or truncated:
                print(f"[EP END] step_moved={self.step_moved}, terminated={terminated}, truncated={truncated}, info={info}")
            
            # [DEBUG] follow training
            # if self.step_moved != 0 and self.step_moved % 100 == 0:
            ## Obs has also image, array no need to print
            #     print(f'obs: {obs}, \nreward: {reward}, \nterminated: {terminated}, \ntruncated: {truncated}, \ninfo: {info}')

            # return collected data to the agent
            return obs, reward, terminated, truncated, info
        
        # if training cancelled, give warning and return zeros
        except KeyboardInterrupt:
            terminated = True
            print("KeyboardInterrupt caught, terminated... _safe_zero_obs returned")
            obs = self._safe_zero_obs()
            return obs, 0.0, True, False, {"terminated_reason": "manual_interrupt"}

        finally:
            pass
            # ensure cleanup when leaving step - cleanup done by gymnasium
            #self.clean_env()
    
    # CLEAN environment once episode ends - managed by gymnasium -
    def clean_env(self):

        # stop listeners and destroy actors safely
        print("[INFO] Cleaning environment...")
        for sensor in list(getattr(self, "sensor_actors", [])):
            if not sensor:
                continue
            try:
                # Stop any listeners safely
                try:
                    if hasattr(sensor, "is_listening") and sensor.is_listening:
                        sensor.stop()
                    elif hasattr(sensor, "stop"):
                        sensor.stop()
                    else:
                        # fallback for sensors without stop()
                        sensor.listen(None)
                except Exception as e:
                    print(f"[WARN] Could not stop sensor {sensor.type_id}: {e}")

                # wait for server process stop
                time.sleep(0.1)

                # Destroy sensor
                if sensor.is_alive:
                    sensor.destroy()
                else:
                    print(f"[INFO] Sensor {sensor.id} already dead")

            except Exception as e:
                print(f"[WARN] Exception while destroying {sensor.type_id}: {e}")
        
        # empty the sensors list
        self.sensor_actors = []

        # destroy ego vehicle
        try:
            if hasattr(self, "vehicle") and self.vehicle is not None:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
                    print("[INFO] Ego vehicle destroyed.")
        except Exception as e:
            print(f"[WARN] Ego vehicle cleanup error: {e}")
        self.vehicle = None

        # tick to ensure cleanup
        try:
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception:
            pass

        # destroy NPC vehicles 
        if hasattr(self, "actor_vehicles"):
            for vehicle in list(self.actor_vehicles):
                if vehicle is None:
                    continue
                try:
                    if vehicle.is_alive:
                        try:
                            vehicle.set_autopilot(False)
                        except Exception as e:
                            print(f"[WARN] Failed to set actor vehicle to autopilot: {e}")
                        vehicle.destroy()
                except RuntimeError as re:
                    # Happens if already destroyed on CARLA side
                    print(f"[WARN] Failed to destroy NPC vehicle: {re}")
                except Exception as e:
                    print(f"[WARN] Failed to destroy NPC vehicle id={vehicle.id}: {e}")

            # verify cleanup with world tick (important in synchronous mode)
            try:
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
            except Exception as ex:
                 print(f"[WARN] Failed to tick after NPC vehicle: {ex}")

            # empty the NPC vehicles list
            self.actor_vehicles = []

        # stop and destroy walker controllers
        for controller in list(getattr(self, "walker_controllers", [])):
            try:
                controller.stop()
                controller.destroy()
            except Exception as e:
                print(f"[WARN] Failed to destroy walker_controllers: {e}")

        # empty the walker controllers list
        self.walker_controllers = []

        # stop and destroy walkers
        for walker in list(getattr(self, "walkers", [])):
            try:
                walker.destroy()
            except Exception as e:
                print(f"[WARN] Failed to destroy walkers: {e}")

        # empty the walkers list
        self.walkers = []

        # tick to ensure cleanup
        try:
            for _ in range(3):
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                time.sleep(0.1)
        except Exception:
            pass

        # used in testing, render pygame window
        if self.SHOW_PYGAME:
            # quit pygame display but only if we created
            try:
                pygame.display.quit()
                pygame.quit()
                self.display = None

            except Exception as e:
                print(f"[WARN] Failed to quit pygame: {e}")

        # lastly notify the cleaning up completed
        print("[INFO] Cleanup complete.")

        # sanity check for remainig actors after cleaning
        self.debug_list_actors('after')
        # print(f'[Remaining after clean-up]: {remaining}')

    # getting ERROR: failed to destroy actor 280 : std::exception from server, to make sure - sanity check 
    def debug_list_actors(self, text):
        all_actors = self.world.get_actors()
        vehicle_count = len(all_actors.filter("vehicle.*"))
        sensor_count = len(all_actors.filter("sensor.*"))
        walker_count = len(all_actors.filter("walker.*"))
        controller_list = all_actors.filter("controller.ai.walker")

        print(f"[DEBUG] Remaining actors {text} : total={len(all_actors)},\
                vehicles={vehicle_count}, \
                sensors={sensor_count}, \
                walkers={walker_count}\
                controllers: {len(controller_list)}")

        return all_actors

    # SPAWN traffic, NPCs, walkers controller and walkers
    def spawn_traffic(self, num_vehicles=20, num_walkers=10): 

        # --- VEHICLES ---
        vehicle_bps = self.blueprint_library.filter('vehicle.*charger*')
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        tm = self.client.get_trafficmanager(8000)
        tm.set_synchronous_mode(False)
        # Hybrid physics mode -- issues
        #tm.set_hybrid_physics_mode(True) 
        tm.global_percentage_speed_difference(0)

        self.actor_vehicles = []
        for i, sp in enumerate(spawn_points[:num_vehicles]):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            try:
                vehicle = self.world.try_spawn_actor(bp, sp)
                if vehicle:
                    vehicle.set_autopilot(True, tm.get_port())
                    tm.distance_to_leading_vehicle(vehicle, 2.0)
                    self.actor_vehicles.append(vehicle)
            except Exception as e:
                print(f"[WARN] vehicle spawn failed at index {i}: {e}")

        print(f"[INFO] Spawned {len(self.actor_vehicles)} vehicles.")

        # --- WALKER CONTROLLER ---
        walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        spawn_points = []
        for _ in range(num_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_points.append(carla.Transform(loc + carla.Location(z=1)))

        # --- WALKERS ---
        walker_batch = []
        walker_speeds = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                walker_speeds.append(float(walker_bp.get_attribute('speed').recommended_values[1]))  # walking speed
            else:
                walker_speeds.append(1.4)
            walker_batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        walker_results = self.client.apply_batch_sync(walker_batch, True)
        walker_ids = [res.actor_id for res in walker_results if res.error is None or res.error == '']

        # tick to ensure setup
        try:
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception:
            pass
        time.sleep(1)

        # sanity check
        if len(walker_ids) == 0:
            print("[WARN] No walkers were spawned by batch.")

        # Spawn controllers
        controller_batch = [
            carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id)
            for walker_id in walker_ids
        ]
        controller_results = self.client.apply_batch_sync(controller_batch, True)
        controller_ids = [res.actor_id for res in controller_results if res.error is None or res.error == '']

        # Start controllers
        all_ids = walker_ids + controller_ids
        all_actors = self.world.get_actors(all_ids)

        # assign walkers to controller
        for i in range(len(walker_ids)):
            wid = walker_ids[i]
            cid = controller_ids[i] if i < len(controller_ids) else None
            walker = all_actors.find(wid)
            controller = all_actors.find(cid) if cid is not None else None
            if controller is not None:
                try:
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(walker_speeds[i])
                    self.walker_controllers.append(controller)
                except Exception as e:
                    print(f"[WARN] Failed to start walker controller {cid}: {e}")
            if walker is not None:
                self.walkers.append(walker)

        # Save for cleanup
        self.walkers = [self.world.get_actor(wid) for wid in walker_ids if self.world.get_actor(wid)]
        self.walker_controllers = [self.world.get_actor(cid) for cid in controller_ids if self.world.get_actor(cid)]

        # sanity check
        print(f"[INFO] Spawned {len(self.actor_vehicles)} vehicles, {len(self.walkers)} walkers, "
            f"{len(self.walker_controllers)} controllers.")

    # helper function to attach sensors to ego vehicle - used during reset
    def attach_sensor(self, sensor_type, transform, attributes=None):

        # some sensors has attributes while attaching
        attributes = attributes or {}
        # call sensor blueprint accordingly
        bp = self.blueprint_library.find(sensor_type)
        # assing attirbutes accordingly
        for k, v in attributes.items():
            try:
                bp.set_attribute(k, str(v))
            except Exception:
                print(f"[WARN] could not set attr {k} on {sensor_type}")
        # if sensor fails, return none
        sensor = None
        try:
            # spwan sensor accordingly / attach to ego vehicle
            sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
            # break to process
            time.sleep(0.3)
            # if sensor found add sensor to the list
            if sensor:
                self.sensor_actors.append(sensor)
        except Exception as e:
            print(f"Failed to spawn sensor in attach sensor {sensor_type}: {e}")

        # return the sensor type accordingly
        return sensor

    # to handle semantic camera image
    def camera_callback(self, image):

        """Lightweight conversion of CARLA Image to numpy BGRA array and store it under a lock.
           This avoids holding direct CARLA image objects across threads which can cause crashes.
        """
        # to compare the sensor data in observations
        #self.image_frame = image.frame

        try:
            # If sensor is semantic, convert to cityscapes color to have a visual RGB mapping:
            try:
                # convert image to CityScapes colors
                image.convert(carla.ColorConverter.CityScapesPalette)

            except Exception as e:
                print(f'[Failed] Converting CityScapes: {e}')

            # raw_data is bytes in BGRA format after conversion; reshape to H,W,4
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))

            # [DEBUG]
            # print(f"[DEBUG] Raw image max: {arr.max()}, min: {arr.min()}, shape: {arr.shape}")
            #arr = arr[:, :, :3]  # keep RGB [WARN] Rendering error: Buffer length does not equal format and resolution size
            
            # threading lock
            with self.image_lock:
                # store a copy so CARLA can free its internal buffer safely
                self.latest_image = arr.copy()
                #self.latest_image = image *****

        except Exception as e:
            print(f"[WARN] camera_callback error: {e}")

    # show pygame - used while testing the agent
    def render_pygame(self):

        # handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close_pygame = True
                break

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:  
                self.close_pygame = True
                break

        # Rendering (read the numpy image copy under lock)
        with self.image_lock:
            img = None if self.latest_image is None else self.latest_image.copy()

        if img is not None:
            # ignore the channel
            h, w, _ = img.shape
            # pygame expects a bytes-like object and format; we used BGRA layout from CARLA raw
            try:
                # converts numpy array to raw bytes -- img.tobytes()
                surface = pygame.image.frombuffer(img.tobytes(), (w, h), 'BGRA')
                # draw the image
                self.display.blit(surface, (0, 0))

            except Exception as e:
                print(f"[WARN] Rendering error: {e}")
        
        # flip display -- update
        pygame.display.flip()

    # handle IMU sensor data - not used in training
    # def imu_callback(self, data):
    #     try:
    #         self.latest_imu_data = (
    #             f'IMU:\nAccel: ({data.accelerometer.x:.2f}, {data.accelerometer.y:.2f}, {data.accelerometer.z:.2f})'
    #         )
    #     except Exception:
    #         self.latest_imu_data = ""

    # handle collision sensor data
    def collision_handler(self, event):

        self.collision_happened = True

        # self.collision_event = event  # store raw event for debug
        # store frame/time to compare
        # try:
        #     self.collision_frame = event.frame
        # except Exception:
        #     self.collision_frame = None

        # read impact impulse vector (the impulse applied to ego)
        ni = event.normal_impulse
        # unit vector
        n_vec = np.array([ni.x, ni.y, ni.z], dtype=np.float32)
        # magnitude of the event impulse
        n_norm = np.linalg.norm(n_vec)

        # compare and avoid zero division
        if n_norm > 1e-6:
            # normalized unit vector
            n_hat = n_vec / n_norm

        else:
            # fallback to direction from ego to other
            ego_loc = np.array([self.vehicle.get_location().x,
                                self.vehicle.get_location().y,
                                self.vehicle.get_location().z])
            other_loc = np.array([event.other_actor.get_location().x,
                                event.other_actor.get_location().y,
                                event.other_actor.get_location().z])
            n_hat = (other_loc - ego_loc)
            n_norm2 = np.linalg.norm(n_hat)
            # add tiny value and avoid zero division
            # normalized unit vector
            n_hat = n_hat / (n_norm2 + 1e-6)

        # ego velocity
        v_ego = np.array([self.vehicle.get_velocity().x,
                        self.vehicle.get_velocity().y,
                        self.vehicle.get_velocity().z], dtype=np.float32)
        # other vehicle velocity
        v_other = np.array([event.other_actor.get_velocity().x,
                            event.other_actor.get_velocity().y,
                            event.other_actor.get_velocity().z], dtype=np.float32)
        # relative velocity
        v_rel = v_ego - v_other
        # calculate normal dot vector
        # whether ego was moving toward or away from the other object along the collision axis
        v_rel_along_normal = float(np.dot(v_rel, n_hat))
        
        # decide fault using thresholds
        V_THRESH = 1.0  # m/s, tune this
        # if the static object is hit then we caused it
        if event.other_actor.type_id.startswith("static"):
            # print(f'Static is crashed...')
            self.accident_caused_by_us = True
        # hitting a pedestrian- then we caused it
        elif event.other_actor.type_id.startswith("walker"):               
            self.accident_caused_by_us = True
        # if the ego moving toward other
        elif v_rel_along_normal > V_THRESH:
            self.accident_caused_by_us = True
        # other moving toward ego
        elif v_rel_along_normal < -V_THRESH:
            self.accident_caused_by_us = False
        else:
            # ambiguous: use impulse magnitude & actor type as tiebreaker
            # if impact impulse large and ego speed > other speed, blame ego
            if n_norm > 100.0 and np.linalg.norm(v_ego) > np.linalg.norm(v_other):
                self.accident_caused_by_us = True
            # else:
            #     self.accident_caused_by_us = None  # unknown / shared this is a bit problematic None for model

        # store impact magnitude for reward calculation
        self.collision_impact = float(n_norm)
        
        # [DEBUG] print debug - trackalong with training and testing
        print(f"[COLLISION] other={event.other_actor.type_id} id={event.other_actor.id} "
            f"v_rel_n={v_rel_along_normal:.2f} impact={n_norm:.2f} caused_by_us={self.accident_caused_by_us}")

        # useful debug info
        # self.collision_info = {
        #     "other_id": event.other_actor.id,
        #     "other_type": event.other_actor.type_id,
        #     "impact_impulse_norm": float(n_norm),
        #     "v_ego_norm": float(np.linalg.norm(v_ego)),
        #     "v_other_norm": float(np.linalg.norm(v_other)),
        #     "v_rel_along_normal": v_rel_along_normal
        # }

    # handle lane invasion detector -- solid line
    def lane_handler(self, data):
        try:
            # to compare the sensor data in observations
            #self.lane_frame = data.frame
            # store lane invasion detection
            self.latest_lane_data = data
            # self.lane_invasion_timer = time.time() *****
        except Exception:
            pass

    # handle obstacle detector
    def obstacle_detector_handler(self, data):

        try:
            # reset detection
            if data is None or data.distance <= 0.0:
                self.detected_actor = None
            # store detection
            else:                
                self.detected_actor = data
                # self.detection_timer = time.time() *****
        except Exception:
            self.detected_actor = None

    # track the traffic light state and the distance to the light
    def update_traffic_light_status(self, vehicle):
        try:
            # get light relative to the ego
            tl = vehicle.get_traffic_light()
            if tl is not None:
                self.last_traffic_light = tl
                # light state: red, green, yellow
                self.last_traffic_light_state = tl.get_state().name

                # keep updating the distance to traffic light
                tl_location = tl.get_location()
                # actual distance to light
                self.tl_distance = tl_location.distance(vehicle.get_location())

                # store the initial distance -- used in reward calculation
                if self.initial_tl_distance is None:
                    self.initial_tl_distance = self.tl_distance

            # once there is no light; Ahead
            else:
                self.last_traffic_light = None
                self.last_traffic_light_state = "Ahead"
                self.tl_distance = None
                
        # reset if any error
        except Exception:
            self.last_traffic_light = None
            self.last_traffic_light_state = "None"
            self.tl_distance = None

  
    # # vector helper
    # def normalize_vector(self, vector):
    #     mag = (vector.x**2 + vector.y**2 + vector.z**2)**0.5
    #     if mag == 0:
    #         return None
    #     return carla.Vector3D(vector.x / mag, vector.y / mag, vector.z / mag)

    # def is_same_direction(self, actor_velocity, actor_transform, ego_forward):
    #     actor_unit = self.normalize_vector(actor_velocity)
    #     if actor_unit is None:
    #         actor_unit = self.normalize_vector(actor_transform.get_forward_vector())
    #     if actor_unit is None or ego_forward is None:
    #         return "unknown"
    #     dot = ego_forward.x * actor_unit.x + ego_forward.y * actor_unit.y + ego_forward.z * actor_unit.z
    #     if dot > 0.5:
    #         return "same"
    #     elif dot < -0.5:
    #         return "opposite"
    #     else:
    #         return "perpendicular or unclear"

    # def get_nearby_actors_by_direction(self, radius=50.0):
    #     try:
    #         # get our location and forward vector (direction)
    #         ego_location = self.vehicle.get_location()
    #         ego_forward = self.vehicle.get_transform().get_forward_vector()
    #         # get all actors
    #         all_actors = self.vehicle.get_world().get_actors()
    #         # empty lists to store vaious direction actors
    #         front_actors, rear_actors, opposite_direction_actors = [], [], []

    #         for actor in all_actors:
    #             # skip our vehicle
    #             if actor.id == self.vehicle.id:
    #                 continue

    #             if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.pedestrian.'):
    #                 actor_location = actor.get_location()
    #                 distance = actor_location.distance(ego_location)

    #                 # get actors in given radius 50m
    #                 if distance <= radius:
                        
    #                     # set actors physics to true closer to our vehicle -- did not work as expected
    #                     #actor.set_simulate_physics(True)

    #                     # if the distance less than 30m calculate direction
    #                     if distance <= 30:

    #                         direction_vector = actor_location - ego_location
    #                         try:
    #                             direction_vector = direction_vector.make_unit_vector()
    #                         except Exception:
    #                             pass
                            
    #                         # calculate direction
    #                         dot = ego_forward.x * direction_vector.x + ego_forward.y * direction_vector.y + ego_forward.z * direction_vector.z
    #                         actor_velocity = actor.get_velocity()
    #                         actor_transform = actor.get_transform()
    #                         direction = self.is_same_direction(actor_velocity, actor_transform, ego_forward)

    #                         # store both front and rear actors if in the same direction
    #                         if dot > 0.3 and direction == 'same':
    #                             front_actors.append((actor, distance))
    #                         elif dot < -0.3 and direction == 'same':
    #                             rear_actors.append((actor, distance))
    #                         elif direction == 'opposite':
    #                             opposite_direction_actors.append((actor, distance))
                        
    #                 # set actors physics to false far to our vehicle
    #                 # else:
    #                 #     actor.set_simulate_physics(False)
    #         # sort actors based on distances
    #         front_actors.sort(key=lambda x: x[1])
    #         rear_actors.sort(key=lambda x: x[1])
    #         opposite_direction_actors.sort(key=lambda x: x[1])

    #         # return listed actors
    #         return front_actors, rear_actors, opposite_direction_actors
        
    #     except Exception:
    #         return [], [], []
        
    # to track the progress in reward calculation and update the observation
    def calculate_remaining_route_distance(self):
        try:
            # if the route is missing return 0
            if not self.lp or not hasattr(self.lp, '_waypoints_queue'):
                return 0.0

            # get waypoints from local planner object
            waypoints = self.lp._waypoints_queue
            # end of route
            if len(waypoints) < 2:
                return 0.0

            total_distance = 0.0
            # store the location of ego
            prev_location = self.vehicle.get_location()

            # compute distance from waypoints
            for wp, _ in waypoints:
                loc = wp.transform.location
                segment = loc.distance(prev_location)
                total_distance += segment
                prev_location = loc

            # return the remaining distance
            return total_distance
        
        except Exception as e:
            print(f"[WARN] Failed to calculate route distance: {e}")
            return 0.0

    # OBSERVATIONS; image and all components of state vector with sequential buffer
    def get_observation(self):

        # Semantic segmentation image
        with self.image_lock:
            # store latest image, if no image returns zeros in the same shape of the image
            img = self.latest_image.copy() if self.latest_image is not None else np.zeros((300, 400, 4), dtype=np.uint8)

        # Convert BGRA → RGB, drop alpha
        img = img[:, :, :3][:, :, ::-1]
        #print("Obs  image shape 1:", img.shape)
        # downsample the size 
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)

        # Make sure dtype and range are correct both during initialization and get observation
        # then normalize in feature_extractor.forward()
        img = img.astype(np.uint8)  # SB3 expects uint8, [0–255] range
        # (H, W, C) → (C, H, W) SB3 expects images in PyTorch format → (C, H, W)
        img = img.transpose(2, 0, 1)  
        # After transpose, before return:
        # assert img.shape == (3, 84, 84), f"[ERROR] Unexpected image shape: {img.shape}"
        # print(f"[DEBUG] Final image shape: {img.shape}, dtype: {img.dtype}, max={img.max()}, min={img.min()}")

        # Low-dimensional state - Carla style
        v = self.vehicle.get_velocity()
        self.speed = (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
   
        # get remaining distance
        remaining_route_distance = self.calculate_remaining_route_distance()
        # call function to update the lights
        self.update_traffic_light_status(self.vehicle)        
        # Reset initial traffic light distance if no light detected
        # once the vehicle is fast, the distance calculation and detection goes crazy
        if self.last_traffic_light is None:            
            self.initial_tl_distance = None
        # if vehicle is too fast and drifts, cannot catch the lights either
        if hasattr(self, "tl_distance") and self.tl_distance is not None:
            # different range of traffic lights -- normalize the distance between 0 - 1          
            tl_distance_norm = np.clip(self.tl_distance / max((self.initial_tl_distance), 1.0), 0.0, 1)
        else:
            tl_distance_norm = 1.0  # assume no light nearby

        # coditions for light states
        tl_state = getattr(self, "last_traffic_light_state", "Ahead") # no light
        tl_is_red = 1.0 if tl_state == "Red" else 0.0 # red -> 1 | no red -> 0
        tl_is_yellow = 1.0 if tl_state == "Yellow" else 0.0 # yellow -> 1 | no yellow -> 0
        # consider green, yellow and ahead as Ahead
        tl_ahead = 1.0 if tl_state in ["Green", "Yellow", "Ahead"] else 0.0 # Ahead -> 1 | no Ahead -> 0
        
        # front obstacle distance (from detector)
        if self.detected_actor is not None:
            # get the distance to object
            self.front_distance = np.clip(self.detected_actor.distance, 0.0, self.detector_distance)
            # raise object detection
            object_present = 1.0
        else:
            # otherwise use sensor range as distance
            self.front_distance = self.detector_distance
            # no object present in front
            object_present = 0.0

        # [DEBUG] printing during the testing, uncomment
        # throttle, brake = self.last_action
        # if self.total_steps % 15 == 0:
        #     print()
        #     type_name = None
        #     if self.detected_actor is not None:
        #         type_name = self.detected_actor.other_actor.type_id

        #     print(f'Object present: {object_present} in Distance {self.front_distance} type_of_object: {type_name}')
        #     print('-------------------------------------------------------------')
        #     print(f"[DEBUG] Trf_L State: {self.last_traffic_light_state}, Trf_L Distance Norm: {tl_distance_norm:.2f}")
        #     print('-------------------------------------------------------------')
        #     print(f'speed: {round(self.speed, 3)} | throttle {throttle}, brake {brake}')
        
        # normalize the distance according to sensor range
        normalized_obj_distance = self.front_distance / self.detector_distance  # 0 - 1
        # reset detection after event
        self.detected_actor = None

        # store solid lane invasion
        # Add lane invasion flag (1 if solid line crossed recently, else 0)
        lane_violation = 0.0
        if self.latest_lane_data:
            for marking in self.latest_lane_data.crossed_lane_markings:
                # only solid line violation considered
                if marking.type == carla.LaneMarkingType.Solid:
                    # raise the lane violation
                    lane_violation = 1.0
                    break

        # check collision flag
        collision_flag = 1.0 if self.collision_happened else 0.0

        # store all in to state vector
        state = np.array(
            [   
                # current steering value from local planner
                self.current_steer, # -1, + 1
                # normalize speed
                self.speed /30.0, # 0 - 1 OR a bit more than 1 due to over speed
                # normalize route distance 
                remaining_route_distance / 200, # not stable due to random route distances
                # traffic light states 
                tl_distance_norm,
                tl_is_red, # 0 - 1
                tl_is_yellow, # 0 - 1
                tl_ahead, # green, yellow, ahead
                # front object distance
                normalized_obj_distance,
                object_present, # 0 - 1
                lane_violation, # 0 - 1
                collision_flag # 0 - 1
                ], 
                dtype=np.float32)

        # add sequential data to observations - 8
        # store to state buffer
        self.state_buffer.append(state)
        # Convert buffer to np.array: shape (seq_len, state_dim)
        state_seq = np.array(self.state_buffer, dtype=np.float32)

        # return all observations in a disctionary format
        return {"image": img, "state": state_seq}

    # REWARD calculation and EPISODE finalization returns: [reward, terminated, truncated, info]
    def compute_reward_done_info(self):
        # reset values
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        # set speed target to ego vehicle
        target_speed = 30.0
        # tolerance to speed, experimented some
        tolerance = 2.0  

        # --- Driving behavior ---
        if hasattr(self, "last_action"):
            # get throttle and brake values
            throttle, brake = self.last_action
            # store in to info for logging during the trainig
            info["throttle"] = throttle
            # store in to info for logging during the trainig
            info["brake"] = brake

            # 20 - 5 = 15m experimented with other values too
            # to check the proximity
            safe_dist = self.detector_distance - 5

            # distance to object
            # Once the distance to object is less safe distance (15m)
            if self.front_distance < safe_dist:
                # --- CAUTION ZONE ---
                if self.detected_actor is not None:
                    # reduce detection for static objects, by the side of the road
                    if self.detected_actor.other_actor.type_id.startswith("static"):
                        # calculate danger; relative to distance to object, less danger due to static object -- normalized
                        danger = ((safe_dist - self.front_distance) / safe_dist) / 2  # 0 -> 1                    

                # objects detected on the road
                else:
                    # relative to distance to the object -- normalized
                    danger = (safe_dist - self.front_distance) / safe_dist  # 0 → 1

                # calculate reward relative to brake and throttle values accordingly
                # encourage braking, discourage throttle
                reward += 8.0 * danger * brake
                reward -= 8.0 * danger * throttle

                # extra penalty if both pressed together -- conflicting action
                if throttle > 0.001 and brake > 0.001:
                    reward -= 5.0 * danger * brake
                    reward -= 5.0 * danger * throttle
                    info["conflicting_action"] = "Throttle+Brake in danger zone"

                info["obstacle_warning"] = True

            # out of safe distance
            else:
                # --- CLEAR ZONE ---
                # encourage throttle, discourage brake while speed less than target speed
                if self.speed <= target_speed: # 30
                    reward += 1 * throttle # constant to experiment                               

                # reward for being close to target speed
                elif self.speed <= target_speed + tolerance:    # 32  
                    # start penalty for throttle
                    reward -= 1.0 * throttle
                    # start motivating for braking
                    reward += 1.0 * brake    

                # over speeding
                elif self.speed >= target_speed + tolerance:  # over 32
                    # heavy penalty if speeding
                    # rewarding related to brake / throttle / speed
                    reward -= throttle * (10.0 * (self.speed - (target_speed + tolerance)) / target_speed)
                    # print(f'OVER SPEEDING penalty {10.0 * (self.speed - (target_speed + tolerance)) / target_speed}')
                    # store over speed value to info for logging - tracking during the training
                    info["overspeed"] = self.speed

                # again, penalize pressing both (bad driving habit)
                if brake > 0.001 and throttle > 0.001 :
                    reward -= throttle * 2
                    reward -= brake * 2
                    # store to track the conflicting action
                    info["conflicting_action"] = "Throttle+Brake on clear road"
                  
                # --- Traffic Light Compliance ---
                # get latest light state
                tl_state = self.last_traffic_light_state
                # store distance to light
                tl_dist_org = getattr(self, "tl_distance", None)

                # set the initial distance
                if self.initial_tl_distance is None:
                    self.initial_tl_distance = tl_dist_org

                # reward situation once the light state is red
                if tl_state == "Red" and self.initial_tl_distance  is not None:
                    # constant 3 used after experiments, not every light appears in the same proximity
                    # distance in reducing manner
                    tl_dist = (3 - (self.initial_tl_distance  - tl_dist_org))
                    # close to stop line -> must reduce speed - less than 3m
                    if tl_dist is not None and tl_dist <= 3.0:
                        # final case, if the vehicle stops
                        if self.speed > 0.5:
                            # rewarding brake
                            reward += brake * 2.0
                            # penalizing the gas
                            reward -= (throttle * 2.0) +1 # +1 to avoid very small throttle
                    
                    # if inside the intersection, less than 2m and speed is more than 0.5 -> failure
                    if tl_dist is not None and tl_dist <= 2.0 and self.speed > 0.5:
                        # penalty related to speed / distance and throttle
                        reward -= (tl_dist * self.speed * throttle) +1 # if the throttle is 0
                        # reward related to speed / distance and throttle
                        reward += (tl_dist * self.speed * brake) 
                        # store to the logs - considered as violation
                        info["traffic_light_violation"] = "Red"

                    # once too late to stop; penalize and terminate
                    if tl_dist is not None and tl_dist < 0.1 and self.speed > 0.05:                        
                        reward -= (tl_dist * self.speed * throttle) *1.5
                        info["terminated_reason"] = "Red light violation"
                        terminated = True            

                # situation for yellow light
                elif tl_state == "Yellow":
                    # encourage slowing down
                    if throttle > 0.1 and self.speed > 5.0:
                        reward -= 5.0
                        info["traffic_light_warning"] = "Yellow"
                    else:
                        # speed less than 5; cautious yellow response, a bit reward
                        reward += 1.0  
                
                # once turned from red to green, sometimes stuck, motivate for green
                elif tl_state == "Green":
                    reward += throttle * 10
        
        # --- Steering safety penalty ---
        # avoid sharp steering
        # get the absolute value; -1 to +1
        abs_steer = abs(self.current_steer)
        speed = self.speed
        # experimented these thresholds, seems ok
        if abs_steer > 0.4 and speed > 20.0:  
            penalty_turn = (abs_steer - 0.4) * (speed - 20.0) * 0.5
            reward -= penalty_turn
            # info["steering_penalty"] = penalty_turn # was to track
            # info["risky_turn"] = True # stored earlier but not tracked any more

        # --- Progress ---
        # Initialize prev_distance if missing, to track the progress
        if not hasattr(self, "prev_distance") or self.prev_distance is None:
            self.prev_distance = self.calculate_remaining_route_distance()        
        # update the remaining distance
        current_distance = self.calculate_remaining_route_distance()
        # track the progress
        progress = self.prev_distance - current_distance
        # clamp negative due to CARLA noise
        progress = max(progress, 0.0)
        # store to the logs
        info["progress"] = progress
        # update the reward
        reward += progress * 10.0 # experimented with various values            
        # update the distance
        self.prev_distance = current_distance

        # --- Stuck detection ---
        if progress < 0.05:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0  # reset if moving
        # if no progress in 2500 steps, truncate case 1
        if self.no_progress_steps > 2500:
            truncated = True
            info["truncated_reason"] = "stuck"

        # --- Speed control ---
        if hasattr(self, "speed"):
            # reward a bit with speed,keep magnitude small
            reward += np.clip(self.speed / 30.0, 0.0, 1.0) * 0.5
            # store to logs for tracking the normal speed
            info["speed"] = self.speed          

        # --- Collision penalty --- 
        # treshold for collision impact
        HIGH_IMPACT_THRESH = 50
        if self.collision_happened:
            # get collision impact
            impact = getattr(self, "collision_impact", None)
            if impact is None:
                # compute from stored normal impulse
                impact = np.linalg.norm([self.collision_normal_impulse.x,
                                        self.collision_normal_impulse.y,
                                        self.collision_normal_impulse.z])
            # scale penalties according to, if we are the guilty or not
            if self.accident_caused_by_us is True: 
                penalty = min(impact * 0.2, 50.0)
            # other party is guilty, small penalty 
            elif self.accident_caused_by_us is False:
                penalty = min(impact * 0.05, 5.0)   # weaker
            # else:
            #     penalty = min(impact * 0.1, 10.0)   # ambiguous
            # update reward
            reward -= penalty

            # store collision impact to logs to track
            info["collision_impact"] = float(impact)
            info["accident_caused_by_us"] = "accident_caused_by_us"
            
            # if impact is higher than treshold
            # store collision info according to their type and terminate episode
            if impact > HIGH_IMPACT_THRESH: 
                info["terminated_reason"] = "high_impact_collision"
                info['collided_id'] = self.collision_info["other_id"]
                # track the accidents according to their types
                if  self.collision_info["other_type"].startswith('walker'):
                    info['collided_walker'] = "walker"
                elif self.collision_info["other_type"].startswith('static'):
                    info['collided_static'] = "static"
                # there was bug here, which caused to crash during the training, later fixed
                # that is why missing from the logs
                elif self.collision_info["other_type"].startswith('vehicle'):
                    info['collided_vehicle'] = "vehicle"
                else:
                    info['collided_other'] = "other"
                # only terminate for high-impact collisions
                terminated = True                
                time.sleep(0.01) # avoid immediate resetting 
            # reset flags so next step doesn't re-penalize
            self.collision_happened = False
            self.accident_caused_by_us = False         

        # --- Lane violation ---
        # if cross the solid lane
        if self.latest_lane_data:
            for marking in self.latest_lane_data.crossed_lane_markings:
                if marking.type == carla.LaneMarkingType.Solid:
                    # print("[DEBUG] Solid lane crossed!")
                    reward -= 50 
                    # store to the logs
                    info["lane_violation"] = "Solid line crossed"
                    info["terminated_reason"] = "Solid line crossed"
                    terminated = True                    
                    break

        # --- Limit the episode, truncate case 2 ---
        # get the value of step_moved, if greater than 5000, truncate the episode
        if getattr(self, "step_moved", 0) > getattr(self, "max_steps", 5000):
            # store to the logs
            info["truncated_reason"] = "max_steps_exceeded"
            truncated = True

        # try to use brake when route is about the end
        # due to speed, it causes either collision or solid line cross
        if current_distance < 1.0:
            # print(f'Last 1 meter...')
            reward += brake * (1 - current_distance)

        # render the image -- true for testing the agent
        if self.SHOW_PYGAME:
            self.render_pygame()
            if self.close_pygame:
                print('Pygame window is closed...')
                info["terminated_reason"] = "pygame_closed"
                terminated = True                

        # --- Completion of the route ---
        if self.lp.done():
            # give a good reward
            reward += 150
            # store to logs
            info["terminated_reason"] = "Route has completed"
            info["COMP_reward"] = reward
            info["success_route_total_waypoints"] = len(self.route)
            terminated = True

        # return collected values
        return reward, terminated, truncated, info

    # once an exception or failure happens, return zero observations
    def _safe_zero_obs(self):
        """
        Return a fallback zero observation with correct shapes for MultiInputPolicy,
        in case sensors/vehicle are invalid.
        """
        # --- Image ---
        img_shape = (3, 84, 84)  # (C, H, W)
        zero_img = np.zeros(img_shape, dtype=np.uint8)

        # --- State vector ---
        # state has 11 components
        self.current_state_dim = 11 
        # state buffer length
        self.seq_len = 8

        # For sequential state with 8 previous steps, create deque of zeros
        zero_state_single = np.zeros(self.current_state_dim, dtype=np.float32)
        zero_state_seq = np.repeat(zero_state_single[None, :], self.seq_len, axis=0)

        # return as expected in dictionary format
        return {"image": zero_img, "state": zero_state_seq}