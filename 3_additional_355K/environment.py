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
    SAFE_DISTANCE = 30.0 # started with 40, reduced to 30 after training reduced to 20, then increased to 30

    def __init__(self,
                 host='localhost', port=2000, 
                 sync_mode=False, 
                 fixed_delta_seconds=0.05, 
                 show_pygame=False
                 ):
        
        super().__init__()

        # TOTAL STEP MODIFIED ACCORDING TO WHERE PREVIOUS TRAINING TOTAL STEPS ACCUMULATED
        # traffic will be spawned immediately
        self.total_steps = 1_509_952 
        
    # MODIFIED REWARD FUNCTION, rest of the code is same (according to best of my remembering)
    def compute_reward_done_info(self):
        # to organize logs new naming applied, to keep similar logs close to each other; easy to compare
        # reset values
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        # set speed target to ego vehicle
        target_speed = 30.0
        # tolerance to speed, experimented some
        tolerance = 2.0  

        # --- Speed control ---
        if hasattr(self, "speed"):
            # reward a bit with speed,keep magnitude small
            reward += np.clip(self.speed / 30.0, 0.0, 1.0) * 0.5
            # store to logs for tracking the normal speed
            info["a_speed"] = self.speed
          
        # --- Progress ---
        # [Initialize progress here and give reward accordingly]
        # [if there is no object in a certain proximity and no red light]
        # Initialize prev_distance if missing
        if not hasattr(self, "prev_distance") or self.prev_distance is None:
            self.prev_distance = self.calculate_remaining_route_distance()        
        ## update the remaining distance
        current_distance = self.calculate_remaining_route_distance()
        # track the progress
        progress = self.prev_distance - current_distance
        # clamp negative due to CARLA noise
        progress = max(progress, 0.0)

        # --- Driving behavior ---
        if hasattr(self, "last_action"):
            # get throttle and brake values
            throttle, brake = self.last_action
            # store in to info for logging during the trainig
            info["a_throttle"] = throttle
            # store in to info for logging during the trainig
            info["a_brake"] = brake

            # penalize pressing both immediately (bad driving habit)
            if brake > 0.001 and throttle > 0.001 :
                reward -= throttle * 13 # increased to 5 , was 2, later increased to 10
                reward -= brake * 13
                info["conflicting_action"] = "Throttle+Brake on clear road"

            # 30 - 5 = 25m experimented with other values too
            # to check the proximity
            safe_dist = self.detector_distance - 5
            # add reward until the object is in close proximity
            # only for reward calculation for the progress
            if self.front_distance >= 6:

                # good reward for progressing
                reward += progress * 15.0
                # small reward for speed
                # we keep this here to not make much change in previous reward logic, made modification    
                # was 0.5 once taken here increased to 0.7 to compensate some losses
                reward += np.clip(self.speed / 30.0, 0.0, 1.0) * 0.7 
                info["a_progress"] = progress

            # ONCE THE OBJECT is in less than safe distance 25m
            if self.front_distance < safe_dist:
                # --- CAUTION ZONE ---
                if self.detected_actor is not None:
                    info["obstacle_warning"] = True

                    # reduce detection for static objects, by the side of the road, which cause unnecessary stuck and braking
                    if self.detected_actor.other_actor.type_id.startswith("static"):
                        # danger is proximity from zero to one -- still reduced by dividing 20
                        danger = ((safe_dist - self.front_distance) / safe_dist) / 20  # increased to 4. was 2 than to 10, than to 15
                     
                    # objects detected on the road
                    else:
                        # danger is proximity -- this time use actual distance as danger -- not normalized
                        danger = (safe_dist - self.front_distance)             
                    
                    # extra close proximity case less than 3.5m
                    if self.front_distance <= 3.5:
                         # extra penalty for almost accident; relative to danger and speed
                        reward -= danger * self.speed
                         # extra reward by incrementing reward constant multiplier, to avoid almost accident
                        reward += 5 * danger * brake # constant reduced due to non-normalization

                    # distance less than 9m
                    elif self.front_distance <= 9.0:
                        # encourage braking by incrementing reward constant multiplier
                        reward += 3 * danger * brake # constant reduced due to non-normalization
                        # reduce speed effect due to non-normalization and penalize relative to speed and danger
                        reward -= danger * self.speed/3.0 
                    
                    # distance less than 12m
                    elif self.front_distance <= 12.0:
                        # encourage braking
                        reward += danger * brake
                        # reduce speed effect due to non-normalization and penalize relative to speed and danger
                        reward -= danger * self.speed/4.0

                    # extra penalty if both pressed together -- conflicting action
                    if throttle > 0.001 and brake > 0.001:
                        reward -= 5.0 * danger * brake # reduced due to actual values used for danger
                        reward -= 5.0 * danger * throttle # reduced due to actual values used for danger
                        info["conflicting_action"] = "Throttle+Brake in danger zone"
                        
            # ONCE THERE IS NO OBJECT in FRONT 25m
            else:
                # --- CLEAR ZONE ---
                # encourage throttle, while speed less than target speed
                if self.speed <= target_speed: # 30                    
                    reward += 3 * throttle                               

                # reward for being close to target speed
                elif self.speed <= target_speed + tolerance:    # 32
                    reward -= (self.speed - target_speed + tolerance) # if throttle is zero
                    reward -= 2.0 * throttle # increased to 2.0. was 1.0
                    reward += 1.0 * brake    

                # over speeding
                # once it is multiplied with throttle, agent figured out that when the throttle is 0, it is not dangerous, no negative reward :))
                elif self.speed >= target_speed + tolerance:  # over 32
                    # heavy penalty if speeding
                    # rewarding related to brake / throttle
                    reward -= throttle * (10.0 * (self.speed - (target_speed + tolerance)) / target_speed) # increased to 15, was 10
                    reward -= (self.speed - target_speed + tolerance) * 2 # if throttle is zero
                    info["a_overspeed"] = self.speed
                 
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
                    tl_dist = (3 - (self.initial_tl_distance  - tl_dist_org))

                    # avoid conlicting action at the beginnig
                    if brake > 0.001 and throttle > 0.001 :
                        reward -= throttle * 15 # increased to 5 , was 2, later increased to 10
                        reward -= brake * 15
                        info["conflicting_action"] = "Throttle+Brake on red light"

                    # must stop - less than 0.4m
                    if tl_dist is not None and tl_dist < 0.4 and self.speed > 0.05:
                        # not every light has same detection distance
                        # to avoid negative distance
                        if tl_dist < 0:
                            tl_dist = 0.1

                        # penalty relative to distance, speed and throttle
                        reward -= (tl_dist * self.speed * (throttle + 2)) # increased constant from 0.2 to 2
                        # good reward if using brake
                        reward += brake * 40 # was 20, increased to 40

                        # more reward if stops
                        if self.speed < 0.05:
                            reward += 10

                        # not anymore terminating, to allow the agent get more data, early termination might break the data
                        info["traffic_light_violation"] = "Red"

                    # if close to stop line, less than 2m
                    elif tl_dist is not None and tl_dist <= 2.0 and self.speed > 0.5:
                        # not every light has same detection distance
                        # to avoid negative distance
                        if tl_dist < 0:
                            tl_dist = 0.1
                        # modify throttle a bit more; so even if it is zero got peanlty due to speed
                        reward -= (tl_dist * self.speed * (throttle + 1)) 
                        # good reward for using brake
                        reward += brake * 35 # was 30, increased to 35

                    # in the light priximity
                    elif tl_dist is not None and tl_dist <= 3.0:
                        # if vehicle has a speed
                        if self.speed > 0.5:                            
                            reward += brake * 40.0 # increased to 5 was 2.0 /10 /40
                            # penalize speed and throttle; throttle incrementing constant increased to 1, was 0.2
                            reward -= (((self.speed / 5) * (throttle + 1)) * 2.0) + 1  


                # situation for yellow light
                elif tl_state == "Yellow":
                    # encourage slowing down
                    if self.current_throttle > 0.1 and self.speed > 5.0:
                        reward -= 5.0 * throttle
                        info["traffic_light_warning"] = "Yellow"
                    else:
                        # speed less than 5; cautious yellow response, a bit reward
                        reward += 1.0 
                
                # once turned from red to green, sometimes stuck, motivate for green
                elif tl_state == "Green":
                    reward += throttle * 12 # increased to 12, was 10
                # once there is not red and yellow light states, give progress reward
                elif tl_state == "Green" or tl_state == "Ahead":
                    reward += progress * 18.0 
                    # small reward for speed also
                    # we keep this here to not make much change in previous reward logic, made modification
                    reward += np.clip(self.speed / 30.0, 0.0, 1.0) * 0.7 # was 0.5 increased to 0.7
                    info["a_progress"] = progress

            # reset detected actor here, for precise conditions
            self.detected_actor = None

        # --- Steering safety penalty ---
        # avoid sharp steering
        # get the absolute value; -1 to +1
        abs_steer = abs(self.current_steer)
        speed = self.speed
        # experimented these thresholds, seems ok
        if abs_steer > 0.4 and speed > 20.0:  
            penalty_turn = (abs_steer - 0.4) * (speed - 20.0) * 0.5
            reward -= penalty_turn
            # info["steering_penalty"] = penalty_turn
            # info["risky_turn"] = True

        # --- Stuck detection ---
        if progress < 0.05:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0  # reset if moving
        # if no progress in 2500 steps, truncate case 1
        if self.no_progress_steps > 2500:
            truncated = True
            info["truncated_reason"] = "stuck"

        # update the distance
        self.prev_distance = current_distance

        # --- Collision penalty --- 
        # treshold for collision impact
        HIGH_IMPACT_THRESH = 50
        # get collision impact
        if self.collision_happened:
            impact = getattr(self, "collision_impact", None)
            if impact is None:
                # compute from stored normal impulse
                impact = np.linalg.norm([self.collision_normal_impulse.x,
                                        self.collision_normal_impulse.y,
                                        self.collision_normal_impulse.z])
            # very small if collided with the static bus; agent has not other option, if the route is planned so
            if self.accident_caused_by_us is True and self.collision_info["other_type"].startswith('static.bu'):
                penalty = min(impact * 0.2, 1.0)
            # scale penalties according to, if we are the guilty or not
            elif self.accident_caused_by_us is True: 
                penalty = min(impact * 0.2, 150.0)   # increased to 150, was 50, 100
            # other party is guilty, small penalty
            elif self.accident_caused_by_us is False:
                penalty = min(impact * 0.05, 5.0)   # weaker
            # experimented earlier use again, if the guilty party is not clear
            else:
                penalty = min(impact * 0.1, 100.0)   # ambiguous
            # update reward
            reward -= penalty

            # store collision impact to logs to track
            info["collision_impact"] = float(impact)
            info["accident_caused_by_us"] = "accident_caused_by_us"
            
            # if impact is higher than treshold
            # store collision info according to their type and terminate episode
            if impact > HIGH_IMPACT_THRESH: 
                info["terminated_reason"] = "high_impact_collision"
                # track the accidents according to their types
                if  self.collision_info["other_type"].startswith('walker'):
                    info['collided_walker'] = "walker"
                    
                elif self.collision_info["other_type"].startswith('static.bu'):
                    info['collided_bus'] = "bus_static"
                    
                elif self.collision_info["other_type"].startswith('static'):
                    info['collided_static'] = "static"

                elif self.collision_info["other_type"].startswith('vehicle'):
                    info['collided_vehicle'] = "vehicle"

                else:
                    info['collided_other'] = "other"

                # only terminate for high-impact collisions
                terminated = True                
                time.sleep(0.01) # avoid immediate resetting 
            # reset flag so next step doesn't re-penalize
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
            #  store to the logs
            info["truncated_reason"] = "max_steps_exceeded"
            truncated = True

        # try to use brake when route is about the end
        # due to speed, it causes either collision or solid line cross
        if current_distance < 1.0:           
            reward += brake * (1 - current_distance) * 10 # increased by multiplier 10
        
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
            reward += len(self.route)
            # store to logs
            info["terminated_reason"] = "Route has completed"
            info["COMP_reward"] = reward
            info["success_route_total_waypoints"] = len(self.route)
            terminated = True

        # return collected values
        return reward, terminated, truncated, info
