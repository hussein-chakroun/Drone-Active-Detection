import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import cm
import time

class Drone3D:
    def __init__(self, position, goal, obstacles, moving_obstacles, grid_size=50):
        
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.moving_obstacles = moving_obstacles
        self.grid_size = grid_size
        
        
        self.max_speed = 2.0  
        self.max_acceleration = 0.8  
        self.mass = 1.2  
        self.drag_coefficient = 0.3
        self.gravity = np.array([0, 0, -0.1])  
        self.battery_level = 100.0  
        self.battery_drain_rate = 0.05  
        
        
        self.detection_radius = 8.0  
        self.prediction_horizon = 8  
        
        
        self.path = []
        self.path_history = [self.position.copy()]
        self.waypoint_radius = 0.8  
        
        
        self.replanning = False
        self.avoidance_count = 0
        self.goal_reached = False
        self.last_replanning_step = -10  
        self.collision_risk_level = 0 
        self.original_path = []  
        self.eps = 1e-5
        
    def update_physics(self):
        
        
        force = self.gravity * self.mass
        
        
        propulsion = self.acceleration * self.mass
        
        
        if np.linalg.norm(self.velocity) > 0:
            drag_direction = -self.velocity / np.linalg.norm(self.velocity)
            drag_magnitude = self.drag_coefficient * np.linalg.norm(self.velocity)**2
            drag = drag_direction * drag_magnitude
        else:
            drag = np.zeros(3)
        
        
        total_force = force + propulsion + drag
        acceleration = total_force / self.mass
        
        
        self.velocity += acceleration
        
        
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
            
        
        self.position += self.velocity
        
        
        for i in range(3):
            if self.position[i] < 0:
                self.position[i] = 0
                self.velocity[i] = 0
            elif self.position[i] > self.grid_size:
                self.position[i] = self.grid_size
                self.velocity[i] = 0
                
        
        self.battery_level -= self.battery_drain_rate * (1 + np.linalg.norm(self.acceleration))
        self.battery_level = max(0, self.battery_level)
    def detect_obstacles(self):
        self.collision_risk_level = 0
        risk_detected = False
        for obs in self.obstacles + self.moving_obstacles:
            distance = np.linalg.norm(self.position - obs['position'])
            if distance < self.detection_radius:
                risk = (1 - (distance / self.detection_radius)) * 10
                self.collision_risk_level = max(self.collision_risk_level, risk)
                risk_detected = True
        return risk_detected
    
    def predict_collisions(self, lookahead=None):
        # Updated to handle obstacle boundary bouncing
        if lookahead is None:
            lookahead = self.prediction_horizon
            
        future_positions = []
        future_velocity = self.velocity.copy()
        future_position = self.position.copy()
        
        for i in range(lookahead):
            future_position += future_velocity
            future_positions.append(future_position.copy())
        
        for obs in self.moving_obstacles:
            obs_pos = obs['position'].copy()
            obs_vel = obs['velocity'].copy()
            
            for i, drone_future_pos in enumerate(future_positions):
                # Simulate obstacle movement with boundary checks
                obs_pos += obs_vel
                for dim in range(3):
                    if obs_pos[dim] <= 0 or obs_pos[dim] >= self.grid_size:
                        obs_vel[dim] *= -1
                        obs_pos[dim] = np.clip(obs_pos[dim], 0, self.grid_size)
                
                distance = np.linalg.norm(drone_future_pos - obs_pos)
                if distance < obs['radius'] + 1.5:
                    return True, obs, i+1
        return False, None, 0
    
    def avoid_obstacle(self, obstacle=None, time_to_collision=5):
        if obstacle:
            avoidance_vector = self.position - obstacle['position']
            distance = np.linalg.norm(avoidance_vector)
            
            if distance < self.eps:  # Handle zero-distance case
                avoidance_vector = np.random.randn(3)
                distance = np.linalg.norm(avoidance_vector)
                
            avoidance_vector /= distance
            # Remove fixed upward bias, use obstacle relative position
            if obstacle['position'][2] > self.position[2]:
                avoidance_vector[2] -= 0.2  # Push down if obstacle is above
            else:
                avoidance_vector[2] += 0.2  # Push up if obstacle is below
                
            avoidance_vector /= np.linalg.norm(avoidance_vector)
            avoidance_strength = self.max_speed * (1.0 / max(1, time_to_collision))
            self.velocity = avoidance_vector * avoidance_strength
            self.avoidance_count += 1
            self.replanning = True
            return True
        return False
    
    def move_to_goal(self):
        # Removed manual velocity updates - only set acceleration
        if len(self.path) > 0:
            next_point = self.path[0]
            direction = next_point - self.position
            distance = np.linalg.norm(direction)
            
            if distance < self.waypoint_radius:
                self.path.pop(0)
                return
                
            if distance > self.eps:
                direction /= distance
                target_velocity = direction * self.max_speed
                acceleration_direction = target_velocity - self.velocity
                acceleration_magnitude = np.linalg.norm(acceleration_direction)
                
                if acceleration_magnitude > self.eps:
                    acceleration_direction /= acceleration_magnitude
                    self.acceleration = acceleration_direction * min(acceleration_magnitude, self.max_acceleration)
                else:
                    self.acceleration = np.zeros(3)
        else:
            direction = self.goal - self.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                target_velocity = direction * self.max_speed
                
                
                acceleration_direction = target_velocity - self.velocity
                acceleration_magnitude = np.linalg.norm(acceleration_direction)
                
                if acceleration_magnitude > 0:
                    acceleration_direction = acceleration_direction / acceleration_magnitude
                    self.acceleration = acceleration_direction * min(acceleration_magnitude, self.max_acceleration)
                else:
                    self.acceleration = np.zeros(3)
    
    def plan_path_rrt_star(self, max_iterations=1500, step_size=2.0, goal_sample_rate=0.1, search_radius=10.0):
        """Enhanced RRT* algorithm for better path planning"""
        start = self.position.copy()
        goal = self.goal.copy()
        
        
        tree = [{'position': start, 'parent': None, 'cost': 0.0}]
        
        for i in range(max_iterations):
            
            if random.random() < goal_sample_rate:
                sample = goal
            else:
                sample = np.array([
                    random.uniform(0, self.grid_size),
                    random.uniform(0, self.grid_size),
                    random.uniform(0, self.grid_size)
                ])
            
            
            distances = [np.linalg.norm(node['position'] - sample) for node in tree]
            nearest_idx = np.argmin(distances)
            nearest = tree[nearest_idx]
            
            
            direction = sample - nearest['position']
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                new_position = nearest['position'] + direction * min(step_size, distance)
            else:
                continue  # Skip this iteration if the direction vector is zero
                
                
            collision = False
            for obs in self.obstacles + self.moving_obstacles:
                # Predict obstacle position
                if 'velocity' in obs:  # Moving obstacle
                    steps_to_reach = int(np.linalg.norm(new_position - self.position) / (self.max_speed + self.eps)
                    predicted_pos = obs['position'] + obs['velocity'] * steps_to_reach
                    obs_pos = predicted_pos
                else:
                    obs_pos = obs['position']
                
                if np.linalg.norm(new_position - obs_pos) < obs['radius'] + 1.0:
                    collision = True
                    break
        # ...[Remaining RRT* code]...
    
    def safe_waypoint(self, point):
        # New method to check waypoint safety
        for obs in self.obstacles + self.moving_obstacles:
            if np.linalg.norm(point - obs['position']) < obs['radius'] + 2.0:
                return False
        return True

# In the animate function:
    # ...[In the replanning section]...
    if temp_goal is not None:
        # Check if original path waypoints are still safe
        safe_waypoints = [p for p in drone.original_path if drone.safe_waypoint(p)]
        if len(safe_waypoints) > 0:
            drone.path.extend(safe_waypoints)

# Update moving obstacles function remains same

# Battery emergency handling (add to update_physics)
    if self.battery_level < 20 and not self.goal_reached:
        # Implement return-to-home
        self.goal = np.array([5.0, 5.0, 5.0])  # Simple example
        if len(self.path) == 0:
            self.plan_path_rrt_star()

# Visualization optimizations (in animate function):
    # Reduce mesh resolution for obstacles
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]  # Was 20j/10j
    # ...[Remaining plotting code]...
