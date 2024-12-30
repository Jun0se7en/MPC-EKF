import numpy as np
from kinematic_bycicle_model import KinematicBicycleModel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import math

# Hàm normalize để điều chỉnh góc về khoảng [-pi, pi]
def normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def find_closest_and_next_points(current_position, reference_path, horizon = 3):
    """
    Find the closest reference point and the next target for MPC.
    
    Args:
        current_position (tuple): The current (x, y) position.
        reference_path (list): A list of reference points [(x1, y1), (x2, y2), ...].
        
    Returns:
        tuple: The closest point and the next target point.
    """
    distances = [np.linalg.norm(np.array(current_position) - np.array(point)) for point in reference_path]
    closest_index = np.argmin(distances)
    closest_point = reference_path[closest_index]
    if closest_index + 1 > len(reference_path):
        return None
    if closest_index + horizon > len(reference_path):
        return closest_point, reference_path[closest_index + 1:]
    else:
        return closest_point, reference_path[closest_index + 1: closest_index+horizon+1]

def calculate_heading(x1, y1, x2, y2):
    """
    Calculate the heading (angle) between two points in 2D space.

    Parameters:
    - x1, y1: Coordinates of the first point.
    - x2, y2: Coordinates of the second point.

    Returns:
    - heading (float): Heading angle in degrees, where:
      - 0 degrees points to the positive x-axis,
      - 90 degrees points to the positive y-axis,
      - 180 degrees points to the negative x-axis,
      - -90 degrees points to the negative y-axis.
    """
    delta_x = x2 - x1
    delta_y = y2 - y1

    # Calculate the angle in radians
    heading_rad = math.atan2(delta_y, delta_x)

    return normalize(heading_rad)

wheelbase = 1.05
lengthrear = 2.1
dt = 0.28
horizon = 3
kbm = KinematicBicycleModel(wheelbase, lengthrear, dt)
# Bounds
# speed = [1, 2, 3, 4, 5]
# steering_angle = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
speed_bounds = (0.5, 3)
steering_angle_bounds = (-np.radians(25), np.radians(25))

current_state = [0, 0, 0.0] # x, y, heading

df = pd.read_csv('./transformed_route_points.csv')
ref_x = df['x'].values
ref_y = df['y'].values

# ref_x = np.linspace(0, 50, 100)

# def f(x):
#     return np.sin(x)

# ref_y = f(ref_x)

path = []

for i in range(len(ref_x)):
    path.append((ref_x[i], ref_y[i]))

def objective(controls, cur_state, closest_point, ref_path):
    cost = 0
    weights = [0, 0, 1, 1, 0.2] # vel, steer, x, y, heading
    x, y, heading = cur_state
    for i in range(len(ref_path)):
        control_vel = controls[i]
        control_steering = controls[len(ref_path) + i]
        ref_x, ref_y = ref_path[i]
        cost_vel = weights[0] * control_vel ** 2
        cost_steer = weights[1] * control_steering ** 2
        cost_x = weights[2] * (ref_x - x) ** 2
        cost_y = weights[3] * (ref_y - y) ** 2
        if i == 0:
            ref_heading = calculate_heading(closest_point[0], closest_point[1], ref_x, ref_y)
        else:
            ref_heading = calculate_heading(ref_path[i-1][0], ref_path[i-1][1], ref_x, ref_y)
        cost_heading = weights[4] * (ref_heading - heading) ** 2
        cost += (cost_vel + cost_steer + cost_x + cost_y + cost_heading) / np.sum(weights)
        x, y, heading = kbm.discrete_kbm(control_vel, control_steering, x, y, heading)
    return cost

x, y, heading = current_state

traj_x, traj_y = [x], [y]

for i in range(500):
    
    closest_point, ref_path = find_closest_and_next_points((x, y), path, horizon)
    print(ref_path)
    vars_init = np.concatenate([
        np.linspace(speed_bounds[0], speed_bounds[1], len(ref_path)),
        np.linspace(steering_angle_bounds[0], steering_angle_bounds[1], len(ref_path)),
    ])

    bounds = [speed_bounds] * len(ref_path) + [steering_angle_bounds] * len(ref_path)

    result = minimize(
        objective, vars_init, args=((x, y, heading), closest_point, ref_path),
        bounds=bounds, method='SLSQP', options={'disp': False}
    )

    optimal_vars = result.x
    optimal_speed = np.clip(optimal_vars[0], speed_bounds[0], speed_bounds[1])
    optimal_steer = np.clip(optimal_vars[len(ref_path)], steering_angle_bounds[0], steering_angle_bounds[1])
    print('Speed:', optimal_speed, 'Angle:', optimal_steer)

    x, y, heading = kbm.discrete_kbm(optimal_speed, optimal_steer, x, y, heading)
    traj_x.append(x)
    traj_y.append(y)
    # print(optimal_vars)

plt.figure(figsize=(10, 10))
plt.plot(ref_x, ref_y, label="Reference Path", marker="o")
plt.plot(traj_x, traj_y, label="MPC Trajectory", marker="o")
plt.scatter(traj_x, traj_y, c='red', label="MPC Positions")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("MPC Trajectory Following Reference Path")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
        


