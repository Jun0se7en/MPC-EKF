import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class KinematicBicycleModel:
    def __init__(self, length_rear: float, wheelbase_length: float, delta_time: float):
        self.length_rear = length_rear
        self.wheelbase_length = wheelbase_length
        self.delta_time = delta_time

    def discrete_kbm(self, x, y, heading, velocity, steering_angle):
        slip_angle = np.arctan((self.length_rear / self.wheelbase_length) * np.tan(steering_angle))

        new_x = x + velocity * np.cos(heading + slip_angle) * self.delta_time
        new_y = y + velocity * np.sin(heading + slip_angle) * self.delta_time
        new_heading = heading + (velocity / self.wheelbase_length) * np.tan(steering_angle) * self.delta_time

        return new_x, new_y, new_heading

    def angular_velocity_to_steering_angle(self, angular_velocity, velocity):
        if velocity == 0:
            return 0
        return np.arctan((angular_velocity * self.wheelbase_length) / velocity)

# Parameters
length_rear = 1.05
wheelbase_length = 2.1
dt = 0.1  # Time step

# Initialize model
kbm = KinematicBicycleModel(length_rear, wheelbase_length, dt)

# Cost function for MPC
def cost_function(u, state, path, N, dt):
    cost = 0
    x, y, heading = state
    for i in range(N):
        steering_angle = u[i * 2]
        velocity = u[i * 2 + 1]
        x, y, heading = kbm.discrete_kbm(x, y, heading, velocity, steering_angle)
        cost += (x - path[i][0])**2 + (y-path[i][1])**2
        cost += (steering_angle**2 + velocity**2)
    return cost

# MPC controller
def mpc_control(state, path, N, dt):
    u0 = np.zeros(2 * N)
    bounds = [(-np.pi / 2, np.pi / 2), (-100, 100)] * N
    result = minimize(cost_function, u0, args=(state, path, N, dt), bounds=bounds, method='SLSQP')
    # if result.success:
    #     return result.x[:2]
    # else:
    #     raise ValueError("MPC optimization failed")
    return result.x[:2]

# Simulation
state = [0, 0, np.deg2rad(0)]
# Số lượng điểm
num_points = 1000

# Tạo mảng x từ 0 đến 2*pi
x = np.linspace(0, 100, num_points)

def f(x):
    return 0.0000001 * (x**10 - 25*x**8 + 210*x**6 - 700*x**4 + 1000*x**2)

# Tính giá trị y = sin(x)
y = f(x)

# Lưu các điểm x, y vào danh sách
path = []
for i in range(len(x)):
    path.append((x[i], y[i]))
path = np.array(path)
print(path.shape)
N = 5
states = [state]

for i in range(len(path) - N):
    steering_angle, velocity = mpc_control(state, path[i:i + N], N, dt)
    state = kbm.discrete_kbm(*state, velocity, steering_angle)
    states.append(state)

states = np.array(states)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(path[:, 0], path[:, 1], 'r--', label="Reference Path")
plt.plot(states[:, 0], states[:, 1], 'b-', label="Vehicle Path")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("MPC with Kinematic Bicycle Model")
plt.legend()
plt.grid()
plt.show()
