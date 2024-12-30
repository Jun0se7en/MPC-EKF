import numpy as np

# Hàm normalize để điều chỉnh góc về khoảng [-pi, pi]
def normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Lớp mô hình xe đạp kinematik
class KinematicBicycleModel:
    def __init__(self, wheelbase: float, length_rear: float, delta_time: float):
        self.delta_time = delta_time
        self.wheelbase = wheelbase
        self.length_rear = length_rear

    def discrete_kbm(self, velocity: float, steering: float, x: float, y: float, heading: float):
        slip_angle = np.arctan(self.length_rear * np.tan(steering) / self.wheelbase)
        # new_velocity = velocity + acceleration * self.delta_time
        # new_steer = steering + steering_rate * self.delta_time
        angular_velocity = (velocity * np.tan(steering) * np.cos(slip_angle)) / self.wheelbase
        new_x = x + velocity * np.cos(slip_angle + heading) * self.delta_time
        new_y = y + velocity * np.sin(slip_angle + heading) * self.delta_time
        new_heading = normalize(heading + angular_velocity * self.delta_time)
        return new_x, new_y, new_heading
