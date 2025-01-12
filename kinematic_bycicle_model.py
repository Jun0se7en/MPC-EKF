import numpy as np

class KinematicBicycleModel:
    def __init__(self, length_rear: float, delta_time: float):
        self.delta_time = delta_time
        self.length_rear = length_rear

    def discrete_kbm(self,x: float, y: float, heading: float,velocity: float,angular_velocity: float, dt: float):

        if(velocity == 0):

            slip_angle = 0

        else:
            tmp = np.radians((angular_velocity * self.length_rear) / velocity)
            if tmp < -1:
                tmp = -1
            elif tmp > 1:
                tmp = 1
            slip_angle = np.arcsin(tmp)
        
        if heading < 0:
            heading = -np.deg2rad(abs(heading))
        else:
            heading = np.deg2rad(heading)

        new_x = x + velocity * np.cos(heading + slip_angle) * dt

        new_y = y + velocity * np.sin(heading + slip_angle) * dt

        new_heading = heading + angular_velocity * dt
        
        return new_x, new_y, new_heading ,slip_angle
