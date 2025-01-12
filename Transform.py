import numpy as np
import math

class Transform():
    def __init__(self):
        self.a = 6378137.0  # semi-major axis, meters
        self.f = 1 / 298.257223563  # flattening
        self.e2 = 2 * self.f - self.f ** 2  # eccentricity squared

        self.ref_lla = [10.87050, 106.802, 0]
    
    def enu_to_wgs84(self, enu):
        lat_origin_rad = np.deg2rad(self.ref_lla[0])
        lon_origin_rad = np.deg2rad(self.ref_lla[1])

        delta_lat = enu[1] / self.a
        delta_lon = enu[0] / (self.a * np.cos(lat_origin_rad))

        return [np.rad2deg(lat_origin_rad + delta_lat), np.rad2deg(lon_origin_rad + delta_lon), 0]

    def wgs84_to_enu(self, lla):
        lat_origin_rad = np.deg2rad(self.ref_lla[0])
        lon_origin_rad = np.deg2rad(self.ref_lla[1])
        lat_rad = np.deg2rad(lla[0])
        lon_rad = np.deg2rad(lla[1])

        delta_lat = lat_rad - lat_origin_rad
        delta_lon = lon_rad - lon_origin_rad

        return [self.a * np.cos(lat_origin_rad) * delta_lon, self.a * delta_lat, 0]


if __name__ == '__main__':
    transform = Transform()
    lla = [10.86975,106.80233,0]
    tmp = transform.wgs84_to_enu(lla)
    print(tmp)
    tmp = transform.enu_to_wgs84(tmp)
    print(tmp)