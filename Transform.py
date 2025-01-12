import numpy as np
import math
import pandas as pd

class Transform():
    def __init__(self):
        self.a = 6378137.0  # WGS-84 Earth semimajor axis (m)
        self.b = 6356752.314245 # Derived Earth semiminor axis (m)
        self.f = (self.a - self.b) / self.a # Ellipsoid Flatness
        self.f_inv = 1 / self.f  # Inverse flattening

        self.a_sq = self.a ** 2
        self.b_sq = self.b ** 2
        self.e_sq = self.f * (2 - self.f)  # Square of Eccentricity

        self.ref_lla = [10.87050, 106.802, 0]
    
    def wgs84_to_ecef(self, lla):
        lat_rad = np.deg2rad(lla[0])
        lon_rad = np.deg2rad(lla[1])
        s = np.sin(lat_rad)
        N = self.a / math.sqrt(1 - self.e_sq * s * s)

        x = (lla[2] + N) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (lla[2] + N) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (lla[2] + (1 - self.e_sq) * N) * np.sin(lat_rad)
        return [x, y, z]

    def ecef_to_wgs84(self, ecef):
        x = ecef[0]
        y = ecef[1]
        z = ecef[2]
        eps = self.e_sq / (1 - self.e_sq)
        p = math.sqrt(x * x + y * y)
        q = math.atan2((z * self.a), (p * self.b))
        sin_q = np.sin(q)
        cos_q = np.cos(q)
        sin_q_3 = sin_q ** 3
        cos_q_3 = cos_q ** 3
        lat = math.atan2((z + eps * self.b * sin_q_3), (p - self.e_sq * self.a * cos_q_3))
        lon = math.atan2(y, x)
        v = self.a / math.sqrt(1.0 - self.e_sq * np.sin(lon) * np.sin(lon))
        alt = (p / np.cos(lon)) - v
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
        return [lat, lon, alt]

    def ecef_to_enu(self, ecef):
        lat_origin_rad = np.deg2rad(self.ref_lla[0])
        lon_origin_rad = np.deg2rad(self.ref_lla[1])
        s = np.sin(lat_origin_rad)
        N = self.a / math.sqrt(1 - self.e_sq * s * s)

        sin_lat = np.sin(lat_origin_rad)
        cos_lat = np.cos(lat_origin_rad)
        sin_lon = np.sin(lon_origin_rad)
        cos_lon = np.cos(lon_origin_rad)

        x = ecef[0]
        y = ecef[1]
        z = ecef[2]

        x0 = (self.ref_lla[2] + N) * cos_lat * cos_lon
        y0 = (self.ref_lla[2] + N) * cos_lat * sin_lon
        z0 = (self.ref_lla[2] + (1 - self.e_sq) * N) * sin_lat

        xd = x-x0
        yd = y-y0
        zd = z-z0

        xE = -sin_lon * xd + cos_lon * yd
        yN = -cos_lon * sin_lat * xd - sin_lat * sin_lon * yd + cos_lat * zd
        zU = cos_lat * cos_lon * xd + cos_lat * sin_lon * yd + sin_lat * zd

        return [xE, yN, zU]

    def enu_to_ecef(self, enu):
        lat_origin_rad = np.deg2rad(self.ref_lla[0])
        lon_origin_rad = np.deg2rad(self.ref_lla[1])
        s = np.sin(lat_origin_rad)
        N = self.a / math.sqrt(1 - self.e_sq * s * s)

        sin_lat = np.sin(lat_origin_rad)
        cos_lat = np.cos(lat_origin_rad)
        sin_lon = np.sin(lon_origin_rad)
        cos_lon = np.cos(lon_origin_rad)

        xE = enu[0]
        yN = enu[1]
        zU = enu[2]

        x0 = (self.ref_lla[2] + N) * cos_lat * cos_lon
        y0 = (self.ref_lla[2] + N) * cos_lat * sin_lon
        z0 = (self.ref_lla[2] + (1 - self.e_sq) * N) * sin_lat

        xd = -sin_lon * xE - cos_lon * sin_lat * yN + cos_lat * cos_lon * zU
        yd = cos_lon * xE - sin_lat * sin_lon * yN + cos_lat * sin_lon * zU
        zd = cos_lat * yN + sin_lat * zU

        x = xd + x0
        y = yd + y0
        z = zd + z0

        return [x, y, z]

    def wgs84_to_enu(self, lla):
        tmp = self.wgs84_to_ecef(lla)
        res = self.ecef_to_enu(tmp)
        return res
    
    def enu_to_wgs84(self, enu):
        tmp = self.enu_to_ecef(enu)
        res = self.ecef_to_wgs84(tmp)
        return res


    # def enu_to_wgs84(self, enu):
    #     lat_origin_rad = np.deg2rad(self.ref_lla[0])
    #     lon_origin_rad = np.deg2rad(self.ref_lla[1])

    #     delta_lat = enu[1] / self.a
    #     delta_lon = enu[0] / (self.a * np.cos(lat_origin_rad))

    #     return [np.rad2deg(lat_origin_rad + delta_lat), np.rad2deg(lon_origin_rad + delta_lon), 0]

    # def wgs84_to_enu(self, lla):
    #     lat_origin_rad = np.deg2rad(self.ref_lla[0])
    #     lon_origin_rad = np.deg2rad(self.ref_lla[1])
    #     lat_rad = np.deg2rad(lla[0])
    #     lon_rad = np.deg2rad(lla[1])

    #     delta_lat = lat_rad - lat_origin_rad
    #     delta_lon = lon_rad - lon_origin_rad

    #     return [self.a * np.cos(lat_origin_rad) * delta_lon, self.a * delta_lat, 0]


if __name__ == '__main__':
    transform = Transform()
    lla = [10.86975,106.80233,0]
    tmp = transform.wgs84_to_enu(lla)
    print(tmp)
    tmp = transform.enu_to_wgs84(tmp)
    print(tmp)

    # df = pd.read_csv('route_points.csv')
    # data_df = df.to_numpy()
    # transform.ref_lla = [data_df[0][0], data_df[0][1], 0]
    # data = {'x': [],
    #         'y': []}
    # for i in data_df:
    #     tmp = transform.wgs84_to_enu([i[0], i[1], 0])
    #     data['x'].append(tmp[1])
    #     data['y'].append(-tmp[0])
    
    # df = pd.DataFrame(data)
    # df.to_csv('transformed_route_points.csv')