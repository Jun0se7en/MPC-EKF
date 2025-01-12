import numpy as np
import matplotlib.pyplot as plt
from extended_kalman_filter import ExtendedKalmanFilter
from kinematic_bycicle_model import KinematicBicycleModel
import pandas as pd
from Transform import Transform
import time
import folium
import webbrowser
import os
transform = Transform()

df = pd.read_csv('./Retrieving_Data1.csv')
data = df.values

def calculate_gnss_heading(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # Calculate the difference in longitude
    delta_lon = lon2 - lon1
    # Calculate the initial bearing
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    initial_bearing = np.atan2(x, y)
    # Convert bearing from radians to degrees
    initial_bearing = np.degrees(initial_bearing)
    # Normalize the bearing to 0-360 degrees
    heading = (initial_bearing + 360) % 360
    return heading

# Thiết lập mô phỏng
np.random.seed(0)
length_rear = 1.05  # chiều dài từ trục sau đến trọng tâm, tính bằng mét
delta_time = 0.01  # khoảng thời gian giữa các lần lấy mẫu
ekf = ExtendedKalmanFilter(length_rear, delta_time)
kbm = KinematicBicycleModel(length_rear, delta_time)
# Thông số mô phỏng
sigma_x, sigma_y, sigma_h = 0.5, 0.5, 30

# Ma trận hiệp phương sai ban đầu và nhiễu quá trình
# P_cov = np.eye(3)  # Khởi tạo ma trận hiệp phương sai
P_cov = np.array([
    [0.1, 0, 0],
    [0, 0.1, 0],
    [0, 0, 0.1],
])

Q_cov = np.array([
    [0.00005**2,0,0],
    [0,0.0005**2,0],
    [0,0, 30**2]
])  # Ma trận nhiễu quá trình, với heading tính bằng độ


#---------------------------------------------------------------------------
#GNSS
lat = data[0][1]
lon = data[0][2]
alt = 0.0
target_lla = [lat, lon, alt]
map = folium.Map(location=[lat, lon], zoom_start=6)
transform.ref_lla = target_lla
enu = transform.wgs84_to_enu(target_lla)
#---------------------------------------------------------------------------
#---------------------------init
x, y= enu[1], -enu[0]
heading = 0 
#--------------------------input
velocity = data[0][3]
angular_velocity = data[0][7]
#---------------------------plot data
measurements = []
predicted_states = []
updated_states=[]
states=[]
#---------------------------------------------------------------------------
prev_lat = lat
prev_lon = lon
prev_lat_x = lat
prev_lon_y = lon
counter = 1
offset = 0
heading_arr = []
#---------------------------------------------------------------------------
# for i in range(1, data.shape[0]):
#     counter = counter + 1
#     if((prev_lat != data[i][1] or prev_lon != data[i][2]) and (data[i][1] != 0 and data[i][2] != 0)):
#         start = time.time()
#         heading_gps = calculate_gnss_heading(prev_lat,prev_lon,data[i][1],data[i][2])
#         print('Calculate Delay:', time.time()-start)
#         print('Heading GPS:',heading_gps)
#         print('Heading BNO:',data[i][5])
#         offset = heading_gps - data[i][5]
#         print(offset)
#         heading = (data[i][5] + offset) % 360
#         velocity = data[i][3]
#         angular_velocity = data[i][7]
#         print(heading)
#         break
# print(counter)

coordinates = []
for i in range(counter+1, data.shape[0]):
    angular_velocity = data[i][7]

    velocity = data[i][3]
    lat = data[i][1]
    lon = data[i][2]
    alt = 0.0
    predicted_state, predicted_err = ekf.prediction_state(P_cov, Q_cov,x, y, heading, velocity,angular_velocity, 0.16)
    predicted_states.append(predicted_state)
    
    if (int(lat) != 0 and int(lon) != 0 ) and (lat != prev_lat or lon != prev_lon):
        ### Offset ###
        # if len(heading_arr) > 0:
        #     tmp = max(heading_arr) - min(heading_arr)
        #     if  tmp < 10 or tmp > 350:
        #         offset = calculate_gnss_heading(prev_lat,prev_lon,lat,lon) - data[i][5]
        #         print('Update Offset:', offset)
        # heading_arr = []
        # heading_meas = (data[i][5] + offset) % 360
        ###############################
        # heading_meas = data[i][5]
        # alt = 0.0 
        # target_lla = [lat, lon, alt]
        # enu = transform.wgs84_to_enu(target_lla)
        # x_meas = enu[1]
        # y_meas = -enu[0]
        # measurements.append([x_meas,y_meas,heading_meas])
        # updated_state, updated_err = ekf.update_state(
        # x_meas, y_meas,heading_meas, sigma_x, sigma_y,sigma_h, predicted_state, predicted_err)
        # x_new, y_new, heading = updated_state
        # # Xoay x, y một góc 90 độ theo chiều kim
        # angle = np.deg2rad(60)
        # x += (x_new-x) * np.cos(angle) + (y_new-y) * np.sin(angle)
        # y += -(x_new-x) * np.sin(angle) + (y_new-y) * np.cos(angle)
        # P_cov = updated_err
        # updated_states.append(updated_state)
        prev_lat = lat
        prev_lon = lon 
    else:
        # heading_arr.append(data[i][5])
        # heading = (data[i][5] + offset) % 360
        heading = data[i][5]
        x_new, y_new = predicted_state[:2]
        # Xoay x, y một góc 90 độ theo chiều kim
        angle = np.deg2rad(60)
        x += (x_new-x) * np.cos(angle) + (y_new-y) * np.sin(angle)
        y += -(x_new-x) * np.sin(angle) + (y_new-y) * np.cos(angle)
        P_cov = predicted_err
        predicted_states.append([x, y,heading])
    
    tmp_lla = transform.enu_to_wgs84([-y, x, 0])
    coordinates.append([tmp_lla[0], tmp_lla[1]])

### Draw ###

for coord in coordinates:
    folium.CircleMarker(
        location=coord,
        radius=2,
        color="red",
        stroke=False,
        fill=True,
        fill_opacity=0.6,
        opacity=1,
        popup="{} pixels".format(1),
        tooltip="I am in pixels",
    ).add_to(map)
# folium.PolyLine(
#     locations=coordinates,
#     color="#FF0000",
#     weight=5,
#     tooltip="From A to B",
# ).add_to(map)

### Show ###

# Lưu bản đồ dưới dạng file HTML
map_file = "map.html"
map.save(map_file)

# Mở file HTML trong trình duyệt mặc định
webbrowser.open(f"file://{os.path.abspath(map_file)}")

################################################
    
# Chuyển đổi dữ liệu để vẽ
predicted_states = np.array(predicted_states)
measurements = np.array(measurements)
updated_states = np.array(updated_states)
# Vẽ đồ thị kết quả
plt.figure(figsize=(10, 8))
if len(predicted_states) > 0:
    plt.scatter(predicted_states[:, 1], predicted_states[:, 0], label="Predicted Position", color="red",s=5)
if len(updated_states) > 0:
    plt.scatter(updated_states[:, 1], updated_states[:, 0], label="updated_state", color="blue",s=6)
if len(measurements) > 0:
    plt.scatter(measurements[:, 1], measurements[:, 0], label="measurements", color="green",s=7)
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.title("EKF Position Estimation with GNSS Measurements")
plt.grid()
plt.show()
