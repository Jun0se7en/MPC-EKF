import numpy as np
import matplotlib.pyplot as plt
from extended_kalman_filter import ExtendedKalmanFilter
from kinematic_bycicle_model import KinematicBicycleModel
import pandas as pd
from Transform import Transform
import time
import math
transform = Transform()

df = pd.read_csv('./Data/New_Data2.csv')
data = df.values

def calculate_gnss_heading(lat1, lon1, lat2, lon2):
    """
    Tính heading giữa hai điểm dựa trên tọa độ LLA.
    Đầu vào:
        lat1, lon1: Vĩ độ và kinh độ điểm 1 (đơn vị: độ)
        lat2, lon2: Vĩ độ và kinh độ điểm 2 (đơn vị: độ)
    Đầu ra:
        heading: Góc hướng giữa điểm 1 và điểm 2 (đơn vị: độ, đo từ Bắc theo chiều kim đồng hồ)
    """
    # Chuyển đổi sang radian
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Tính toán chênh lệch kinh độ và các thành phần cần thiết
    delta_lon = lon2 - lon1

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

    # Tính góc và chuyển đổi sang độ
    initial_heading = math.atan2(x, y)
    heading_degrees = math.degrees(initial_heading)

    # Đảm bảo góc trong khoảng [0, 360)
    return (heading_degrees + 360) % 360

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
    [0.5**2,0,0],
    [0,0.5**2,0],
    [0,0, 30**2]
])  # Ma trận nhiễu quá trình, với heading tính bằng độ


#---------------------------------------------------------------------------
#GNSS
lat = data[0][1]
lon = data[0][2]
alt = 0.0
target_lla = [lat, lon, alt]
ned = transform.lla_to_ned(target_lla)
#---------------------------------------------------------------------------
#---------------------------init
x, y= ned[0], ned[1]
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
for i in range(1, data.shape[0]):
    counter = counter + 1
    if((prev_lat != data[i][1] or prev_lon != data[i][2]) and (data[i][1] != 0 and data[i][2] != 0)):
        start = time.time()
        heading_gps = calculate_gnss_heading(prev_lat,prev_lon,data[i][1],data[i][2])
        print('Calculate Delay:', time.time()-start)
        print('Heading GPS:',heading_gps)
        print('Heading BNO:',data[i][5])
        offset = heading_gps - data[i][5]
        print(offset)
        heading = (data[i][5] + offset) % 360
        velocity = data[i][3]
        angular_velocity = data[i][7]
        print(heading)
        break
print(counter)

offset = 270

for i in range(counter+1, data.shape[0]):
    angular_velocity = data[i][7]

    velocity = data[i][3]
    lat = data[i][1]
    lon = data[i][2]
    alt = 0.0
    predicted_state, predicted_err = ekf.prediction_state(P_cov, Q_cov,x, y, heading, velocity,angular_velocity, 0.28)
    predicted_states.append(predicted_state)
    
    if (int(lat) != 0 and int(lon) != 0 ) and (lat != prev_lat or lon != prev_lon):
        if len(heading_arr) > 0:
            tmp = max(heading_arr) - min(heading_arr)
            # if  tmp < 10 or tmp > 350:
                # offset = calculate_gnss_heading(prev_lat,prev_lon,lat,lon) - data[i][5]
                # print('Update Offset:', offset)
        heading_arr = []
        heading_meas = (data[i][5] + offset) % 360
        alt = 0.0 
        target_lla = [lat, lon, alt]
        ned = transform.lla_to_ned(target_lla)
        x_meas = ned[0]
        y_meas = ned[1]
        measurements.append([x_meas,y_meas,heading_meas])
        print('Update:', x_meas, y_meas)
        updated_state, updated_err = ekf.update_state(
        x_meas, y_meas,heading_meas, sigma_x, sigma_y,sigma_h, predicted_state, predicted_err)
        x, y, heading = updated_state
        P_cov = updated_err
        prev_lat = lat
        prev_lon = lon 
        updated_states.append(updated_state)
    else:
        heading_arr.append(data[i][5])
        heading = (data[i][5] + offset) % 360
        x, y = predicted_state[:2]
        P_cov = predicted_err
        predicted_states.append([x, y,heading])
    
# Chuyển đổi dữ liệu để vẽ
predicted_states = np.array(predicted_states)
measurements = np.array(measurements)
updated_states = np.array(updated_states)
# Vẽ đồ thị kết quả
plt.figure(figsize=(10, 8))
plt.scatter(predicted_states[:, 1], predicted_states[:, 0], label="Predicted Position", color="red",s=5)
plt.scatter(updated_states[:, 1], updated_states[:, 0], label="updated_state", color="blue",s=6)
plt.scatter(measurements[:, 1], measurements[:, 0], label="measurements", color="green",s=7)
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.title("EKF Position Estimation with GNSS Measurements")
plt.grid()
plt.show()
