import cv2
import threading
import base64
import time
import numpy as np
import os
import sys
import json
import random
import ctypes
import pandas as pd

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Record,
    Config,
    CarControl,
    FilterGPS,
)
from src.templates.threadwithstop import ThreadWithStop
import socket
import json
import math
import signal
from lib.extended_kalman_filter import ExtendedKalmanFilter
from lib.Transform import Transform

class threadEKF(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, flag, debugger):
        super(threadEKF, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.debugger = debugger
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.flag = flag
        # self.client.send(self.speed, self.smooth_angle)
        self.subscribe()
        self.Configs()
        self.message = {}
        self.message_type = ""
        self.initial_flag = False
        self.transform = Transform()
        self.prev_lat = 0.0
        self.prev_lon = 0.0
        length_rear = 1.05  # chiều dài từ trục sau đến trọng tâm, tính bằng mét
        delta_time = 0.01  # khoảng thời gian giữa các lần lấy mẫu
        self.ekf = ExtendedKalmanFilter(length_rear, delta_time)
        # Thông số mô phỏng
        self.sigma_x, self.sigma_y, self.sigma_h = 0.0001, 0.0001, 0.0000005 # Nhiễu đo lường GNSS: 0.5m và 1 độ cho heading

        # Ma trận hiệp phương sai ban đầu và nhiễu quá trình
        # P_cov = np.eye(3)  # Khởi tạo ma trận hiệp phương sai
        self.P_cov = np.array([
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.1],
        ])

        self.Q_cov = np.array([
            [0.001**2,0,0],
            [0,0.001**2,0],
            [0,0, 1**2]
        ])  # Ma trận nhiễu quá trình, với heading tính bằng độ
        self.data = {
            "X": [],
            "Y": [],
        }
    
    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadEKF", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadEKF", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        super(threadEKF, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()
    
    # =============================== UTILITIES ============================================
    def calculate_gnss_heading(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        # Calculate the difference in longitude
        delta_lon = lon2 - lon1
        # Calculate the initial bearing
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        initial_bearing = math.atan2(x, y)
        # Convert bearing from radians to degrees
        initial_bearing = math.degrees(initial_bearing)
        # Normalize the bearing to 0-360 degrees
        heading = (initial_bearing + 360) % 360
        return heading
    
    def save_data_to_csv(self, csv_file):
        df = pd.DataFrame(self.data)

        df.to_csv(csv_file, mode='a', index=False, header=False)

        for key in self.data.keys():
            self.data[key].clear()

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies and then it send the data to process gateway."""   
        while self._running:
            ### RUNNING ###
            # print('EKF Running!!!')
            if not self.queuesList["EKFCarControlInfo"].empty():
                msg = self.queuesList["EKFCarControlInfo"].get()["msgValue"]
                msg = json.loads(msg)
                gps_latitude = msg['GPS_lat']
                gps_longitude = msg['GPS_long']
                speed = msg["vel"]
                steer = msg["steer"]
                heading = msg["head"]
                # acc = msg["acc"]
                if steer != 0:
                    angular_velocity = msg["angular_velocity"]
                else:
                    angular_velocity = 0

                if not self.initial_flag:
                    if (int(gps_latitude) != 0 and int(gps_longitude) != 0) and (gps_latitude != self.prev_lat or gps_longitude != self.prev_lon):
                        lat = gps_latitude
                        lon = gps_longitude
                        alt = 0.0
                        target_lla = [lat, lon, alt]
                        ned = self.transform.lla_to_ned(target_lla)
                        self.x, self.y= ned[0], ned[1]
                        self.heading = 0
                        self.prev_lat = gps_latitude
                        self.prev_lon = gps_longitude
                        # time_step = 0.28
                        # self.predicted_state, self.predicted_err = self.ekf.prediction_state(self.P_cov, self.Q_cov, self.x, self.y, self.heading, speed, angular_velocity, time_step)
                        # self.x, self.y, self.heading = self.predicted_state
                        # self.P_cov = self.predicted_err
                        self.initial_flag = True
                elif speed > 0:
                    # print('EKF Running!!!')
                    time_step = 0.28
                    self.predicted_state, self.predicted_err = self.ekf.prediction_state(self.P_cov, self.Q_cov, self.x, self.y, self.heading, speed, angular_velocity, time_step)
                    # self.x, self.y = self.predicted_state[:2]
                    # self.P_cov = self.predicted_err
                    # self.x, self.y, self.heading = self.predicted_state
                    # self.P_cov = self.predicted_err
                    if (int(gps_latitude) != 0 and int(gps_longitude) != 0) and (gps_latitude != self.prev_lat or gps_longitude != self.prev_lon):
                        heading_meas = self.calculate_gnss_heading(self.prev_lat,self.prev_lon,gps_latitude,gps_longitude)
                        # heading_meas = heading
                        lat = gps_latitude
                        lon = gps_longitude
                        alt = 0.0
                        target_lla = [lat, lon, alt]
                        ned = self.transform.lla_to_ned(target_lla)
                        x_meas = ned[0]
                        y_meas = ned[1]
                        updated_state, updated_err = self.ekf.update_state(
                        x_meas, y_meas, heading_meas, self.sigma_x, self.sigma_y, self.sigma_h, self.predicted_state, self.predicted_err)
                        self.x, self.y, self.heading = updated_state
                        self.P_cov = updated_err
                        self.prev_lat = gps_latitude
                        self.prev_lon = gps_longitude
                    else:
                        ned_convert_s = [self.x,self.y,0]
                        ned_convert_p = [self.predicted_state[0],self.predicted_state[1],0]
                        lla_convert_s = self.transform.ned_to_lla(ned_convert_s)
                        lla_convert_p = self.transform.ned_to_lla(ned_convert_p)
                        self.heading = self.calculate_gnss_heading(lla_convert_s[0],lla_convert_s[1],lla_convert_p[0],lla_convert_p[1])
                        self.x, self.y = self.predicted_state[:2]
                        self.P_cov = self.predicted_err
                    print('EKF Predict: ',self.predicted_state)
                    self.data['X'].append(self.x)
                    self.data['Y'].append(self.y)
                    output_csv_path = './EKF_Data.csv'
                    self.save_data_to_csv(output_csv_path)
                    lla = self.transform.ned_to_lla([self.x, self.y, 0.0])
                    data = {
                        "lat": lla[0],
                        "long": lla[1],
                    }
                    self.queuesList[FilterGPS.Queue.value].put(
                    {
                        "Owner": FilterGPS.Owner.value,
                        "msgID": FilterGPS.msgID.value,
                        "msgType": FilterGPS.msgType.value,
                        "msgValue": data,
                    })
            

    # =============================== START ===============================================
    def start(self):
        super(threadEKF, self).start()

        
