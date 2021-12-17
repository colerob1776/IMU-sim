import json
from dataclasses import dataclass, field
from math import asin, atan, pi
from typing import List
import numpy as np
from numpy import matrixlib
from numpy.matrixlib.defmatrix import matrix
import math

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    print(R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

@dataclass 
class INS:
    vYaw: float
    vPitch: float
    vRoll: float
    yaw: list[float] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    roll: list[float] = field(default_factory=list)
    DCM: matrix = field(default_factory=lambda: np.identity(3))
    Omega: matrix = field(default_factory=lambda: np.zeros(3))

    def updateDCM(self, velYaw: float, velPitch: float, velRoll: float, dT: float):
        dt = dT
        self.Omega = np.matrix([[1, -velYaw * dt, -velRoll * dt],
                                [velYaw * dt, 1, -velPitch * dt],
                                [velRoll * dt, velPitch * dt, 1]])
        self.DCM = np.matmul(self.DCM, self.Omega)
        return
    
    def normalizeDCM(self):
        rx = self.DCM[0].tolist()[0]
        ry = self.DCM[1].tolist()[0]
        e = np.dot(rx, ry)
        t0 = np.subtract(rx, np.multiply(ry, 0.5 * e))
        t1 = np.subtract(ry, np.multiply(rx, 0.5 * e))
        t2 = np.cross(t0, t1)
        self.DCM = np.matrix([t0, t1, t2])

    def updateAngles(self, velYaw: float, velPitch: float, velRoll: float, dT: float):
        self.updateDCM(velYaw, velPitch, velRoll, dT)
        self.normalizeDCM()

        # save Euler Angles
        [pitch, roll, yaw] = rotationMatrixToEulerAngles(self.DCM)
        # self.yaw.append(atan(self.DCM[1,0]/self.DCM[0,0]) * 180/pi)
        # self.pitch.append(-asin(self.DCM[2,0]) * 180/pi)
        # self.roll.append(atan(self.DCM[2,1]/self.DCM[2,2]) * 180/pi)
        self.yaw.append(yaw * 180/pi)
        self.pitch.append(pitch * 180/pi)
        self.roll.append(roll * 180/pi)

@dataclass
class AccelerometerData:
    timestamp: float
    x: float
    y: float
    z: float

@dataclass
class GyroscopeData:
    timestamp: float
    pitch: float
    roll: float
    yaw: float

@dataclass
class SimulateINS(INS):
    accel: list[AccelerometerData] = field(default_factory=list)
    gyro: list[GyroscopeData] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            with open('./iphone_data.txt', 'r') as dat: 
                data = json.loads(dat.read())
            accel_data = sorted(data['accel'], key=lambda d: d['timestamp']) 
            gyro_data = sorted(data['gyro'], key=lambda d: d['timestamp']) 
            [self.accel.append(AccelerometerData(d['timestamp'], d['x'], d['y'], d['z'])) for d in accel_data]
            [self.gyro.append(GyroscopeData(d['timestamp'], d['pitch'], d['roll'], d['yaw'])) for d in gyro_data]
        except Exception as e:
            print(f"INIT ERROR: {e}")
            return

    def simulate(self):
        # Gyro
        for i in range(1, len(self.gyro)):
            vYaw = self.gyro[i].yaw
            vPitch = self.gyro[i].pitch
            vRoll = self.gyro[i].roll
            dT = self.gyro[i].timestamp - self.gyro[i-1].timestamp
            self.updateAngles(vYaw, vPitch, vRoll, dT)
        

ins = SimulateINS(0,0,0)
ins.simulate()
print(f'Yaw: {ins.yaw}')
print(f'Pitch: {ins.pitch}')
print(f'Roll: {ins.roll}')

    




