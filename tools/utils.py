import carla
import numpy as np
import cv2
import threading
import queue
import time



def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert angles to radians if they are in degrees
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Final rotation matrix (adjust order if needed)
    R = R_z @ R_y @ R_x
    return R


def cam_bp_maker(bp_lib, width, height, fov):
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{width}')
    camera_bp.set_attribute('image_size_y', f'{height}')
    camera_bp.set_attribute('fov', f'{fov}')
    return camera_bp