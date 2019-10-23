import cv2
import numpy as np
import os
from utils.data_fun import load_frames_from_folder_as_npy


# b = np.load('data/kinetic-samples/v_CricketShot_g04_c01_flow.npy')


folder_frame = "data/frames_data/v_ApplyEyeMakeup_g01_c04"

a = load_frames_from_folder_as_npy(folder_frames=folder_frame)

print(a)


