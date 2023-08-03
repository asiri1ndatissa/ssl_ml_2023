import mediapipe as mp
import cv2
import numpy as np

import torch
import torch.nn as nn
import subprocess
import os
import sys
from queue import Queue



ROWS_PER_FRAME = 543

class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        LEN = x.shape[0]

        return x.reshape(LEN, -1)
    
feature_converter = FeatureGen()

def load_relevant_data_subset(data):
    n_frames = int(len(data) / ROWS_PER_FRAME)
    print('frames',n_frames)
    data = data.reshape(n_frames, ROWS_PER_FRAME, 3)
    return data.astype(np.float32)


def convert_row(npArray, label, signer_id):
    x = load_relevant_data_subset(npArray)
    return feature_converter(torch.tensor(x)).cpu().numpy(), label, signer_id

def convert_and_save_test_data(npArray):
    data_list = []
    data, label, signer_id = convert_row(npArray, 20, 3)
    data_list.append({'data': data, 'label': label, 'signer_id': signer_id})
    return np.array(data_list)

class KeypointsProcessor:
    def __init__(self):
        self.xyz_array = np.empty((0, 3))
        self.mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False)

    def retriveFrameQue(self, frameQue):
        self.frameQue = frameQue
        for frame in self.frameQue.queue:
            self.keyPointstoNP(frame)
        return self.xyz_array.size

    def keyPointstoNP(self, frame):
        frame_count = 0

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_holistic.process(image)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
              # Extract face keypoints
              if results.face_landmarks is not None:
                  for idx, landmark in enumerate(results.face_landmarks.landmark):
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      new_point = np.array([[capped_x, capped_y, capped_z]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

              if results.face_landmarks is None:
                  for i in range(468):
                      new_point = np.array([[0, 0, 0]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

              if results.left_hand_landmarks is not None:
                  for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      new_point = np.array([[capped_x, capped_y, capped_z]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

              if results.left_hand_landmarks is None:
                  for i in range(21):
                      new_point = np.array([[0, 0, 0]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

              # Extract pose keypoints
              if results.pose_landmarks is not None:
                  for idx, landmark in enumerate(results.pose_landmarks.landmark):
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      new_point = np.array([[capped_x, capped_y, capped_z]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

              if results.pose_landmarks is None:
                  for i in range(33):
                      new_point = np.array([[0, 0, 0]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

              # Extract right hand keypoints
              if results.right_hand_landmarks is not None:
                  for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      new_point = np.array([[capped_x, capped_y, capped_z]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

              if results.right_hand_landmarks is None:
                  for i in range(21):
                      new_point = np.array([[0, 0, 0]])  # Create a new point as a NumPy array
                      self.xyz_array = np.append(self.xyz_array, new_point, axis=0)  # Append the new point to the array

        frame_count += 1

    def get_xyz_array(self):
        return convert_and_save_test_data(self.xyz_array)
